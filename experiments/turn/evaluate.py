"""Turn-taking / interruption evaluation: runs scenarios against turn-detector configs and
scores them with turn-specific metrics that `duplexagent/metrics.py` does not have.

    python experiments/turn/evaluate.py scenarios/turn/*.toml -c configs/exp/turn/adaptive.toml
    python experiments/turn/evaluate.py scenarios/turn/*.toml -c configs/default.toml -c configs/exp/turn/adaptive.toml \
        --set asr.impl=lagged_mock --out experiments/turn/results/text_lagged.json

Scenario utterance kinds (field `kind`):
  turn         a user turn while the agent is silent; agent should respond
  cont         continuation of the previous utterance after a thinking pause (same turn)
  barge_in     user interrupts the agent mid-speech; agent must stop and take the turn
  backchannel  "嗯/对/好的…" while the agent talks; agent must keep talking
  noise        cough / keyboard / knock…; must neither interrupt nor start a turn

Metrics (all from the timeline, against the scripted ground truth):
  cut_in        a turn whose user was still mid-utterance (thinking pause) when TURN_END fired  (抢话)
  eot_latency   TURN_END time - true end of the user's turn
  resp_latency  agent audio start - true end of the user's turn
  missed_eot    turn with no TURN_END within 3 s after it ended
  barge_*       detection / stop latency for kind=barge_in (stop latency counted from utterance start)
  false_barge   backchannel/noise that interrupted the agent  (误打断)
  false_turn    noise (or a backchannel) that produced a TURN_END
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import statistics
import sys
import time
import tomllib
from dataclasses import dataclass, field

sys.path.insert(0, os.getcwd())

from duplexagent.config import build_engine, load_config  # noqa: E402
from duplexagent.core.timeline import Timeline  # noqa: E402
from duplexagent.metrics import score as base_score  # noqa: E402
from duplexagent.plugins.io.sim import Scenario  # noqa: E402

BARGE_SLACK = 0.8  # same as metrics.py
EOT_WINDOW = 3.0


@dataclass
class Group:
    kind: str
    pieces: list = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.pieces[0].at

    @property
    def end(self) -> float:
        return self.pieces[-1].end

    @property
    def text(self) -> str:
        return "".join(p.text for p in self.pieces)


def group_utterances(sc: Scenario) -> list[Group]:
    groups: list[Group] = []
    for u in sc.utterances:
        if u.kind == "cont" and groups:
            groups[-1].pieces.append(u)
        else:
            groups.append(Group(u.kind, [u]))
    return groups


def agent_speaking_intervals(tl: Timeline) -> list[tuple[float, float]]:
    out, start = [], None
    for e in tl.events:
        if e.name == "agent_audio_start":
            start = e.t
        elif e.name == "agent_end" and start is not None:
            out.append((start, e.t))
            start = None
    if start is not None:
        out.append((start, 1e9))
    return out


def turn_score(tl: Timeline, sc: Scenario) -> dict:
    groups = group_utterances(sc)
    turn_ends = [e.t for e in tl.of("turn_end")]
    interrupts = [e.t for e in tl.of("interrupt") if e.data["reason"] in ("barge_in", "new_turn")]
    audio_starts = [e.t for e in tl.of("agent_audio_start")]
    speaking = agent_speaking_intervals(tl)

    def agent_talking(t: float) -> bool:
        return any(a <= t <= b for a, b in speaking)

    rows = []
    for i, g in enumerate(groups):
        nxt = groups[i + 1].start if i + 1 < len(groups) else 1e9
        row = {"kind": g.kind, "text": g.text, "start": round(g.start, 2), "end": round(g.end, 2)}
        if g.kind in ("turn", "barge_in"):
            # cut-in: TURN_END while the user still had more to say (before the last piece started)
            row["paused"] = len(g.pieces) > 1
            row["cut_in"] = any(g.start < t < g.pieces[-1].at + 0.05 for t in turn_ends)
            after = [t for t in turn_ends if g.end - 0.05 <= t <= min(g.end + EOT_WINDOW, nxt + 0.05)]
            row["eot_latency"] = round(after[0] - g.end, 3) if after else None
            row["missed_eot"] = not after
            resp = [t for t in audio_starts if g.end <= t <= min(g.end + 5.0, nxt)]
            row["resp_latency"] = round(resp[0] - g.end, 3) if resp else None
        if g.kind == "barge_in":
            row["agent_was_talking"] = agent_talking(g.start)
            hits = [t for t in interrupts if g.start <= t <= g.end + BARGE_SLACK]
            row["barge_detected"] = bool(hits)
            row["barge_stop_latency"] = round(hits[0] - g.start, 3) if hits else None
        if g.kind in ("backchannel", "noise"):
            row["agent_was_talking"] = agent_talking(g.start)
            row["false_barge"] = any(g.start <= t <= g.end + BARGE_SLACK for t in interrupts)
            # a backchannel said after the agent already stopped is a legitimate turn; noise never is
            counts = g.kind == "noise" or row["agent_was_talking"]
            row["false_turn"] = counts and any(g.start <= t <= g.end + 1.5 for t in turn_ends)
        rows.append(row)
    return {"scenario": sc.name, "groups": rows}


def _pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 3)


def aggregate(per_scenario: list[dict]) -> dict:
    rows = [r for s in per_scenario for r in s["turn"]["groups"]]
    turns = [r for r in rows if r["kind"] in ("turn", "barge_in")]
    paused = [r for r in turns if r.get("paused")]
    barges = [r for r in rows if r["kind"] == "barge_in"]
    valid_barges = [r for r in barges if r["agent_was_talking"]]
    distractors = [r for r in rows if r["kind"] in ("backchannel", "noise")]
    during_agent = [r for r in distractors if r["agent_was_talking"]]
    eot = [r["eot_latency"] for r in turns if r["eot_latency"] is not None and not r["cut_in"]]
    resp = [r["resp_latency"] for r in turns if r["resp_latency"] is not None and not r["cut_in"] and r["kind"] == "turn"]
    stop = [r["barge_stop_latency"] for r in valid_barges if r["barge_stop_latency"] is not None]

    def rate(n, d):
        return round(n / d, 3) if d else None

    agg = {
        "scenarios": len(per_scenario),
        "scenarios_passed": sum(1 for s in per_scenario if s["base"]["passed"]),
        "turns": len(turns),
        "paused_turns": len(paused),
        "cut_in": sum(r["cut_in"] for r in turns),
        "cut_in_rate": rate(sum(r["cut_in"] for r in paused), len(paused)),
        "missed_eot": sum(r["missed_eot"] for r in turns),
        "barge_cases": len(valid_barges),
        "barge_invalid": len(barges) - len(valid_barges),
        "missed_barge": sum(not r["barge_detected"] for r in valid_barges),
        "missed_barge_rate": rate(sum(not r["barge_detected"] for r in valid_barges), len(valid_barges)),
        "distractors": len(during_agent),
        "false_barge": sum(r["false_barge"] for r in during_agent),
        "false_barge_rate": rate(sum(r["false_barge"] for r in during_agent), len(during_agent)),
        "false_turn": sum(r["false_turn"] for r in distractors),
        "eot_p50": _pct(eot, 0.5),
        "eot_p90": _pct(eot, 0.9),
        "resp_p50": _pct(resp, 0.5),
        "resp_p90": _pct(resp, 0.9),
        "stop_p50": _pct(stop, 0.5),
        "stop_p90": _pct(stop, 0.9),
    }
    # One number for ranking: error rates dominate, latency breaks ties.
    # errors: each kind of mistake counts per opportunity; latency: seconds over a 0.3 s floor.
    err = (agg["cut_in_rate"] or 0) + (agg["missed_barge_rate"] or 0) + (agg["false_barge_rate"] or 0)
    err += agg["false_turn"] / max(1, len(distractors)) + agg["missed_eot"] / max(1, len(turns))
    lat = max(0.0, (agg["eot_p50"] or 3.0) - 0.3) + max(0.0, (agg["stop_p50"] or 3.0) - 0.3)
    agg["score"] = round(err + 0.5 * lat, 3)
    return agg


async def run_scenario(cfg: dict, path: str) -> dict:
    cfg = json.loads(json.dumps(cfg))
    cfg.setdefault("input", {}).update(impl="sim", scenario=path)
    tl = Timeline()
    engine = build_engine(cfg, tl)
    await engine.run()
    sc = Scenario.load(path, cfg["input"].get("user_cps", 5.0))
    base = base_score(tl, sc)
    turn = turn_score(tl, sc)
    events = [{"t": round(e.t, 3), "name": e.name, **{k: v for k, v in e.data.items() if k != "modules"}}
              for e in tl.events if e.name not in ("asr_partial",)]
    return {"scenario": sc.name, "path": path, "base": base, "turn": turn, "events": events}


async def run_config(cfg: dict, paths: list[str], concurrency: int) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)

    async def one(p):
        async with sem:
            return await run_scenario(cfg, p)

    return await asyncio.gather(*(one(p) for p in paths))


def print_table(rows: list[tuple[str, dict]]) -> None:
    cols = ["score", "scenarios_passed", "cut_in_rate", "missed_barge_rate", "false_barge_rate", "false_turn",
            "missed_eot", "eot_p50", "eot_p90", "resp_p50", "stop_p50", "stop_p90"]
    head = f"{'config':<28}" + "".join(f"{c:>18}" for c in cols)
    print(head)
    for name, a in sorted(rows, key=lambda r: r[1]["score"]):
        cells = []
        for c in cols:
            v = a[c]
            if c == "scenarios_passed":
                v = f"{v}/{a['scenarios']}"
            cells.append(f"{str(v):>18}")
        print(f"{name:<28}" + "".join(cells))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="+")
    ap.add_argument("-c", "--config", action="append", required=True, help="repeatable: compare several configs")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="applied to every config")
    ap.add_argument("--name", action="append", help="display names for the configs (same order)")
    ap.add_argument("-j", "--concurrency", type=int, default=8)
    ap.add_argument("--out", help="write full results JSON")
    ap.add_argument("--verbose", "-v", action="store_true", help="print per-group rows")
    a = ap.parse_args(argv)

    paths = sorted({p for s in a.scenarios for p in glob.glob(s)})
    names = a.name or [os.path.splitext(os.path.basename(c))[0] for c in a.config]
    table, full = [], {}
    for name, cpath in zip(names, a.config):
        cfg = load_config(cpath, a.set)
        if any(s.startswith("asr.impl=lagged_mock") for s in a.set) or cfg.get("asr", {}).get("impl") == "lagged_mock":
            mods = cfg.setdefault("plugins", {}).setdefault("modules", [])
            if "experiments.turn.lagged_asr" not in mods:
                mods.append("experiments.turn.lagged_asr")
        t0 = time.time()
        res = asyncio.run(run_config(cfg, paths, a.concurrency))
        agg = aggregate(res)
        agg["wall_s"] = round(time.time() - t0, 1)
        table.append((name, agg))
        full[name] = {"config": cpath, "overrides": a.set, "aggregate": agg, "scenarios": res}
        if a.verbose:
            for s in res:
                bad = [k for k, ok in s["base"]["checks"].items() if not ok]
                print(f"  [{name}] {s['scenario']}: {'PASS' if s['base']['passed'] else 'FAIL ' + str(bad)}")
                for r in s["turn"]["groups"]:
                    print("     ", {k: v for k, v in r.items() if v not in (None, False) or k in ("barge_detected",)})
    print_table(table)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"aggregate": {n: v["aggregate"] for n, v in full.items()}, "runs": full}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
