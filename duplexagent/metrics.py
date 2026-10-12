"""Score a simulated session: latency, turn-taking, interruption and fusion metrics."""

from __future__ import annotations

import statistics
from typing import Any

from duplexagent.core.timeline import Timeline
from duplexagent.plugins.io.sim import Scenario


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 3)


def score(tl: Timeline, sc: Scenario) -> dict[str, Any]:
    lat = [e.data["latency"] for e in tl.of("agent_audio_start") if e.data.get("latency") is not None]
    interrupts = [e for e in tl.of("interrupt") if e.data["reason"] in ("barge_in", "new_turn")]
    played = "".join(e.data["played"] for e in tl.of("agent_end"))

    def interrupted_during(u, slack=0.8) -> list:
        return [e for e in interrupts if u.at <= e.t <= u.end + slack]

    barge = [u for u in sc.utterances if u.kind == "barge_in"]
    barge_hits = [(u, interrupted_during(u)) for u in barge]
    backch = [u for u in sc.utterances if u.kind == "backchannel"]
    false_barge = sum(1 for u in backch if interrupted_during(u))

    spawned = {e.data["id"]: e.t for e in tl.of("task_spawn")}
    ended = {e.data["id"]: e.t for e in tl.of("task_end")}
    delivered: dict[str, float] = {}
    for e in tl.of("respond"):
        for tid in e.data.get("deliver", []):
            delivered.setdefault(tid, e.t)
    fusion_wait = [delivered[t] - ended[t] for t in delivered if t in ended]

    m = {
        "scenario": sc.name,
        "latency_p50": _pct(lat, 0.5),
        "latency_p90": _pct(lat, 0.9),
        "latency_mean": round(statistics.mean(lat), 3) if lat else None,
        "turns_expected": sum(1 for u in sc.utterances if u.kind in ("turn", "barge_in")),
        # user turns as the conversation saw them: detector turn ends minus those merged into the previous one
        "turns_detected": len(tl.of("turn_end")) - len(tl.of("turn_merged")),
        "barge_in_expected": len(barge),
        "barge_in_detected": sum(1 for _, hits in barge_hits if hits),
        "barge_in_stop_latency": _pct([hits[0].t - u.at for u, hits in barge_hits if hits], 0.5),
        "backchannels": len(backch),
        "false_barge_in": false_barge,
        "tasks_spawned": len(spawned),
        "tasks_delivered": len(delivered),
        "fusion_wait_p50": _pct(fusion_wait, 0.5),
        "errors": len(tl.of("response_error")) + len(tl.of("task_crash")),
        "agent_said": played,
    }
    m["checks"] = check(m, sc.expect)
    m["passed"] = all(m["checks"].values())
    return m


def check(m: dict, expect: dict) -> dict[str, bool]:
    out = {}
    for k, v in expect.items():
        if k == "said_contains":
            for s in v:
                out[f"said_contains:{s}"] = s in m["agent_said"]
        elif k.startswith("max_"):
            val = m.get(k[4:])
            out[k] = val is not None and val <= v
        elif k.startswith("min_"):
            val = m.get(k[4:])
            out[k] = val is not None and val >= v
        else:
            out[k] = m.get(k) == v
    out.setdefault("no_errors", m["errors"] == 0)
    return out
