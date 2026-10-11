"""Command line: run a session, evaluate scenarios, list plugins.

  python -m duplexagent list
  python -m duplexagent run  -c configs/default.toml -s scenarios/delegate.toml
  python -m duplexagent eval -c configs/default.toml scenarios/*.toml --out results.json
  python -m duplexagent bench fast --data bench_data/fast_delegation.jsonl
  ... --set asr.impl=funasr --set fast.think_ms=50     (override any config key)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from duplexagent.config import build_engine, load_config
from duplexagent.core.registry import available
from duplexagent.core.timeline import Timeline


async def run_one(cfg: dict, scenario: str | None, echo: bool, timeline_out: str | None = None) -> dict | None:
    from duplexagent.metrics import score
    from duplexagent.plugins.io.sim import Scenario

    if scenario:
        cfg.setdefault("input", {}).update(impl="sim", scenario=scenario)
    tl = Timeline(echo=echo)
    engine = build_engine(cfg, tl)
    await engine.run()
    if timeline_out:
        tl.dump(timeline_out)
    inp = cfg.get("input", {})
    if inp.get("impl") == "sim":
        sc_src = inp["scenario"]
        sc = Scenario.from_dict(sc_src, inp.get("user_cps", 5.0)) if isinstance(sc_src, dict) else Scenario.load(sc_src, inp.get("user_cps", 5.0))
        return score(tl, sc)
    return None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="duplexagent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list registered plugins")
    r = sub.add_parser("run", help="run one session")
    r.add_argument("-c", "--config", default="configs/default.toml")
    r.add_argument("-s", "--scenario")
    r.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    r.add_argument("--timeline", help="write the event log as JSONL")
    r.add_argument("-q", "--quiet", action="store_true")
    e = sub.add_parser("eval", help="run scenarios and score them")
    e.add_argument("scenarios", nargs="+")
    e.add_argument("-c", "--config", default="configs/default.toml")
    e.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    e.add_argument("--out", help="write results JSON here")
    e.add_argument("--serial", action="store_true", help="run scenarios one by one (default: concurrently)")
    b = sub.add_parser("bench", help="offline benchmark of one module")
    b.add_argument("module", choices=["asr", "fast", "slow", "tts"])
    b.add_argument("--data", required=True, help="JSONL dataset")
    b.add_argument("-c", "--config", default="configs/default.toml")
    b.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    b.add_argument("--out", help="write results JSON here")
    b.add_argument("--realtime", action="store_true", help="asr: feed audio at real-time pace")
    b.add_argument("--wav-dir", help="tts: write synthesized audio here")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        for kind, names in available().items():
            print(f"{kind:<7} {', '.join(names)}")
        return
    if a.cmd == "run":
        m = asyncio.run(run_one(load_config(a.config, a.set), a.scenario, not a.quiet, a.timeline))
        if m:
            print(json.dumps(m, ensure_ascii=False, indent=2))
        return
    if a.cmd == "bench":
        from duplexagent.bench.offline import BENCHES, load_jsonl

        cfg = load_config(a.config, a.set)
        kwargs = {"asr": {"realtime": a.realtime}, "tts": {"wav_dir": a.wav_dir}}.get(a.module, {})
        res = asyncio.run(BENCHES[a.module](cfg, load_jsonl(a.data), **kwargs))
        summary = {k: v for k, v in res.items() if k != "rows"}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump({"config": a.config, "overrides": a.set, **res}, f, ensure_ascii=False, indent=2)
        return

    async def run_all() -> list[dict]:
        if a.serial:
            return [await run_one(load_config(a.config, a.set), s, False) for s in a.scenarios]
        return await asyncio.gather(*(run_one(load_config(a.config, a.set), s, False) for s in a.scenarios))

    results = asyncio.run(run_all())
    for m in results:
        flag = "PASS" if m["passed"] else "FAIL"
        failed = [k for k, ok in m["checks"].items() if not ok]
        print(f"{flag} {m['scenario']:<22} latency_p50={m['latency_p50']} barge={m['barge_in_detected']}/{m['barge_in_expected']} "
              f"false_barge={m['false_barge_in']} tasks={m['tasks_delivered']}/{m['tasks_spawned']} {failed or ''}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"config": a.config, "overrides": a.set, "results": results}, f, ensure_ascii=False, indent=2)
    sys.exit(0 if all(m["passed"] for m in results) else 1)
