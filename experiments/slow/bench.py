"""Slow-brain benchmark: accuracy, latency, progress quality, tool success, amendments, cancellation.

    python experiments/slow/bench.py -c configs/exp/slow/router.toml --name router
    python experiments/slow/bench.py -c configs/exp/slow/agent_qwen3_4b.toml --name agent_qwen3_4b --parts tasks,amend,cancel
    ... --set slow.max_steps=3 --limit 10 --cats calc,prog

Writes experiments/slow/results/<name>.json and prints a summary. Datasets live in bench_data/slow/:
  tasks.jsonl       {query, cat, answer_contains[], accept_any[[alt…]…], reject_any[], rubric{keywords[[alt…]…], min_hits}}
  amendments.jsonl  {query, amend_after_s, amendment, answer_contains[], stale[]}
  cancel.jsonl      {query, cancel_after_s}
`answer_contains` keeps the core `bench slow` format, so the same file also works with
`python -m duplexagent bench slow --data bench_data/slow/tasks.jsonl` (that grader ignores the extras).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from duplexagent.bench.offline import load_jsonl, normalize  # noqa: E402
from duplexagent.config import load_config  # noqa: E402
from duplexagent.core.registry import create, load_modules  # noqa: E402
from duplexagent.core.types import TaskRecord  # noqa: E402

FIXED_NOW = "2026-10-11T10:00"  # date tasks are written relative to this "today"
GENERIC = re.compile(r"^(开始处理|已开始处理，正在分析问题|处理中|完成了第 ?\d+/\d+ ?步|开始推理|正在思考)$")


_CN_BIG = re.compile(r"(\d+(?:\.\d+)?)亿(?:(\d{1,4})万)?(\d{1,4})?|(\d+(?:\.\d+)?)万(\d{1,4})?")


def expand_cn_units(s: str) -> str:
    """'700万6652' -> '7006652', '1.5万' -> '15000' (spoken-style numbers count as correct)."""
    def sub(m: re.Match) -> str:
        if m.group(1):
            v = float(m.group(1)) * 1e8 + int(m.group(2) or 0) * 1e4 + int(m.group(3) or 0)
        else:
            v = float(m.group(4)) * 1e4 + int(m.group(5) or 0)
        return str(int(v)) if v == int(v) else str(v)
    return _CN_BIG.sub(sub, s)


def grade(item: dict, result: str, detail: str = "") -> bool | None:
    if item.get("rubric"):  # plans are judged on the full answer (spoken summary + detailed plan)
        result = result + "\n" + detail
    result = expand_cn_units(result)
    r = normalize(result)
    checks = []
    if item.get("answer_contains"):
        checks.append(all(normalize(w) in r for w in item["answer_contains"]))
    for group in item.get("accept_any", []):
        checks.append(any(normalize(w) in r for w in group))
    if item.get("reject_any"):
        # reject phrases are checked on the raw text so "不是闰年" is not confused with "是闰年"
        checks.append(not any(w in result for w in item["reject_any"]))
    if item.get("rubric"):
        rb = item["rubric"]
        hits = sum(any(normalize(w) in r for w in g) for g in rb["keywords"])
        checks.append(hits >= rb.get("min_hits", len(rb["keywords"])))
    return all(checks) if checks else None


def informative(p: str) -> bool:
    p = p.strip()
    return bool(p) and not GENERIC.match(p) and not re.match(r"^开始处理[:：]", p)


def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(q * len(s)))], 3)


def _inject_now(cfg: dict) -> None:
    s = cfg["slow"]
    if s.get("impl") in ("agent", "claude", "claude_agent", "router", "hybrid"):
        s.setdefault("fixed_now", FIXED_NOW)
    if isinstance(s.get("inner"), dict) and s["inner"].get("impl") in ("agent", "claude", "claude_agent"):
        s["inner"].setdefault("fixed_now", FIXED_NOW)


async def run_task(brain, item: dict, amend: tuple[float, str] | None = None, cancel_after: float | None = None, timeout: float = 300.0) -> dict:
    rec = TaskRecord(query=item["query"], context=item.get("context", ""))
    t0 = time.perf_counter()
    ev: list[tuple[float, str, str]] = []

    async def consume():
        async for u in brain.run(rec):
            ev.append((time.perf_counter() - t0, u.kind, u.text))

    runner = asyncio.create_task(consume())
    row: dict = {"id": item.get("id"), "cat": item.get("cat"), "query": item["query"]}
    try:
        if amend:
            done, _ = await asyncio.wait({runner}, timeout=amend[0])
            row["amend_while_running"] = not done
            if not done:
                rec.amendments.append(amend[1])
                row["amend_at"] = round(time.perf_counter() - t0, 3)
        if cancel_after is not None:
            done, _ = await asyncio.wait({runner}, timeout=cancel_after)
            row["finished_before_cancel"] = bool(done)
            if not done:
                tc = time.perf_counter()
                runner.cancel()
                await asyncio.gather(runner, return_exceptions=True)
                row["cancel_latency_s"] = round(time.perf_counter() - tc, 4)
                row["leftover_sandbox_procs"] = sandbox_procs()
        await asyncio.wait_for(asyncio.shield(runner), timeout) if not runner.done() else None
    except asyncio.TimeoutError:
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)
        ev.append((time.perf_counter() - t0, "error", "TIMEOUT"))
    except Exception as e:
        ev.append((time.perf_counter() - t0, "error", f"CRASH {type(e).__name__}: {e}"))
    if runner.done() and not runner.cancelled() and runner.exception() is not None:
        e = runner.exception()
        ev.append((time.perf_counter() - t0, "error", f"CRASH {type(e).__name__}: {e}"))
    end = time.perf_counter() - t0
    result = next((t for _, k, t in reversed(ev) if k in ("result", "error")), "")
    progress = [(t, x) for t, k, x in ev if k == "progress"]
    stamps = [0.0] + [t for t, k, _ in ev] + ([end] if not ev or ev[-1][1] not in ("result", "error") else [])
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    trace = getattr(brain, "traces", {}).get(rec.id, [])
    row.update(
        result=result,
        status=next((k for _, k, _ in reversed(ev) if k in ("result", "error")), "none"),
        latency_s=round(end, 3),
        first_progress_s=round(progress[0][0], 4) if progress else None,
        progress=[x for _, x in progress],
        n_progress=len(progress),
        n_informative=sum(informative(x) for _, x in progress),
        max_silence_s=round(max(gaps), 3) if gaps else None,
        spoken_chars=len(result),
        has_detail=any(k == "partial" and x.startswith("详细") for _, k, x in ev),
        detail=next((x[3:] for _, k, x in ev if k == "partial" and x.startswith("详细")), ""),
        tool_calls=len(trace),
        tool_ok=sum(1 for t in trace if t.get("ok")),
        tools=[t.get("tool") for t in trace],
        trace=trace,
    )
    return row


def sandbox_procs() -> int:
    n = 0
    for p in Path("/proc").iterdir():
        try:
            cmd = (p / "cmdline").read_bytes() if p.name.isdigit() else b""
            n += b"\x00-I\x00-S\x00-c\x00" in cmd and b"_BLOCK_MODS" in cmd
        except OSError:
            pass
    return n


async def main(a) -> dict:
    cfg = load_config(a.config, a.set)
    _inject_now(cfg)
    load_modules(cfg.get("plugins", {}).get("modules", []))
    brain = create("slow", cfg["slow"])
    await brain.start()
    out: dict = {"name": a.name, "config": a.config, "overrides": a.set, "slow_cfg": cfg["slow"], "hardware": hw()}
    sem = asyncio.Semaphore(a.concurrency)

    async def guarded(coro):
        async with sem:
            return await coro

    try:
        parts = a.parts.split(",")
        if "tasks" in parts:
            data = load_jsonl(a.data)
            if a.cats:
                data = [d for d in data if d.get("cat") in a.cats.split(",")]
            data = data[: a.limit] if a.limit else data
            t0 = time.perf_counter()
            rows = await asyncio.gather(*(guarded(run_task(brain, d, timeout=a.timeout)) for d in data))
            for r, d in zip(rows, data):
                r["ok"] = grade(d, r["result"], r["detail"]) if r["status"] == "result" else False
                if a.verbose:
                    print(f"{'OK ' if r['ok'] else 'BAD'} {r['latency_s']:6.1f}s {r['id']:9} {r['query'][:28]:30} -> {r['result'][:70]}", flush=True)
            out["tasks"] = summarize(rows)
            out["tasks"]["wall_s"] = round(time.perf_counter() - t0, 1)
            out["tasks"]["rows"] = rows
        if "amend" in parts:
            data = load_jsonl(a.amend_data)
            rows = []
            for d in data[: a.limit or None]:
                r = await run_task(brain, d, amend=(d["amend_after_s"], d["amendment"]), timeout=a.timeout)
                res = normalize(expand_cn_units(r["result"]))
                r["adopted"] = r["status"] == "result" and all(normalize(w) in res for w in d["answer_contains"])
                r["stale"] = r["status"] == "result" and any(normalize(w) in res for w in d.get("stale", []))
                r["acknowledged"] = any("补充" in p for p in r["progress"])
                rows.append(r)
                if a.verbose:
                    print(f"AMEND running={r['amend_while_running']} adopted={r['adopted']} {r['query'][:24]} -> {r['result'][:60]}", flush=True)
            running = [r for r in rows if r["amend_while_running"]]
            out["amend"] = {
                "n": len(rows),
                "amend_while_running": len(running),
                "adopted_rate_all": round(sum(r["adopted"] for r in rows) / max(1, len(rows)), 3),
                "adopted_rate_running": round(sum(r["adopted"] for r in running) / len(running), 3) if running else None,
                "stale_rate_running": round(sum(r["stale"] and not r["adopted"] for r in running) / len(running), 3) if running else None,
                "acknowledged_rate_running": round(sum(r["acknowledged"] for r in running) / len(running), 3) if running else None,
                "rows": rows,
            }
        if "cancel" in parts:
            data = load_jsonl(a.cancel_data)
            rows = [await run_task(brain, d, cancel_after=d["cancel_after_s"], timeout=a.timeout) for d in data[: a.limit or None]]
            lat = [r["cancel_latency_s"] for r in rows if "cancel_latency_s" in r]
            out["cancel"] = {
                "n": len(rows),
                "cancelled_while_running": len(lat),
                "cancel_latency_p50_s": pct(lat, 0.5),
                "cancel_latency_max_s": round(max(lat), 4) if lat else None,
                "leftover_procs": sum(r.get("leftover_sandbox_procs", 0) for r in rows),
                "rows": rows,
            }
    finally:
        await brain.close()
        for attr in ("usage", "cost_usd"):
            if hasattr(brain, attr):
                out[attr] = getattr(brain, attr)
    return out


def summarize(rows: list[dict]) -> dict:
    graded = [r for r in rows if r["ok"] is not None]
    by_cat: dict[str, list] = {}
    for r in graded:
        by_cat.setdefault(r["cat"] or "?", []).append(r["ok"])
    calls = sum(r["tool_calls"] for r in rows)
    lat = [r["latency_s"] for r in rows]
    fp = [r["first_progress_s"] for r in rows if r["first_progress_s"] is not None]
    prog = [r["n_progress"] for r in rows]
    inf = sum(r["n_informative"] for r in rows)
    return {
        "n": len(rows),
        "accuracy": round(sum(r["ok"] for r in graded) / len(graded), 4) if graded else None,
        "accuracy_by_cat": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_cat.items())},
        "errors": sum(r["status"] != "result" for r in rows),
        "latency_p50_s": pct(lat, 0.5),
        "latency_p90_s": pct(lat, 0.9),
        "latency_mean_s": round(statistics.mean(lat), 3) if lat else None,
        "first_progress_p50_s": pct(fp, 0.5),
        "progress_per_task": round(statistics.mean(prog), 2) if prog else None,
        "informative_progress_ratio": round(inf / max(1, sum(prog)), 3),
        "tasks_with_informative_progress": round(sum(r["n_informative"] > 0 for r in rows) / max(1, len(rows)), 3),
        "max_silence_p50_s": pct([r["max_silence_s"] for r in rows if r["max_silence_s"] is not None], 0.5),
        "max_silence_p90_s": pct([r["max_silence_s"] for r in rows if r["max_silence_s"] is not None], 0.9),
        "tool_calls_per_task": round(calls / max(1, len(rows)), 2),
        "tool_success_rate": round(sum(r["tool_ok"] for r in rows) / calls, 3) if calls else None,
        "spoken_chars_mean": round(statistics.mean(r["spoken_chars"] for r in rows), 1) if rows else None,
        "detail_rate": round(sum(r["has_detail"] for r in rows) / max(1, len(rows)), 3),
    }


def hw() -> dict:
    import platform

    return {"cpus": os.cpu_count(), "machine": platform.machine(), "python": platform.python_version()}


def cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/default.toml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--name", required=True)
    ap.add_argument("--parts", default="tasks,amend,cancel")
    ap.add_argument("--data", default="bench_data/slow/tasks.jsonl")
    ap.add_argument("--amend-data", default="bench_data/slow/amendments.jsonl")
    ap.add_argument("--cancel-data", default="bench_data/slow/cancel.jsonl")
    ap.add_argument("--cats")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out-dir", default="experiments/slow/results")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)
    out = asyncio.run(main(a))
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(a.out_dir) / f"{a.name}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    brief = {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in out.items() if isinstance(v, dict) and "rows" in v}
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    print(f"-> {path}")


if __name__ == "__main__":
    cli()
