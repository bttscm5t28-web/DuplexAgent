"""Summarize a live session's timeline (python -m duplexagent run ... --timeline x.jsonl).

    python scripts/live_summary.py results/live/2026-10-12_mac.jsonl

Prints one line per turn and the session's latency / interruption / task numbers. Pair it with
the mic recording (`input.record_path`) to label what *should* have happened.
"""

from __future__ import annotations

import json
import statistics
import sys


def main(path: str) -> None:
    ev = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    lat, rows = [], []
    user_text = ""
    for e in ev:
        if e["name"] == "turn_end":
            user_text = e.get("text", "")
        elif e["name"] == "agent_audio_start" and e.get("latency") is not None:
            lat.append(e["latency"])
        elif e["name"] == "agent_end":
            rows.append((e["t"], user_text, e.get("played", ""), e.get("interrupted")))
            user_text = ""
    for t, u, a, cut in rows:
        print(f"[{t:7.2f}] 用户: {u or '（慢脑结果）'}\n          助手: {a}{' ✂' if cut else ''}")
    n = lambda name: sum(1 for e in ev if e["name"] == name)
    ends = {e["id"]: e["t"] for e in ev if e["name"] == "task_end"}
    waits = [r["t"] - ends[t] for r in ev if r["name"] == "respond" for t in r.get("deliver", []) if t in ends]
    q = lambda xs, p: round(sorted(xs)[min(len(xs) - 1, int(p * len(xs)))], 3) if xs else None
    print()
    print(f"响应延迟 p50/p90: {q(lat, .5)} / {q(lat, .9)} s  (n={len(lat)}, mean={round(statistics.mean(lat), 3) if lat else None})")
    print(f"用户轮次 {n('turn_end')} · 打断 {n('barge_in')} · 附和 {n('backchannel')} · 中断回复 {n('interrupt')}")
    print(f"慢脑任务 {n('task_spawn')} · 融合 {len(waits)} · 结果等待融合 p50 {q(waits, .5)} s · 出错 {n('response_error') + n('task_crash')}")


if __name__ == "__main__":
    main(sys.argv[1])
