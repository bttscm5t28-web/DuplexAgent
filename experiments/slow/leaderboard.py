"""Print the slow-brain leaderboard (markdown) from experiments/slow/results/*.json.

    python experiments/slow/leaderboard.py [--results experiments/slow/results]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(v, pct=False, nd=1):
    if v is None:
        return "—"
    return f"{v * 100:.{nd}f}%" if pct else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).parent / "results"))
    a = ap.parse_args()
    rows = []
    for p in sorted(Path(a.results).glob("*.json")):
        d = json.loads(p.read_text())
        t, am, ca = d.get("tasks", {}), d.get("amend", {}), d.get("cancel", {})
        rows.append((t.get("accuracy") or 0, d["name"], t, am, ca))
    rows.sort(key=lambda r: -r[0])
    print("| 候选 | 正确率 | 计算/单位/日期 | 事实 | 推理/编程 | 规划 | p50 耗时 s | p90 耗时 s | 首次进度 s | 进度/任务 | 有信息量进度占比 | 最长静默 p50 s | 工具调用/任务 | 工具成功率 | 补充采纳 | 取消延迟 p50 ms | 残留进程 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, name, t, am, ca in rows:
        c = t.get("accuracy_by_cat", {})

        def avg(*ks):
            v = [c[k] for k in ks if k in c]
            return sum(v) / len(v) if v else None

        adopt = am.get("adopted_rate_running")
        adopt_s = f"{am.get('adopted_rate_all', 0) * 100:.0f}% ({fmt(adopt, True, 0)} 运行中)" if am else "—"
        lat = ca.get("cancel_latency_p50_s")
        print(f"| {name} | {fmt(t.get('accuracy'), True)} | {fmt(avg('calc', 'unit', 'date'), True, 0)} | {fmt(c.get('fact'), True, 0)} | "
              f"{fmt(avg('reason', 'prog'), True, 0)} | {fmt(c.get('plan'), True, 0)} | {fmt(t.get('latency_p50_s'))} | {fmt(t.get('latency_p90_s'))} | "
              f"{fmt(t.get('first_progress_p50_s'), nd=2)} | {fmt(t.get('progress_per_task'))} | {fmt(t.get('informative_progress_ratio'), True, 0)} | "
              f"{fmt(t.get('max_silence_p50_s'))} | {fmt(t.get('tool_calls_per_task'), nd=2)} | {fmt(t.get('tool_success_rate'), True, 0)} | {adopt_s} | "
              f"{fmt(lat * 1000 if lat is not None else None)} | {ca.get('leftover_procs', '—')} |")


if __name__ == "__main__":
    main()
