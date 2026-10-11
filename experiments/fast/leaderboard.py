"""汇总 experiments/fast/results/bench_*.json 与 e2e_*.json，打印 Markdown 排行榜。

    python experiments/fast/leaderboard.py [results_dir]
"""

from __future__ import annotations

import glob
import json
import os
import sys


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}"


def sec(x):
    return "-" if x is None else f"{x:.2f}"


def main() -> None:
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results")
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "bench_*.json"))):
        r = json.load(open(f, encoding="utf-8"))
        name = os.path.basename(f)[6:-5]
        dl, ho = r.get("delegation", {}), r.get("heldout", {})
        fu, fh = r.get("fusion", {}), r.get("fusion_heldout", {})
        # say-turn TTFT: what the user waits for when the fast brain answers itself
        say_ttft = sorted(x["ttft"] for x in dl.get("rows", []) + ho.get("rows", []) if x["got"] == "say" and x["ttft"] is not None)
        all_ttft = sorted(x["ttft"] for x in dl.get("rows", []) + ho.get("rows", []) if x["ttft"] is not None)
        q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None  # noqa: E731
        rows.append((name, dl, ho, fu, fh, q(all_ttft, 0.5), q(all_ttft, 0.9), q(say_ttft, 0.5), q(say_ttft, 0.9)))
    rows.sort(key=lambda r: -(r[2].get("action_accuracy") or 0))
    print("| 候选 | 委派准确率 主/留出 | 精确率 留出 | 召回 留出 | 动作准确率 主/留出 | 瞎猜违规 | 融合通过 主/留出 | TTFT p50/p90 全部 | TTFT p50/p90 自答轮 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, dl, ho, fu, fh, a50, a90, s50, s90 in rows:
        forbid = [x for x in (dl.get("forbid_violation_rate"), ho.get("forbid_violation_rate")) if x is not None]
        print(
            f"| {name} | {pct(dl.get('delegation_accuracy'))} / {pct(ho.get('delegation_accuracy'))} | {pct(ho.get('delegation_precision'))} | "
            f"{pct(ho.get('delegation_recall'))} | {pct(dl.get('action_accuracy'))} / {pct(ho.get('action_accuracy'))} | "
            f"{pct(max(forbid) if forbid else None)} | {pct(fu.get('fusion_all'))} / {pct(fh.get('fusion_all'))} | "
            f"{sec(a50)} / {sec(a90)} | {sec(s50)} / {sec(s90)} |"
        )
    e2e = sorted(glob.glob(os.path.join(d, "e2e_*.json")))
    if e2e:
        print("\n| 候选 | 场景通过 | 未通过的检查 | 端到端延迟 p50 均值 |")
        print("|---|---|---|---|")
        for f in e2e:
            r = json.load(open(f, encoding="utf-8"))["results"]
            lat = [m["latency_p50"] for m in r if m["latency_p50"] is not None]
            fails = [f"{m['scenario']}:{k}" for m in r for k, ok in m["checks"].items() if not ok]
            print(f"| {os.path.basename(f)[4:-5]} | {sum(m['passed'] for m in r)}/{len(r)} | {', '.join(fails) or '-'} | {sec(sum(lat) / len(lat) if lat else None)} |")


if __name__ == "__main__":
    main()
