"""Print a Markdown leaderboard from experiments/tts/results/*.json (eval_tts.py outputs).

    python experiments/tts/leaderboard.py [results/*.json ...]
"""

from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def fmt(v, nd=3):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main(paths: list[str]) -> None:
    paths = paths or sorted(p for p in glob.glob(os.path.join(HERE, "results", "*.json"))
                            if not os.path.basename(p).startswith(("bench_", "sweep_", "e2e_")))
    rows = []
    for p in paths:
        s = json.load(open(p, encoding="utf-8"))["summary"]
        name = os.path.basename(p).removesuffix(".json")
        rows.append((name, s))
    rows.sort(key=lambda r: (r[1]["cer"] if r[1].get("cer") is not None else 9))
    print("| 候选 | n | 首包 p50 / p90 (s) | RTF p50 | CER | UTMOS | 对齐命中 | 截断误差(字) | 流式首包 p50 (s) | 卡顿次数 | 取消→空闲 p90 (s) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, s in rows:
        print(f"| {name} | {s['n']} | {fmt(s['first_chunk_s']['p50'])} / {fmt(s['first_chunk_s']['p90'])} | {fmt(s['rtf']['p50'])} | "
              f"{fmt(s.get('cer'), 4)} | {fmt(s.get('utmos'), 2)} | {fmt(s.get('align_hit'), 3)} | {fmt(s.get('trunc_err_chars'), 2)} | "
              f"{fmt((s.get('stream_first_s') or {}).get('p50'))} | {fmt(s.get('stalls'))} | {fmt(s['cancel_idle_s']['p90'])} |")
    print()
    cats = sorted({c for _, s in rows for c in s.get("cer_by_cat", {})})
    print("| 候选 | " + " | ".join(f"CER {c}" for c in cats) + " |")
    print("|---|" + "---|" * len(cats))
    for name, s in rows:
        print(f"| {name} | " + " | ".join(fmt(s.get("cer_by_cat", {}).get(c), 4) for c in cats) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
