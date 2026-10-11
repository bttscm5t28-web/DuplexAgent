"""Offline benchmark of text end-of-turn scorers on labeled utterances (no engine, no timing noise).

    python experiments/turn/bench_text_eot.py --scorer text --scorer lm --scorer lm15 --scorer ten_turn \
        --out experiments/turn/results/text_eot_bench.json

Data: experiments/turn/bench/text_eot.jsonl — 50 finished / 50 unfinished Chinese utterances written as
minimal pairs (the unfinished one is a prefix of a finished one: "帮我设一个" / "帮我设一个明天早上七点的闹钟").
Reports ROC-AUC, accuracy at 0.5, and the share of unfinished texts confidently called finished (P >= 0.75,
which in the hybrid detector means the shortest timeout = a likely cut-in).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.getcwd())

from duplexagent.plugins.turn.eot_models import make_scorer  # noqa: E402

PRESETS = {
    "text": ("text", {}),
    "lm": ("lm", {}),
    "lm_raw": ("lm", {"calib": "none"}),
    "lm15": ("lm", {"repo": "onnx-community/Qwen2.5-1.5B-Instruct"}),
    "ten_turn": ("ten_turn", {}),
}


def auc(pos: list[float], neg: list[float]) -> float:
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="experiments/turn/bench/text_eot.jsonl")
    ap.add_argument("--scorer", action="append", required=True, help=f"one of {sorted(PRESETS)}")
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    report = {}
    for name in a.scorer:
        kind, cfg = PRESETS[name]
        s = make_scorer(kind, cfg)
        s("你好")  # warm-up
        scores, lat = [], []
        for r in rows:
            t0 = time.perf_counter()
            scores.append(s(r["text"]))
            lat.append(time.perf_counter() - t0)
        pos = [p for p, r in zip(scores, rows) if r["done"]]
        neg = [p for p, r in zip(scores, rows) if not r["done"]]
        rep = {
            "auc": round(auc(pos, neg), 3),
            "acc@0.5": round(sum((p >= 0.5) == bool(r["done"]) for p, r in zip(scores, rows)) / len(rows), 3),
            "unfinished_called_done@0.75": round(sum(p >= 0.75 for p in neg) / len(neg), 3),
            "finished_called_undone@0.25": round(sum(p <= 0.25 for p in pos) / len(pos), 3),
            "latency_ms_p50": round(1000 * statistics.median(lat), 1),
            "latency_ms_max": round(1000 * max(lat), 1),
        }
        report[name] = rep
        print(name, rep, flush=True)
        report[name + ":rows"] = [{"text": r["text"], "done": r["done"], "p": round(p, 3)} for p, r in zip(scores, rows)]
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
