"""标定 hybrid 的概率门限：只在主集上选门限，留出集只用来报告。

    python experiments/fast/calibrate.py -c configs/exp/fast/hybrid-qwen2.5-1.5b.toml --out experiments/fast/results/calib_qwen2.5-1.5b.json

对每条数据算一次首 token 概率（只做 prefill，不生成），离线模拟决策：
路由器有把握的直接采用；否则 P(非“说”标记) >= 门限 才采用模型的动作，否则自己答。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from bench import _history, _task  # noqa: E402

from duplexagent.bench.offline import _make, load_jsonl  # noqa: E402
from duplexagent.config import load_config  # noqa: E402
from duplexagent.core.types import BrainContext, Message  # noqa: E402
from duplexagent.plugins.fast.common import route, running_tasks  # noqa: E402
from duplexagent.plugins.fast.llamacpp import build_messages  # noqa: E402

TAG2ACT = {"说": "say", "委派": "delegate", "取消": "cancel", "补充": "amend"}
GRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]


async def probs_for(fast, data):
    out = []
    for item in data:
        ctx = BrainContext("user_turn", _history(item.get("history", [])) + [Message("user", item["text"])], item["text"],
                           [_task(t) for t in item.get("tasks", [])])
        r = route(ctx)
        tags = ["说", "委派", "取消", "补充"] if running_tasks(ctx) else ["说", "委派"]
        t0 = time.perf_counter()
        p = await fast.backend.tag_probs(build_messages(ctx, fast.system_prompt), fast._prefill("["), tags)
        out.append({"id": item.get("id"), "want": item.get("action", "delegate" if item["delegate"] else "say"),
                    "router": r.kind if r.confident else "chat", "probs": p, "dt": time.perf_counter() - t0})
    return out


def decide(row, gate, mode):
    if mode == "hybrid" and row["router"] != "chat":
        return {"status": "say", "clarify": "say"}.get(row["router"], row["router"])
    p = row["probs"]
    best = max((t for t in p if t != "说"), key=p.get)
    return TAG2ACT[best] if p[best] >= gate else "say"


def score(rows, gate, mode):
    ok = sum(decide(r, gate, mode) == ("say" if r["want"] == "status" else r["want"]) for r in rows)
    tp = sum(decide(r, gate, mode) == "delegate" and r["want"] == "delegate" for r in rows)
    fp = sum(decide(r, gate, mode) == "delegate" and r["want"] != "delegate" for r in rows)
    fn = sum(decide(r, gate, mode) != "delegate" and r["want"] == "delegate" for r in rows)
    return {"gate": gate, "action_acc": round(ok / len(rows), 4), "prec": round(tp / max(1, tp + fp), 4), "rec": round(tp / max(1, tp + fn), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)

    async def go():
        fast = await _make("fast", cfg)
        try:
            return (await probs_for(fast, load_jsonl("bench_data/fast/delegation.jsonl")),
                    await probs_for(fast, load_jsonl("bench_data/fast/delegation_heldout.jsonl")))
        finally:
            await fast.close()

    main_rows, ho_rows = asyncio.run(go())
    res = {"config": a.config}
    for mode in ("llm", "hybrid"):
        curve = [score(main_rows, g, mode) for g in GRID]
        best = max(curve, key=lambda s: (s["action_acc"], s["gate"]))
        res[mode] = {"main_curve": curve, "chosen_gate": best["gate"], "main": best, "heldout": score(ho_rows, best["gate"], mode),
                     "heldout_curve": [score(ho_rows, g, mode) for g in GRID]}
        print(mode, "gate", best["gate"], "main", best, "heldout", res[mode]["heldout"])
    dts = sorted(r["dt"] for r in main_rows + ho_rows)
    res["tag_probs_latency_p50"] = dts[len(dts) // 2]
    print("tag_probs latency p50", round(res["tag_probs_latency_p50"], 3))
    res["rows"] = {"main": main_rows, "heldout": ho_rows}
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
