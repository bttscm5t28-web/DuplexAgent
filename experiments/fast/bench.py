"""快脑扩展基准：委派 / 取消 / 补充 / 进度 + 不瞎猜 + 融合质量。

    python experiments/fast/bench.py delegation --data bench_data/fast/delegation.jsonl --set fast.impl=router
    python experiments/fast/bench.py fusion     --data bench_data/fast/fusion.jsonl     --set fast.impl=router
    python experiments/fast/bench.py all --set fast.impl=hybrid --out experiments/fast/results/hybrid.json

与 `python -m duplexagent bench fast` 兼容（同样的 text / delegate / history 字段），额外支持
tasks / action / forbid（见 LABELING.md）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from duplexagent.bench.offline import _make, _stats, load_jsonl  # noqa: E402
from duplexagent.config import load_config  # noqa: E402
from duplexagent.core.types import (  # noqa: E402
    AmendTask,
    BrainContext,
    CancelTask,
    Delegate,
    Message,
    Say,
    TaskRecord,
    TaskStatus,
)

CORRECTION_WORDS = ("更正", "不对", "说错", "不准", "纠正", "其实", "应该是", "搞错", "有误")
MARKDOWN_RE = re.compile(r"\*\*|^\s*[-*#]\s|\n\s*\d+[.、]\s|[😀-🙏]", re.M)


def _task(d: dict) -> TaskRecord:
    t = TaskRecord(query=d["query"], id=d["id"])
    t.status = TaskStatus(d.get("status", "running"))
    t.progress = list(d.get("progress", []))
    t.result = d.get("result")
    return t


def _history(rows: list[dict]) -> list[Message]:
    return [Message(m["role"], m["content"], {"interrupted": True} if m.get("interrupted") else {}) for m in rows]


WARM = False  # --warm: let the plugin cache the previous turns first, as a live session would


async def _run(fast, ctx: BrainContext, timeout: float) -> dict:
    if WARM and hasattr(fast, "prime"):
        await fast.prime(ctx)
    t0 = time.perf_counter()
    first = decision = None
    said, actions = "", []

    async def go():
        nonlocal first, decision, said
        async for a in fast.respond(ctx):
            now = time.perf_counter() - t0
            if isinstance(a, Say):
                if first is None and a.text.strip():
                    first = now
                said += a.text
            else:
                decision = decision if decision is not None else now
                actions.append(a)

    err = None
    try:
        await asyncio.wait_for(go(), timeout)
    except Exception as e:  # noqa: BLE001 - a broken candidate should score, not crash the bench
        err = f"{type(e).__name__}: {e}"
    return {"ttft": first, "decision_s": decision, "total_s": time.perf_counter() - t0, "said": said, "actions": actions, "error": err}


def predicted_action(actions: list) -> str:
    kinds = {type(a) for a in actions}
    if CancelTask in kinds:
        return "cancel"
    if AmendTask in kinds:
        return "amend"
    if Delegate in kinds:
        return "delegate"
    return "say"


async def bench_delegation(fast, data: list[dict], timeout: float = 60.0) -> dict:
    tp = fp = fn = tn = 0
    ttft, rows, forbid_hits, silent_delegate, errors = [], [], 0, 0, 0
    act_ok, per_cat, confusion = 0, defaultdict(lambda: [0, 0]), Counter()
    for item in data:
        history = _history(item.get("history", [])) + [Message("user", item["text"])]
        tasks = [_task(t) for t in item.get("tasks", [])]
        ctx = BrainContext(trigger="user_turn", history=history, user_text=item["text"], tasks=tasks)
        r = await _run(fast, ctx, timeout)
        errors += r["error"] is not None
        delegated = any(isinstance(a, Delegate) for a in r["actions"])
        want = bool(item["delegate"])
        tp += want and delegated
        fn += want and not delegated
        fp += delegated and not want
        tn += not want and not delegated
        want_act = item.get("action", "delegate" if want else "say")
        got_act = predicted_action(r["actions"])
        ok = got_act == ("say" if want_act == "status" else want_act)
        if ok and want_act in ("cancel", "amend"):  # must target an existing running task
            ids = {t.id for t in tasks if t.status == TaskStatus.RUNNING}
            ok = all(a.task_id in ids for a in r["actions"] if isinstance(a, (CancelTask, AmendTask)))
        if want_act in ("say", "status") and got_act in ("cancel", "amend"):
            ok = False
        act_ok += ok
        per_cat[item.get("cat", "?")][0] += ok
        per_cat[item.get("cat", "?")][1] += 1
        confusion[f"{want_act}->{got_act}"] += 1
        hit = [w for w in item.get("forbid", []) if w in r["said"]]
        forbid_hits += bool(hit)
        if delegated and not r["said"].strip():
            silent_delegate += 1
        if r["ttft"] is not None:
            ttft.append(r["ttft"])
        rows.append({
            "id": item.get("id"), "cat": item.get("cat"), "text": item["text"], "want": want_act, "got": got_act, "ok": ok,
            "said": r["said"], "forbid_hit": hit, "ttft": r["ttft"] and round(r["ttft"], 4),
            "decision_s": r["decision_s"] and round(r["decision_s"], 4), "error": r["error"],
            "actions": [f"{type(a).__name__}({getattr(a, 'query', getattr(a, 'task_id', ''))})" for a in r["actions"]],
        })
    n = len(data)
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
    n_forbid = sum(1 for i in data if i.get("forbid"))
    return {
        "n": n,
        "delegation_accuracy": round((tp + tn) / max(1, n), 4),
        "delegation_precision": prec and round(prec, 4),
        "delegation_recall": rec and round(rec, 4),
        "delegation_f1": f1 and round(f1, 4),
        "action_accuracy": round(act_ok / max(1, n), 4),
        "forbid_violation_rate": round(forbid_hits / max(1, n_forbid), 4) if n_forbid else None,
        "silent_delegate": silent_delegate,
        "errors": errors,
        "ttft_s": _stats(ttft),
        "per_cat_accuracy": {k: round(a / b, 3) for k, (a, b) in sorted(per_cat.items())},
        "confusion": dict(sorted(confusion.items())),
        "rows": rows,
    }


def grade_fusion(item: dict, said: str) -> dict:
    s = said.replace(" ", "")
    mention = all(any(w in s for w in grp) for grp in item["must_mention_any"])
    contain = all(any(w in s for w in grp) for grp in item["must_contain_any"])
    short = len(s) <= item.get("max_chars", 80)
    spoken = not MARKDOWN_RE.search(said)
    corr = any(w in s for w in CORRECTION_WORDS)
    correction_ok = corr == bool(item.get("correction"))
    return {"mention": mention, "contain": contain, "short": short, "spoken": spoken, "correction_ok": correction_ok,
            "all": mention and contain and short and spoken and correction_ok}


async def bench_fusion(fast, data: list[dict], timeout: float = 60.0) -> dict:
    rows, ttft, agg = [], [], Counter()
    for item in data:
        history = _history(item.get("history", []))
        deliver = []
        for d in item["deliver"]:
            t = _task(d)
            t.delivered = True
            deliver.append(t)
            history.append(Message("task", t.result or "", {"task_id": t.id, "query": t.query, "status": t.status.value}))
        ctx = BrainContext(trigger="task_result", history=history, tasks=deliver, deliver=deliver)
        r = await _run(fast, ctx, timeout)
        g = grade_fusion(item, r["said"])
        agg.update(k for k, v in g.items() if v)
        if r["ttft"] is not None:
            ttft.append(r["ttft"])
        rows.append({"id": item["id"], "said": r["said"], "chars": len(r["said"]), "ttft": r["ttft"], **g,
                     "stray_actions": [type(a).__name__ for a in r["actions"]], "error": r["error"]})
    n = len(data)
    return {"n": n, **{f"fusion_{k}": round(agg[k] / max(1, n), 4) for k in ("all", "mention", "contain", "short", "spoken", "correction_ok")},
            "ttft_s": _stats(ttft), "rows": rows}


DEFAULT_DATA = {
    "delegation": "bench_data/fast/delegation.jsonl",
    "heldout": "bench_data/fast/delegation_heldout.jsonl",
    "fusion": "bench_data/fast/fusion.jsonl",
    "fusion_heldout": "bench_data/fast/fusion_heldout.jsonl",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["delegation", "heldout", "fusion", "fusion_heldout", "all"])
    ap.add_argument("--data")
    ap.add_argument("-c", "--config", default="configs/default.toml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--rows", action="store_true", help="print failing rows")
    ap.add_argument("--warm", action="store_true", help="prime the plugin's prompt cache with the previous turns")
    a = ap.parse_args()
    global WARM
    WARM = a.warm
    cfg = load_config(a.config, a.set)
    which = ["delegation", "heldout", "fusion", "fusion_heldout"] if a.which == "all" else [a.which]

    async def go():
        fast = await _make("fast", cfg)
        out = {"impl": cfg["fast"]["impl"], "config": a.config, "overrides": a.set, "warm": a.warm}
        try:
            for w in which:
                data = load_jsonl(a.data if a.data and len(which) == 1 else DEFAULT_DATA[w])[: a.limit]
                fn = bench_fusion if w.startswith("fusion") else bench_delegation
                out[w] = await fn(fast, data)
        finally:
            await fast.close()
        return out

    res = asyncio.run(go())
    for w in which:
        r = res[w]
        print(f"== {w}")
        print(json.dumps({k: v for k, v in r.items() if k != "rows"}, ensure_ascii=False))
        if a.rows:
            for row in r["rows"]:
                if not row.get("ok", row.get("all")) or row.get("forbid_hit"):
                    print("  ", json.dumps(row, ensure_ascii=False))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
