"""跑「候选 × 数据集」矩阵并汇总排行榜。每个任务一个子进程（内存/CPU 统计互不干扰），串行执行（实时延迟不受争抢影响）。

    python experiments/asr/matrix.py run  [--cands sensevoice ...] [--sets aishell ...] [--rt] [--force]
    python experiments/asr/matrix.py summary          # -> experiments/asr/results/leaderboard.md + summary.json

结果：experiments/asr/results/<候选>/<数据集>.json（实时模式为 <数据集>_rt.json）。
CER 评测统一加 `--set asr.itn=false`（参考文本里的数字是汉字），引擎场景里配置默认 itn=true。
非实时任务对伪流式候选加 `--set asr.partial_ms=0`：RTF 只反映 final 解码成本；partial 反复重解码的开销体现在实时任务的 CPU 占用里。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "experiments", "asr", "results")
CFG = os.path.join(ROOT, "configs", "exp", "asr")
SETS = ["aishell", "wenet", "ascend", "noisy_white10", "noisy_white0", "noisy_babble10", "noisy_babble0"]
RT_SET = "aishell_rt"
NOISY = [s for s in SETS if s.startswith("noisy")]


def candidates() -> list[str]:
    return sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(CFG, "*.toml")))


def asr_cfg(cand: str) -> dict:
    import tomllib

    with open(os.path.join(CFG, f"{cand}.toml"), "rb") as f:
        return tomllib.load(f)["asr"]


def job(cand: str, dataset: str, realtime: bool, extra: list[str], force: bool, limit: int | None, tag: str = "") -> None:
    out = os.path.join(RES, cand + tag, f"{dataset}{'_rt' if realtime else ''}.json")
    if os.path.exists(out) and not force:
        return
    cmd = [sys.executable, "experiments/asr/bench.py", "-c", f"configs/exp/asr/{cand}.toml", "--data", f"bench_data/asr/{dataset}.jsonl",
           "--set", "asr.itn=false", "--out", out] + sum((["--set", s] for s in extra), [])
    if realtime:
        cmd.append("--realtime")
    elif "partial_ms" in asr_cfg(cand):
        cmd += ["--set", "asr.partial_ms=0"]  # RTF = cost of the finals; partial re-decoding cost shows up as CPU in the realtime job
    if limit:
        cmd += ["--limit", str(limit)]
    print(">>", cand + tag, dataset, "realtime" if realtime else "", flush=True)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-3000:], flush=True)
    else:
        print("  ", r.stdout.strip().splitlines()[-1][:400], flush=True)


def load(cand: str, name: str) -> dict | None:
    p = os.path.join(RES, cand, f"{name}.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def fmt(v, pct: bool = False, nd: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.1f}" if pct else f"{v:.{nd}f}"


def summary() -> None:
    rows = []
    for cand in sorted(os.listdir(RES)):
        if not os.path.isdir(os.path.join(RES, cand)):
            continue
        r = {"cand": cand}
        for s in SETS:
            d = load(cand, s)
            r[s] = d and (d["mer"] if s == "ascend" else d["cer"])
            if s == "aishell" and d:
                r["rtf"], r["rss"], r["load_s"] = d["rtf"], d["peak_rss_mb"], d["load_s"]
        nz = [r[s] for s in NOISY if r.get(s) is not None]
        r["noisy"] = sum(nz) / len(nz) if len(nz) == len(NOISY) else None
        d = load(cand, RT_SET + "_rt")
        if d:
            r["fp50"] = d["first_partial_after_onset_s"]["p50"]
            r["fin50"], r["fin90"] = d["final_after_speech_end_s"]["p50"], d["final_after_speech_end_s"]["p90"]
            r["cpu"], r["fpu"], r["rt_cer"] = d["cpu_cores"], d["finals_per_utt"], d["cer"]
            r["rt_rss"] = d["peak_rss_mb"]
        rows.append(r)
    rows.sort(key=lambda r: (r.get("aishell") is None, r.get("aishell") or 9))
    head = ("| 候选 | AISHELL CER% | Wenet CER% | ASCEND MER% | 噪声均值 CER% | 白噪10/0 | babble10/0 | RTF | 首 partial p50 (s) | final 延迟 p50/p90 (s) | CPU 核 | finals/句 | 内存 MB |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    lines = [head]
    for r in rows:
        lines.append(
            f"| {r['cand']} | {fmt(r.get('aishell'), 1)} | {fmt(r.get('wenet'), 1)} | {fmt(r.get('ascend'), 1)} | {fmt(r.get('noisy'), 1)} | "
            f"{fmt(r.get('noisy_white10'), 1)}/{fmt(r.get('noisy_white0'), 1)} | {fmt(r.get('noisy_babble10'), 1)}/{fmt(r.get('noisy_babble0'), 1)} | "
            f"{fmt(r.get('rtf'), nd=3)} | {fmt(r.get('fp50'))} | {fmt(r.get('fin50'))}/{fmt(r.get('fin90'))} | {fmt(r.get('cpu'))} | {fmt(r.get('fpu'))} | "
            f"{fmt(r.get('rt_rss') or r.get('rss'), nd=0)} |"
        )
    md = "\n".join(lines)
    with open(os.path.join(RES, "leaderboard.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    with open(os.path.join(RES, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(md)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--cands", nargs="*")
    ap.add_argument("--sets", nargs="*")
    ap.add_argument("--rt", action="store_true", help="also run the realtime latency job")
    ap.add_argument("--rt-only", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="extra config override for every job")
    ap.add_argument("--tag", default="", help="suffix for the result dir (parameter sweeps)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "summary":
        return summary()
    for cand in a.cands or candidates():
        if not a.rt_only:
            for s in a.sets or SETS:
                job(cand, s, False, a.set, a.force, a.limit, a.tag)
        if a.rt or a.rt_only:
            job(cand, RT_SET, True, a.set, a.force, a.limit, a.tag)
    summary()


if __name__ == "__main__":
    main()
