"""单个候选 × 单个数据集的 ASR 评测（官方 `bench asr` 的超集）。

    python experiments/asr/bench.py -c configs/exp/asr/sherpa_zipformer_zh.toml --data bench_data/asr/aishell.jsonl
    python experiments/asr/bench.py -c ... --data bench_data/asr/aishell_rt.jsonl --realtime --out experiments/asr/results/x.json

与 duplexagent/bench/offline.py::bench_asr 用同样的切帧、文本归一化和 CER 定义，另外给出：
  mer                         中英混合错误率：汉字按字、英文按词计（中英混说数据更合理）
  rtf                         非实时模式：总处理墙钟 / 总送入音频时长（含 1.5 s 尾部静音）
  first_partial_after_onset_s 首个 partial 相对「开始说话」的延迟（能量检测语音起点）
  final_after_speech_end_s    实时模式：最后一个 final 相对「真正说完」（能量检测语音终点）的延迟
  final_after_wav_end_s       实时模式：同上但相对 wav 结尾（= 官方 bench 定义，只取最后一个 final）
  finals_per_utt              一句话被切成几个 final（>1 说明句中停顿被当成断句）
  cpu_cores                   评测期间进程平均占用的核数（CPU 时间 / 墙钟）
  peak_rss_mb                 进程峰值内存
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import resource
import sys
import time
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from duplexagent.bench.offline import _stats, _utterance_frames, edit_distance, load_jsonl, normalize  # noqa: E402
from duplexagent.config import load_config  # noqa: E402
from duplexagent.core.registry import create, load_modules  # noqa: E402
from duplexagent.core.types import FRAME_MS  # noqa: E402

TAIL_S = 1.5  # _utterance_frames appends this much silence


def mixed_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+|[^\sa-z0-9']", normalize(text))


def voiced_span(path: str) -> tuple[float, float]:
    """(onset, end) of speech in seconds, same rule as prepare_data.trim."""
    import numpy as np

    with wave.open(path, "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype("float32") / 32768
    f = 320
    n = len(x) // f
    e = 10 * np.log10((x[: n * f].reshape(n, f) ** 2).mean(axis=1) + 1e-10)
    v = np.where(e > max(e.max() - 40, np.percentile(e, 10) + 10))[0]
    return (v[0] * f / 16000, (v[-1] + 1) * f / 16000) if len(v) else (0.0, len(x) / 16000)


async def run(cfg: dict, data: list[dict], realtime: bool) -> dict:
    load_modules(cfg.get("plugins", {}).get("modules", []))
    t_load = time.perf_counter()
    asr = create("asr", cfg["asr"])
    await asr.start()
    load_s = time.perf_counter() - t_load
    cpu0, wall0 = sum(os.times()[:2]), time.perf_counter()
    errs = chars = merr = mtok = 0
    audio_s = 0.0
    fp_onset, fin_end, fin_wav, finals_n, rows = [], [], [], [], []
    try:
        for item in data:
            frames = _utterance_frames(item)
            wav_end = (len(frames) - int(TAIL_S * 1000 / FRAME_MS)) * FRAME_MS / 1000
            onset, vend = voiced_span(os.path.join(ROOT, item["wav"])) if item.get("wav") else (0.0, wav_end)
            audio_s += len(frames) * FRAME_MS / 1000
            t0 = time.perf_counter()

            async def feed():
                for i, f in enumerate(frames):
                    f.ts = i * FRAME_MS / 1000
                    if realtime:
                        await asyncio.sleep(max(0.0, t0 + f.ts - time.perf_counter()))
                    yield f

            hyp, first_partial, last_final, nfin, partials = "", None, None, 0, 0
            async for t in asr.transcribe(feed()):
                now = time.perf_counter() - t0
                if not t.is_final:
                    partials += 1
                    if first_partial is None:
                        first_partial = now
                else:
                    hyp += t.text
                    nfin += 1
                    last_final = now
                    if first_partial is None:  # a final with no partial before it counts as the first result
                        first_partial = now
            ref, h = normalize(item["text"]), normalize(hyp)
            e = edit_distance(ref, h)
            errs, chars = errs + e, chars + len(ref)
            rt, ht = mixed_tokens(item["text"]), mixed_tokens(hyp)
            me = edit_distance(rt, ht)
            merr, mtok = merr + me, mtok + len(rt)
            finals_n.append(nfin)
            row = {"id": item.get("id"), "ref": item["text"], "hyp": hyp, "cer": round(e / max(1, len(ref)), 4), "finals": nfin, "partials": partials}
            if realtime:
                if first_partial is not None:
                    fp_onset.append(first_partial - onset)
                    row["first_partial_after_onset_s"] = round(first_partial - onset, 3)
                if last_final is not None:
                    fin_end.append(last_final - vend)
                    fin_wav.append(last_final - wav_end)
                    row["final_after_speech_end_s"] = round(last_final - vend, 3)
            rows.append(row)
    finally:
        await asr.close()
    wall = time.perf_counter() - wall0
    cpu = sum(os.times()[:2]) - cpu0
    out = {
        "impl": cfg["asr"]["impl"],
        "asr": cfg["asr"],
        "n": len(data),
        "realtime": realtime,
        "cer": round(errs / max(1, chars), 4),
        "mer": round(merr / max(1, mtok), 4),
        "finals_per_utt": round(sum(finals_n) / max(1, len(finals_n)), 3),
        "empty_hyp": sum(1 for r in rows if not r["hyp"]),
        "load_s": round(load_s, 2),
        "cpu_cores": round(cpu / wall, 3),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }
    if realtime:
        out["first_partial_after_onset_s"] = _stats(fp_onset)
        out["final_after_speech_end_s"] = _stats(fin_end)
        out["final_after_wav_end_s"] = _stats(fin_wav)
    else:
        out["rtf"] = round(wall / audio_s, 4)
    out["rows"] = rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default="configs/default.toml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    os.chdir(ROOT)
    cfg = load_config(a.config, a.set)
    data = load_jsonl(a.data)[: a.limit]
    res = asyncio.run(run(cfg, data, a.realtime))
    res = {"config": a.config, "overrides": a.set, "data": a.data, **res}
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
