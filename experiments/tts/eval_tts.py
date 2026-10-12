"""TTS experiment evaluation: everything `bench tts` measures, plus intelligibility, naturalness,
time-alignment quality, streaming stalls and cancellation.

    python experiments/tts/eval_tts.py -c configs/exp/tts/matcha-zh-en.toml \
        --data bench_data/tts/sentences.jsonl --out experiments/tts/results/matcha-zh-en.json \
        [--wav-dir out/tts/matcha-zh-en] [--cps 0] [--set tts.first_min_chars=2] [--no-mos]

Metrics
  first_chunk_s     time from the first text piece to the first audio chunk
  rtf               synthesis wall time / audio duration (text fed instantly, like `bench tts`)
  coverage_ok       concatenated TTSChunk.text == input text (= `bench tts` alignment_ok)
  cer               SenseVoice (sherpa-onnx) re-recognition CER; ref and hyp are both verbalized
                    with our normalizer, so "23" vs "二十三" is not an error
  utmos             UTMOS22-strong (ONNX) MOS prediction, 16 kHz
  align_hit         share of re-recognized characters whose ASR timestamp falls inside the
                    chunk that carries that character (+-80 ms)
  trunc_err_chars   simulated barge-in every 250 ms: |chars our chunks say were played -
                    chars the ASR heard by then|, averaged (what the engine writes to history)
  stall_s / stalls  with --cps > 0 the text arrives at that rate (like an LLM) and playback is
                    simulated in real time; total playback gap after the first chunk
  cancel_*          closing the stream mid-sentence: time for the generator to close and time
                    until the synthesis thread is idle again

Model files: see experiments/tts/download.py. Evaluation models (not in git):
  models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/
  models/eval/utmos22_strong.onnx   (huggingface.co/TigreGotico/utmos-onnx, MIT)
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import statistics
import sys
import time
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from duplexagent.bench.offline import _stats, edit_distance, load_jsonl  # noqa: E402
from duplexagent.config import load_config  # noqa: E402
from duplexagent.core.registry import create, load_modules  # noqa: E402
from duplexagent.core.types import SAMPLE_RATE  # noqa: E402
from duplexagent.plugins.tts.textnorm import Normalizer, spoken_text  # noqa: E402

NORM = Normalizer("keep")


def is_cjk(c: str) -> bool:
    return "一" <= c <= "鿿"


def canon(text: str) -> str:
    """Verbalize + strip punctuation/space + lowercase, for CER."""
    s = spoken_text(NORM.units(text)).lower()
    return "".join(c for c in s if c.isalnum())


# ---- evaluation models ---------------------------------------------------------------------------


class Judges:
    def __init__(self, asr: bool = True, mos: bool = True) -> None:
        self.asr = self.mos = None
        if asr:
            import sherpa_onnx as so

            d = sorted(glob.glob("models/asr/sherpa-onnx-sense-voice-*"))[-1]
            self.asr = so.OfflineRecognizer.from_sense_voice(
                model=f"{d}/model.int8.onnx", tokens=f"{d}/tokens.txt", use_itn=False, num_threads=4, language="auto"
            )
        if mos:
            import onnxruntime as ort

            o = ort.SessionOptions()
            o.intra_op_num_threads = 4
            self.mos = ort.InferenceSession("models/eval/utmos22_strong.onnx", o, providers=["CPUExecutionProvider"])

    def recognize(self, x) -> tuple[str, list[str], list[float]]:
        s = self.asr.create_stream()
        s.accept_waveform(SAMPLE_RATE, x)
        self.asr.decode_stream(s)
        r = s.result
        return r.text, list(r.tokens), list(r.timestamps)

    def utmos(self, x) -> float:
        import numpy as np

        return float(self.mos.run(None, {"wave": np.asarray(x, dtype=np.float32)[None, :]})[0][0])


# ---- alignment metrics ----------------------------------------------------------------------------


def align_pairs(ref: list[str], hyp: list[str]) -> list[tuple[int, int]]:
    """Levenshtein backtrace -> list of (ref_idx, hyp_idx) for equal characters."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, out = n, m, []
    while i > 0 and j > 0:
        if ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j - 1] + 1:
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return out[::-1]


# SenseVoice CTC timestamps precede the true character onset by ~0.10 s (median, measured on
# concatenated clips with a known onset, range 0..0.2 s); a character counts as heard once
# ~0.12 s of it (about half) has played.
ASR_TS_BIAS = 0.10
HEARD_AFTER = 0.12


def alignment_metrics(chunks: list[tuple[float, float, str]], tokens: list[str], ts: list[float], tol: float = 0.08):
    """chunks: (start_s, end_s, text). Compare chunk spans with (bias-corrected) ASR character onsets."""
    ref, owner = [], []
    for k, (_, _, text) in enumerate(chunks):
        for c in spoken_text(NORM.units(text)):
            if is_cjk(c):
                ref.append(c)
                owner.append(k)
    hyp = [(t, s + ASR_TS_BIAS) for t, s in zip(tokens, ts) if len(t) == 1 and is_cjk(t)]
    pairs = align_pairs(ref, [t for t, _ in hyp])
    if not pairs:
        return None, None, 0
    hits = 0
    for ri, hi in pairs:
        a, b, _ = chunks[owner[ri]]
        hits += a - tol <= hyp[hi][1] <= b + tol
    dur = chunks[-1][1]
    errs = []
    t = 0.25
    while t < dur:
        ours = sum(1 for ri, _ in pairs if chunks[owner[ri]][1] <= t)
        truth = sum(1 for _, hi in pairs if hyp[hi][1] + HEARD_AFTER <= t)
        errs.append(abs(ours - truth))
        t += 0.25
    return hits / len(pairs), (statistics.mean(errs) if errs else 0.0), len(pairs)


# ---- runs ------------------------------------------------------------------------------------------


async def run_sentence(tts, text: str, cps: float, step: int = 4) -> dict:
    async def pieces():
        for i in range(0, len(text), step):
            if cps > 0 and i:
                await asyncio.sleep(step / cps)
            yield text[i : i + step]

    t0 = time.perf_counter()
    first = None
    audio = bytearray()
    spans = []
    covered = ""
    play_end = None  # simulated speaker: when the audio handed out so far finishes
    stall, stalls = 0.0, 0
    async for ch in tts.synthesize(pieces()):
        now = time.perf_counter()
        if first is None:
            first = now - t0
            play_end = now
        elif cps > 0 and now > play_end + 0.001:
            gap = now - play_end
            stall += gap
            stalls += gap > 0.05
            play_end = now
        a = len(audio) / 2 / SAMPLE_RATE
        audio += ch.pcm
        covered += ch.text
        spans.append((a, len(audio) / 2 / SAMPLE_RATE, ch.text))
        if cps > 0:
            play_end += ch.duration
            await asyncio.sleep(max(0.0, play_end - time.perf_counter()))
    wall = time.perf_counter() - t0
    return {"first": first, "wall": wall, "audio": bytes(audio), "spans": spans, "covered": covered, "stall": stall, "stalls": stalls}


async def cancel_probe(tts, texts: list[str]) -> dict:
    close_s, idle_s = [], []
    for text in texts:
        async def pieces():
            yield text

        agen = tts.synthesize(pieces())
        await agen.__anext__()
        t = time.perf_counter()
        await agen.aclose()
        close_s.append(time.perf_counter() - t)
        loaded = getattr(tts, "_loaded", None)
        if loaded is not None:
            await asyncio.get_running_loop().run_in_executor(loaded.pool, lambda: None)
        idle_s.append(time.perf_counter() - t)
    return {"cancel_close_s": _stats(close_s), "cancel_idle_s": _stats(idle_s)}


async def evaluate(cfg: dict, data: list[dict], judges: Judges, cps: float, wav_dir: str | None) -> dict:
    import numpy as np

    load_modules(cfg.get("plugins", {}).get("modules", []))
    tts = create("tts", cfg["tts"])
    t = time.perf_counter()
    await tts.start()
    load_s = time.perf_counter() - t
    rows = []
    try:
        for idx, item in enumerate(data):
            text = item["text"]
            r = await run_sentence(tts, text, 0.0)
            x = np.frombuffer(r["audio"], dtype="<i2").astype(np.float32) / 32768
            dur = len(x) / SAMPLE_RATE
            row = {
                "text": text, "cat": item.get("cat", ""), "first_chunk_s": round(r["first"] or 0, 4),
                "audio_s": round(dur, 3), "rtf": round(r["wall"] / dur, 4) if dur else None,
                "coverage_ok": "".join(c for c in r["covered"] if c.isalnum()) == "".join(c for c in text if c.isalnum()),
                "chunks": len(r["spans"]),
            }
            if cps > 0:
                rs = await run_sentence(tts, text, cps)
                row.update(stream_first_s=round(rs["first"] or 0, 4), stall_s=round(rs["stall"], 3), stalls=rs["stalls"])
            if judges.asr and dur:
                hyp, toks, ts = judges.recognize(x)
                ref_c, hyp_c = canon(text), canon(hyp)
                row.update(hyp=hyp, cer=round(edit_distance(ref_c, hyp_c) / max(1, len(ref_c)), 4), ref_len=len(ref_c), errs=edit_distance(ref_c, hyp_c))
                hit, terr, npairs = alignment_metrics(r["spans"], toks, ts)
                row.update(align_hit=hit and round(hit, 4), trunc_err_chars=terr and round(terr, 3), align_pairs=npairs)
            if judges.mos and dur:
                row["utmos"] = round(judges.utmos(x), 3)
            if wav_dir:
                os.makedirs(wav_dir, exist_ok=True)
                with wave.open(os.path.join(wav_dir, f"{idx:03d}.wav"), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(SAMPLE_RATE)
                    w.writeframes(r["audio"])
            rows.append(row)
            print(f"[{idx + 1}/{len(data)}] first={row['first_chunk_s']:.3f} rtf={row['rtf']} cer={row.get('cer')} "
                  f"utmos={row.get('utmos')} hit={row.get('align_hit')} | {text[:30]}", file=sys.stderr)
        long_texts = [i["text"] for i in data if i.get("cat") == "long"][:5] or [data[-1]["text"]]
        cancel = await cancel_probe(tts, long_texts)
    finally:
        await tts.close()

    def agg(key: str, rows=rows) -> list:
        return [r[key] for r in rows if r.get(key) is not None]

    graded = [r for r in rows if "errs" in r]
    cer = sum(r["errs"] for r in graded) / max(1, sum(r["ref_len"] for r in graded)) if graded else None
    by_cat = {}
    for cat in sorted({r["cat"] for r in rows}):
        rs = [r for r in graded if r["cat"] == cat]
        if rs:
            by_cat[cat] = round(sum(r["errs"] for r in rs) / max(1, sum(r["ref_len"] for r in rs)), 4)
    hit_num = sum(r["align_hit"] * r["align_pairs"] for r in rows if r.get("align_hit") is not None)
    hit_den = sum(r["align_pairs"] for r in rows if r.get("align_hit") is not None)
    summary = {
        "impl": cfg["tts"]["impl"], "preset": cfg["tts"].get("preset"), "n": len(rows), "load_s": round(load_s, 2),
        "first_chunk_s": _stats(agg("first_chunk_s")), "rtf": _stats(agg("rtf")),
        "coverage_ok": round(sum(r["coverage_ok"] for r in rows) / max(1, len(rows)), 4),
        "cer": cer and round(cer, 4), "cer_by_cat": by_cat,
        "utmos": round(statistics.mean(agg("utmos")), 3) if agg("utmos") else None,
        "align_hit": round(hit_num / hit_den, 4) if hit_den else None,
        "trunc_err_chars": round(statistics.mean(agg("trunc_err_chars")), 3) if agg("trunc_err_chars") else None,
        "chunks_per_s": round(sum(agg("chunks")) / max(1e-9, sum(agg("audio_s"))), 2),
        **cancel,
    }
    if cps > 0:
        summary.update(cps=cps, stream_first_s=_stats(agg("stream_first_s")), stall_s_total=round(sum(agg("stall_s")), 3),
                       stalls=sum(agg("stalls")), stall_rows=sum(1 for r in rows if r.get("stalls")))
    return {"summary": summary, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/default.toml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--data", default="bench_data/tts/sentences.jsonl")
    ap.add_argument("--out")
    ap.add_argument("--wav-dir")
    ap.add_argument("--cps", type=float, default=0.0, help="also stream text at this many chars/s and measure stalls")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--no-mos", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    data = load_jsonl(a.data)
    if a.limit:
        data = data[: a.limit]
    judges = Judges(asr=not a.no_asr, mos=not a.no_mos)
    res = asyncio.run(evaluate(cfg, data, judges, a.cps, a.wav_dir))
    res["summary"].update(config=a.config, overrides=a.set)
    print(json.dumps(res["summary"], ensure_ascii=False, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
