"""Offline per-module benchmarks. Each takes a JSONL dataset and the module's config section.

  python -m duplexagent bench asr  --data bench_data/asr_smoke.jsonl  [-c cfg] [--set asr.impl=...]
  python -m duplexagent bench fast --data bench_data/fast_delegation.jsonl
  python -m duplexagent bench slow --data bench_data/slow_tasks.jsonl
  python -m duplexagent bench tts  --data bench_data/tts_sentences.jsonl [--wav-dir out/]

The turn detector is benchmarked end to end with scenarios (`python -m duplexagent eval`).
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
import wave
from typing import Any

from duplexagent.core.audio import noise, silence
from duplexagent.core.registry import create, load_modules
from duplexagent.core.types import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, AudioFrame, BrainContext, Delegate, Message, Say, TaskRecord


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _stats(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"mean": None, "p50": None, "p90": None}
    s = sorted(xs)
    return {"mean": round(statistics.mean(s), 4), "p50": round(s[len(s) // 2], 4), "p90": round(s[min(len(s) - 1, int(0.9 * len(s)))], 4)}


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalize(text: str) -> str:
    import re

    return re.sub(r"[\s,.!?;:，。！？；：、…\"'“”‘’]+", "", text.lower())


async def _make(kind: str, cfg: dict):
    load_modules(cfg.get("plugins", {}).get("modules", []))
    p = create(kind, cfg[kind])
    await p.start()
    return p


# ---- ASR ----------------------------------------------------------------------------


def _utterance_frames(item: dict) -> list[AudioFrame]:
    if item.get("wav"):
        with wave.open(item["wav"], "rb") as w:
            pcm = w.readframes(w.getnframes())
        step = FRAME_SAMPLES * 2
        frames = [AudioFrame(pcm[i : i + step].ljust(step, b"\0"), 0.0) for i in range(0, len(pcm), step)]
    else:  # text-only item: simulated speech carrying ground truth (for the mock ASR)
        text = item["text"]
        n = max(1, int(len(text) / 5.0 * 1000 / FRAME_MS))
        frames = [AudioFrame(noise(amplitude=0.1, seed=i), 0.0) for i in range(n)]
        for k, ch in enumerate(text):
            f = frames[min(n - 1, int(k * n / len(text)))]
            f.truth = (f.truth or "") + ch
    tail = [AudioFrame(silence(), 0.0) for _ in range(int(1500 / FRAME_MS))]  # let the endpointer fire
    return frames + tail


async def bench_asr(cfg: dict, data: list[dict], realtime: bool = False) -> dict[str, Any]:
    asr = await _make("asr", cfg)
    errs = chars = 0
    final_lat, first_partial, rows = [], [], []
    try:
        for item in data:
            frames = _utterance_frames(item)
            speech_end = (len(frames) - int(1500 / FRAME_MS)) * FRAME_MS / 1000
            t0 = time.perf_counter()

            async def feed():
                for i, f in enumerate(frames):
                    f.ts = i * FRAME_MS / 1000
                    if realtime:
                        await asyncio.sleep(max(0.0, t0 + f.ts - time.perf_counter()))
                    yield f

            hyp, got_partial = "", None
            async for t in asr.transcribe(feed()):
                if not t.is_final and got_partial is None:
                    got_partial = time.perf_counter() - t0
                if t.is_final:
                    hyp += t.text
                    if realtime:
                        final_lat.append(time.perf_counter() - t0 - speech_end)
            if got_partial is not None:
                first_partial.append(got_partial)
            ref, h = normalize(item["text"]), normalize(hyp)
            e = edit_distance(ref, h)
            errs, chars = errs + e, chars + len(ref)
            rows.append({"ref": item["text"], "hyp": hyp, "cer": round(e / max(1, len(ref)), 4)})
    finally:
        await asr.close()
    out = {"module": "asr", "impl": cfg["asr"]["impl"], "n": len(data), "cer": round(errs / max(1, chars), 4), "first_partial_s": _stats(first_partial), "rows": rows}
    if realtime:
        out["final_after_speech_end_s"] = _stats(final_lat)
    return out


# ---- fast brain ---------------------------------------------------------------------


async def bench_fast(cfg: dict, data: list[dict]) -> dict[str, Any]:
    fast = await _make("fast", cfg)
    ttft, tp, fp, fn, tn, rows = [], 0, 0, 0, 0, []
    try:
        for item in data:
            history = [Message(m["role"], m["content"]) for m in item.get("history", [])] + [Message("user", item["text"])]
            ctx = BrainContext(trigger="user_turn", history=history, user_text=item["text"])
            t0 = time.perf_counter()
            first, said, delegated = None, "", False
            async for a in fast.respond(ctx):
                if isinstance(a, Say):
                    first = first if first is not None else time.perf_counter() - t0
                    said += a.text
                elif isinstance(a, Delegate):
                    delegated = True
            if first is not None:
                ttft.append(first)
            want = bool(item["delegate"])
            tp += want and delegated
            fn += want and not delegated
            fp += delegated and not want
            tn += not want and not delegated
            rows.append({"text": item["text"], "want": want, "got": delegated, "said": said, "ttft": first})
    finally:
        await fast.close()
    n = len(data)
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    return {
        "module": "fast",
        "impl": cfg["fast"]["impl"],
        "n": n,
        "delegation_accuracy": round((tp + tn) / max(1, n), 4),
        "delegation_precision": prec and round(prec, 4),
        "delegation_recall": rec and round(rec, 4),
        "ttft_s": _stats(ttft),
        "rows": rows,
    }


# ---- slow brain ---------------------------------------------------------------------


async def bench_slow(cfg: dict, data: list[dict], concurrency: int = 4) -> dict[str, Any]:
    slow = await _make("slow", cfg)
    sem = asyncio.Semaphore(concurrency)

    async def one(item: dict) -> dict:
        async with sem:
            rec = TaskRecord(query=item["query"], context=item.get("context", ""))
            t0 = time.perf_counter()
            result, progress = "", 0
            try:
                async for u in slow.run(rec):
                    progress += u.kind == "progress"
                    if u.kind in ("result", "error"):
                        result = u.text
            except Exception as e:
                result = f"ERROR {type(e).__name__}: {e}"
            dt = time.perf_counter() - t0
            want = item.get("answer_contains", [])
            ok = all(normalize(w) in normalize(result) for w in want) if want else None
            return {"query": item["query"], "result": result, "ok": ok, "latency": round(dt, 3), "progress_updates": progress}

    try:
        rows = await asyncio.gather(*(one(i) for i in data))
    finally:
        await slow.close()
    graded = [r for r in rows if r["ok"] is not None]
    return {
        "module": "slow",
        "impl": cfg["slow"]["impl"],
        "n": len(rows),
        "accuracy": round(sum(r["ok"] for r in graded) / len(graded), 4) if graded else None,
        "latency_s": _stats([r["latency"] for r in rows]),
        "rows": rows,
    }


# ---- TTS ----------------------------------------------------------------------------


async def bench_tts(cfg: dict, data: list[dict], wav_dir: str | None = None) -> dict[str, Any]:
    tts = await _make("tts", cfg)
    first, rtf, aligned, rows = [], [], 0, []
    try:
        for idx, item in enumerate(data):
            text = item["text"]

            async def pieces():  # stream the text the way a fast brain would (a few chars at a time)
                for i in range(0, len(text), 4):
                    yield text[i : i + 4]

            t0 = time.perf_counter()
            f, audio, covered = None, bytearray(), ""
            async for ch in tts.synthesize(pieces()):
                f = f if f is not None else time.perf_counter() - t0
                audio += ch.pcm
                covered += ch.text
            wall = time.perf_counter() - t0
            dur = len(audio) / 2 / SAMPLE_RATE
            if f is not None:
                first.append(f)
            if dur:
                rtf.append(wall / dur)
            ok = normalize(covered) == normalize(text)
            aligned += ok
            if wav_dir:
                os.makedirs(wav_dir, exist_ok=True)
                with wave.open(os.path.join(wav_dir, f"{idx:03d}.wav"), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(SAMPLE_RATE)
                    w.writeframes(bytes(audio))
            rows.append({"text": text, "first_chunk_s": f, "audio_s": round(dur, 3), "aligned": ok})
    finally:
        await tts.close()
    return {
        "module": "tts",
        "impl": cfg["tts"]["impl"],
        "n": len(data),
        "first_chunk_s": _stats(first),
        "rtf": _stats(rtf),
        "alignment_ok": round(aligned / max(1, len(data)), 4),
        "rows": rows,
    }


BENCHES = {"asr": bench_asr, "fast": bench_fast, "slow": bench_slow, "tts": bench_tts}
