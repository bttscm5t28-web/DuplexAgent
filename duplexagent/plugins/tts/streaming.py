"""Shared machinery for real streaming TTS plugins.

A concrete backend only implements `_load()` and `_synth(text, stop) -> (float32 samples, sr)`.
This module handles everything else:

  * Segmentation of the incrementally arriving text (`Segmenter`): a short first segment for a
    fast first chunk, longer clause/sentence segments afterwards for better prosody, and a
    deadline-driven "urgent" cut when playback is about to run dry while the fast brain stalls.
  * Text normalization with alignment (`textnorm.Normalizer`).
  * Splitting each synthesized segment back into per-character chunks (`align_units`), so
    `TTSChunk.text` is accurate to roughly one character: energy-based pause detection anchors
    the punctuation, and the speech between anchors is split by expected syllable weights.
  * Resampling to the engine's 16 kHz int16 PCM.
  * Cancellation: closing the generator stops the producer at once; the backend's `stop`
    event lets it abort between sentences; nothing more is synthesized ahead than needed.

numpy is required by all real backends; it is imported lazily.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from duplexagent.core.interfaces import TTS
from duplexagent.core.types import SAMPLE_RATE, TTSChunk
from duplexagent.plugins.tts.textnorm import (
    CLAUSE_END,
    PAUSE_WEIGHT,
    SENTENCE_END,
    SILENT,
    Normalizer,
    Unit,
    spoken_text,
)

# Characters after which a cut inside a long clause is acceptable (function words / particles).
SOFT_BREAK_AFTER = set("的了着过吧呢吗啊呀嘛是在和与或就也都还又把被给让对向从到而且但")


def _is_punct(c: str) -> bool:
    return c in PAUSE_WEIGHT or c in SILENT or c.isspace()


def _speech_len(s: str) -> int:
    return sum(1 for c in s if not _is_punct(c))


class Segmenter:
    """Incremental text segmentation for streaming synthesis.

    feed() pieces as they arrive; pop() returns the next segment ready to synthesize, or None.
      first segment: cut at the first punctuation once it has >= first_min chars, or at a soft
                     break once it exceeds first_max chars;
      later ones:    cut at the last sentence end within max_chars, else the last clause end with
                     >= min_chars, else a soft break when longer than max_chars;
      urgent=True:   playback is about to starve: take the longest safe prefix available;
      final=True:    the text stream ended: same cuts, and the remainder when no cut applies.
    """

    def __init__(self, first_min: int = 4, first_max: int = 14, min_chars: int = 10, max_chars: int = 60) -> None:
        self.first_min, self.first_max = first_min, first_max
        self.min_chars, self.max_chars = min_chars, max_chars
        self.buf = ""
        self.emitted = 0

    def feed(self, piece: str) -> None:
        self.buf += piece

    def _safe_cut(self, i: int, final: bool) -> bool:
        """Can we cut between buf[i-1] and buf[i]? Never inside a number or a Latin word."""
        if i >= len(self.buf):
            if final:
                return True
            # buffer end: "3." may become "3.5", "Pyth" may become "Python"
            last = self.buf[i - 1]
            return not (last.isascii() and last.isalnum()) and not (last in ".,:" and i >= 2 and self.buf[i - 2].isdigit())
        a, b = self.buf[i - 1], self.buf[i]
        if a in ".,:" and i >= 2 and self.buf[i - 2].isdigit() and b.isdigit():
            return False
        return not (a.isascii() and a.isalnum() and b.isascii() and b.isalnum())

    def _cut(self, i: int) -> str:
        # keep trailing punctuation/spaces with the segment
        while i < len(self.buf) and _is_punct(self.buf[i]) and self._safe_cut(i + 1, False):
            i += 1
        seg, self.buf = self.buf[:i], self.buf[i:]
        self.emitted += 1
        return seg

    def pop(self, final: bool = False, urgent: bool = False) -> str | None:
        buf = self.buf
        if not buf:
            return None
        if final:
            seg = self._pop(urgent=False)
            if seg is None:
                seg, self.buf = buf, ""
                self.emitted += 1
            return seg
        return self._pop(urgent)

    def _pop(self, urgent: bool) -> str | None:
        buf = self.buf
        first = self.emitted == 0
        lo, hi = (self.first_min, self.first_max) if first else (self.min_chars, self.max_chars)
        strong, weak, soft = [], [], []
        n = 0
        for i, c in enumerate(buf):
            if not _is_punct(c):
                n += 1
            j = i + 1
            if c in SENTENCE_END and self._safe_cut(j, False):
                strong.append((j, n))
            elif c in CLAUSE_END and self._safe_cut(j, False):
                weak.append((j, n))
            elif (c in SOFT_BREAK_AFTER or c == " ") and self._safe_cut(j, False):
                soft.append((j, n))
        total = n
        if first:
            for j, k in sorted(strong + weak):
                if lo <= k <= hi:
                    return self._cut(j)
        else:
            within = [(j, k) for j, k in strong if k <= hi and k >= 2]
            if within:
                return self._cut(within[-1][0])
            within = [(j, k) for j, k in weak if lo <= k <= hi]
            # wait for a possible sentence end unless the clause is already long
            if within and (total >= hi or within[-1][1] >= lo * 2):
                return self._cut(within[-1][0])
            if within and urgent:
                return self._cut(within[-1][0])
        if total > hi:
            cands = [(j, k) for j, k in strong + weak + soft if k <= hi and k >= max(1, lo // 2)]
            if cands:
                return self._cut(max(cands)[0])
            # no break at all: cut at the limit (but never inside a number / word)
            n = 0
            for i, c in enumerate(buf):
                n += not _is_punct(c)
                if n >= hi and self._safe_cut(i + 1, False):
                    return self._cut(i + 1)
        if urgent:
            cands = [(j, k) for j, k in strong + weak + soft if k >= 1]
            if cands:
                return self._cut(max(cands)[0])
            if self._safe_cut(len(buf), False) and total >= 1:
                return self._cut(len(buf))
        return None


# ---- audio helpers ------------------------------------------------------------------------------


def resample(x, sr_from: int, sr_to: int = SAMPLE_RATE):
    import numpy as np

    if sr_from == sr_to or len(x) == 0:
        return x.astype(np.float32)
    try:
        import soxr

        return soxr.resample(x.astype(np.float32), sr_from, sr_to, quality="HQ").astype(np.float32)
    except ImportError:
        pass
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr_from, sr_to)
        return resample_poly(x, sr_to // g, sr_from // g).astype(np.float32)
    except ImportError:
        n = int(round(len(x) * sr_to / sr_from))
        return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def to_pcm16(x) -> bytes:
    import numpy as np

    return (np.clip(x, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def voiced_mask(x, sr: int, frame_ms: int = 10, rel_db: float = 35.0, floor: float = 0.004):
    """Per-frame voice-activity mask from energy relative to the segment's loud frames."""
    import numpy as np

    hop = max(1, sr * frame_ms // 1000)
    n = len(x) // hop
    if n == 0:
        return np.zeros(0, dtype=bool), hop
    fr = x[: n * hop].reshape(n, hop)
    rms = np.sqrt((fr.astype(np.float64) ** 2).mean(axis=1) + 1e-12)
    ref = np.percentile(rms, 95)
    thr = max(ref * 10 ** (-rel_db / 20), floor)
    return rms > thr, hop


def _silent_runs(mask, min_frames: int) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, v in enumerate(list(mask) + [True]):
        if not v and start is None:
            start = i
        elif v and start is not None:
            if i - start >= min_frames:
                runs.append((start, i))
            start = None
    return runs


def align_units(units: list[Unit], x, sr: int, min_pause_ms: int = 80, anchor: bool = True) -> list[tuple[int, int]]:
    """Split a segment's audio among its units. Returns [(start, end)] sample spans covering x.

    Speech region = first..last voiced frame. Punctuation units are matched (in order) to the
    detected internal pauses closest to where the weights predict them ("anchors"); the speech
    between anchors is split proportionally to the units' syllable weights.
    """
    import numpy as np

    n_units, total = len(units), len(x)
    if n_units == 0:
        return []
    mask, hop = voiced_mask(x, sr)
    voiced = np.flatnonzero(mask)
    if len(voiced) == 0:
        s0, s1 = 0, total
    else:
        s0, s1 = int(voiced[0]) * hop, min(total, (int(voiced[-1]) + 1) * hop)
    weights = [u.weight for u in units]
    # Trailing punctuation sits after the speech region; it should get the trailing silence.
    last_speech = max((i for i, u in enumerate(units) if not u.pause and u.weight > 0), default=n_units - 1)
    inner_w = [w if i <= last_speech else 0.0 for i, w in enumerate(weights)]
    W = sum(inner_w) or 1.0
    # predicted boundaries (sample positions) inside [s0, s1]
    cum = np.concatenate([[0.0], np.cumsum(inner_w)]) / W
    pred = s0 + cum * (s1 - s0)

    # anchors: (unit index i) -> (start, end) for pause units matched to detected pauses
    anchors: dict[int, tuple[int, int]] = {}
    if anchor and len(voiced):
        runs = [(a * hop, b * hop) for a, b in _silent_runs(mask, max(1, min_pause_ms // 10)) if a * hop > s0 and b * hop < s1]
        tol = min(max(0.2 * (s1 - s0), 0.3 * sr), 0.8 * sr)
        j0 = 0
        for i, u in enumerate(units[: last_speech + 1]):
            if not u.pause or u.weight <= 0:
                continue
            centre = (pred[i] + pred[i + 1]) / 2
            best, bestc = None, None
            for j in range(j0, len(runs)):
                a, b = runs[j]
                d = abs((a + b) / 2 - centre)
                if d > tol:
                    continue
                cost = d - 2.0 * (b - a)  # prefer long pauses: short gaps are often stop closures
                if bestc is None or cost < bestc:
                    best, bestc = j, cost
            if best is not None:
                anchors[i] = runs[best]
                j0 = best + 1
        # Drop anchors that imply an implausible speaking rate in the pieces they delimit
        # (a model that does not pause at a comma would otherwise drag text onto a wrong gap).
        while anchors:
            keys = sorted(anchors)
            edges = [(0, float(s0))] + [(k, float(anchors[k][0])) for k in keys] + [(last_speech + 1, float(s1))]
            ends = [(0, float(s0))] + [(k + 1, float(anchors[k][1])) for k in keys] + [(last_speech + 1, float(s1))]
            rate = (s1 - s0) / W
            worst, worst_dev = None, 0.0
            for q in range(len(keys) + 1):
                i0, t0 = ends[q]
                i1, t1 = edges[q + 1]
                w = sum(inner_w[i0:i1])
                if w < 1.0:
                    continue
                r = (t1 - t0) / w / rate
                dev = max(r, 1 / max(r, 1e-6))
                if dev > 1.6 and dev > worst_dev:
                    worst_dev = dev
                    worst = keys[q] if q < len(keys) else keys[q - 1]
                    if q > 0 and q < len(keys):  # piece between two anchors: drop the later one
                        worst = keys[q]
            if worst is None:
                break
            del anchors[worst]

    # piecewise proportional split between anchors
    bounds = [0.0] * (n_units + 1)
    fixed = {0: float(s0)}
    for i, (a, b) in anchors.items():
        fixed[i], fixed[i + 1] = float(a), float(b)
    fixed[last_speech + 1] = float(s1)
    keys = sorted(fixed)
    for k in range(len(keys) - 1):
        i0, i1 = keys[k], keys[k + 1]
        a, b = fixed[i0], fixed[i1]
        w = inner_w[i0:i1]
        sw = sum(w)
        acc = 0.0
        for t in range(i0, i1 + 1):
            bounds[t] = a + (b - a) * (acc / sw if sw else (t - i0) / max(1, i1 - i0))
            if t < i1:
                acc += w[t - i0]
    for t in range(last_speech + 1, n_units + 1):
        bounds[t] = float(s1)
    bounds[0] = 0.0
    bounds[n_units] = float(total)
    out = []
    for i in range(n_units):
        a, b = int(round(bounds[i])), int(round(bounds[i + 1]))
        out.append((a, max(a, b)))
    return out


def trim_silence(x, sr: int, lead_ms: int = 20, tail_ms: int = 60, with_offset: bool = False):
    mask, hop = voiced_mask(x, sr)
    import numpy as np

    v = np.flatnonzero(mask)
    if len(v) == 0:
        return (x[:0], 0) if with_offset else x[:0]
    a = max(0, int(v[0]) * hop - sr * lead_ms // 1000)
    b = min(len(x), (int(v[-1]) + 1) * hop + sr * tail_ms // 1000)
    return (x[a:b], a) if with_offset else x[a:b]


def marks_to_spans(units: list[Unit], marks: list[tuple[str, float, float]], n: int, sr: int) -> list[tuple[int, int]] | None:
    """Unit spans from backend timing marks [(spoken text, start_s, end_s)], e.g. word boundaries.

    Mark characters are matched in order to the units' spoken characters; characters without a
    mark are interpolated. Returns None when too little matches (caller falls back to energy).
    """
    chars = [(i, c) for i, u in enumerate(units) if not u.pause for c in u.spoken.lower() if c.isalnum()]
    if not chars or not marks:
        return None
    times: list[tuple[float, float] | None] = [None] * len(chars)
    k = 0
    for text, a, b in marks:
        mc = [c for c in text.lower() if c.isalnum()]
        for q, c in enumerate(mc):
            for look in range(k, min(len(chars), k + 4)):
                if chars[look][1] == c:
                    s0 = a + (b - a) * q / len(mc)
                    times[look] = (s0, a + (b - a) * (q + 1) / len(mc))
                    k = look + 1
                    break
    known = [j for j, t in enumerate(times) if t is not None]
    if len(known) < max(1, len(chars) // 2):
        return None
    for j in range(len(chars)):  # interpolate gaps
        if times[j] is None:
            lo = max((x for x in known if x < j), default=None)
            hi = min((x for x in known if x > j), default=None)
            if lo is None:
                times[j] = (times[hi][0], times[hi][0])
            elif hi is None:
                times[j] = (times[lo][1], times[lo][1])
            else:
                f = (j - lo) / (hi - lo)
                t = times[lo][1] + (times[hi][0] - times[lo][1]) * f
                times[j] = (t, t)
    start: dict[int, float] = {}
    end: dict[int, float] = {}
    for (i, _), (a, b) in zip(chars, times):
        start[i] = min(start.get(i, a), a)
        end[i] = max(end.get(i, b), b)
    bounds = [0.0] * (len(units) + 1)
    prev_end = 0.0
    for i in range(len(units)):
        if i in start:
            bounds[i] = start[i] if i else 0.0
            prev_end = end[i]
        else:
            bounds[i] = prev_end if i else 0.0
    bounds[len(units)] = n / sr
    out, last = [], 0
    for i in range(len(units)):
        a = max(last, min(n, int(round(bounds[i] * sr))))
        b = max(a, min(n, int(round(bounds[i + 1] * sr)))) if i + 1 < len(units) else n
        out.append((a, b))
        last = b
    return out


# ---- model cache -----------------------------------------------------------------------------------


@dataclass
class _Loaded:
    model: Any
    pool: concurrent.futures.ThreadPoolExecutor
    lock: threading.Lock = field(default_factory=threading.Lock)


_CACHE: dict[tuple, _Loaded] = {}
_CACHE_LOCK = threading.Lock()


# ---- the plugin base -----------------------------------------------------------------------------


@dataclass
class _Play:
    """Rough playback bookkeeping shared between consumer and producer."""

    horizon: float = 0.0  # monotonic time when audio handed out so far finishes playing
    queued: float = 0.0  # seconds of audio produced but not yet handed out

    def ahead(self, now: float) -> float:
        return max(0.0, self.horizon - now) + self.queued


class StreamingTTS(TTS):
    """Base class: subclasses implement `_cache_key()`, `_load()` and `_synth(text, stop)`."""

    #: default English handling for this backend ("keep" for bilingual models)
    english_default = "spell"

    def __init__(
        self,
        first_min_chars: int = 4,
        first_max_chars: int = 14,
        min_chars: int = 10,
        max_chars: int = 60,
        first_wait_ms: int = 300,
        lead_ms: int = 250,
        max_ahead_s: float = 6.0,
        english: str | None = None,
        align: str = "anchor",  # "anchor" | "uniform" | "weight" | "segment"
        pause_tail_ms: int | None = None,
        **ignored: Any,
    ) -> None:
        self.seg_cfg = dict(first_min=first_min_chars, first_max=first_max_chars, min_chars=min_chars, max_chars=max_chars)
        self.first_wait = first_wait_ms / 1000
        self.lead = lead_ms / 1000
        self.max_ahead = max_ahead_s
        self.norm = Normalizer(english or self.english_default)
        self.align = align
        self.pause_tail_ms = pause_tail_ms
        self.ignored = sorted(ignored)  # e.g. mock's chars_per_sec when switched with --set tts.impl=...
        self._loaded: _Loaded | None = None
        self.stats: dict[str, Any] = {"segments": [], "cancelled_synth_s": 0.0}

    # -- backend hooks --
    def _cache_key(self) -> tuple:
        raise NotImplementedError

    def _load(self) -> Any:
        raise NotImplementedError

    def _synth(self, model: Any, text: str, stop: threading.Event):
        """Return (float32 samples, sample_rate[, marks]); runs in a worker thread.

        Check `stop` to abort early if possible. Optional `marks` are timing marks
        [(spoken text, start_s, end_s)] relative to the returned audio (exact alignment).
        """
        raise NotImplementedError

    #: backends doing network I/O can set this and implement `_synth_async` instead
    is_async = False

    async def _synth_async(self, model: Any, text: str, stop: threading.Event):
        raise NotImplementedError

    # -- lifecycle --
    async def start(self) -> None:
        key = (type(self).__name__,) + self._cache_key()

        def get_or_load() -> _Loaded:
            # Sessions running concurrently in one process (eval) share one model instance.
            with _CACHE_LOCK:
                if key not in _CACHE:
                    pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix=f"tts-{self.plugin_name}")
                    _CACHE[key] = _Loaded(self._load_and_warm(), pool)
                return _CACHE[key]

        self._loaded = await asyncio.to_thread(get_or_load)

    def _load_and_warm(self) -> Any:
        model = self._load()
        try:  # first ONNX Runtime run allocates buffers; keep that out of the first response
            self._synth(model, "你好。", threading.Event())
        except Exception:
            pass
        return model

    def synth_blocking(self, text: str):
        """Synchronous helper for scripts: normalized synthesis of a whole text at the model's rate."""
        if self._loaded is None:
            asyncio.run(self.start())
        return self._synth(self._loaded.model, spoken_text(self.norm.units(text)), threading.Event())

    # -- streaming --
    async def synthesize(self, text: AsyncIterator[str]) -> AsyncIterator[TTSChunk]:
        if self._loaded is None:
            await self.start()
        loop = asyncio.get_running_loop()
        out: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()
        play = _Play(horizon=time.monotonic())
        producer = asyncio.create_task(self._produce(text, out, stop, play))
        try:
            while True:
                item = await out.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                now = time.monotonic()
                play.queued = max(0.0, play.queued - item.duration)
                play.horizon = max(play.horizon, now) + item.duration
                yield item
        finally:
            stop.set()
            producer.cancel()
            del loop

    async def _produce(self, text: AsyncIterator[str], out: asyncio.Queue, stop: threading.Event, play: _Play) -> None:
        seg = Segmenter(**self.seg_cfg)
        pieces: asyncio.Queue = asyncio.Queue()

        async def reader() -> None:
            try:
                async for p in text:
                    pieces.put_nowait(p)
            finally:
                pieces.put_nowait(None)

        rd = asyncio.create_task(reader())
        ended = False
        first_text_at: float | None = None
        try:
            while True:
                now = time.monotonic()
                if seg.emitted == 0:
                    deadline = (first_text_at + self.first_wait) if first_text_at is not None else None
                else:
                    deadline = now + play.ahead(now) - self.lead
                urgent = deadline is not None and now >= deadline
                s = seg.pop(final=ended, urgent=urgent)
                if s is not None:
                    while play.ahead(time.monotonic()) > self.max_ahead and not stop.is_set():
                        await asyncio.sleep(0.05)
                    await self._synth_segment(s, out, stop, play)
                    continue
                if ended:
                    break
                timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
                if urgent:
                    timeout = 0.02  # nothing safe to cut yet: poll for more text
                try:
                    p = await asyncio.wait_for(pieces.get(), timeout)
                except asyncio.TimeoutError:
                    continue
                if p is None:
                    ended = True
                else:
                    if first_text_at is None and p.strip():
                        first_text_at = time.monotonic()
                    seg.feed(p)
                    while not pieces.empty():  # drain whatever already arrived
                        p = pieces.get_nowait()
                        if p is None:
                            ended = True
                            break
                        seg.feed(p)
            out.put_nowait(None)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # surface backend errors to the consumer
            out.put_nowait(e)
        finally:
            rd.cancel()

    async def _synth_segment(self, s: str, out: asyncio.Queue, stop: threading.Event, play: _Play) -> None:
        import numpy as np

        units = self.norm.units(s)
        spoken = spoken_text(units)
        t0 = time.monotonic()
        marks = None
        if any(not u.pause and u.spoken.strip() for u in units):
            if self.is_async:
                res = await self._synth_async(self._loaded.model, spoken, stop)
            else:
                res = await asyncio.get_running_loop().run_in_executor(self._loaded.pool, self._synth_locked, spoken, stop)
            x, sr = res[0], res[1]
            marks = res[2] if len(res) > 2 else None
            x = resample(np.asarray(x, dtype=np.float32), sr, SAMPLE_RATE)
            peak = float(np.abs(x).max()) if len(x) else 0.0
            if peak > 0.95:  # simple limiter: some models (zero-shot ones) overshoot and would clip
                x = x * (0.95 / peak)
        else:
            x = np.zeros(0, dtype=np.float32)
        dt = time.monotonic() - t0
        if stop.is_set():
            self.stats["cancelled_synth_s"] += dt
            return
        x, shift = self._shape_silence(x, units)
        if marks:
            marks = [(t, a - shift / SAMPLE_RATE, b - shift / SAMPLE_RATE) for t, a, b in marks]
        chunks = self._split(units, x, marks)
        dur = len(x) / SAMPLE_RATE
        self.stats["segments"].append({"text": s, "spoken": spoken, "synth_s": round(dt, 4), "audio_s": round(dur, 3)})
        for c in chunks:
            play.queued += c.duration
            out.put_nowait(c)

    def _synth_locked(self, text: str, stop: threading.Event):
        with self._loaded.lock:
            if stop.is_set():
                import numpy as np

                return np.zeros(0, dtype=np.float32), SAMPLE_RATE
            return self._synth(self._loaded.model, text, stop)

    def _shape_silence(self, x, units: list[Unit]):
        """Trim model-added silence; keep a pause proportional to the segment's final punctuation."""
        import numpy as np

        tail_w = 0.0
        for u in reversed(units):
            if not u.pause:
                break
            tail_w = max(tail_w, u.weight)
        tail_ms = self.pause_tail_ms if self.pause_tail_ms is not None else int(40 + 120 * tail_w)
        if len(x) == 0:
            return (np.zeros(int(SAMPLE_RATE * tail_ms / 1000), dtype=np.float32) if tail_w else x), 0
        y, shift = trim_silence(x, SAMPLE_RATE, lead_ms=20, tail_ms=30, with_offset=True)
        pad = np.zeros(int(SAMPLE_RATE * max(0, tail_ms - 30) / 1000), dtype=np.float32)
        return np.concatenate([y, pad]), shift

    def _split(self, units: list[Unit], x, marks: list | None = None) -> list[TTSChunk]:
        if not units:
            return []
        if self.align == "segment" or len(x) == 0:
            return [TTSChunk(to_pcm16(x), "".join(u.orig for u in units))]
        spans = None
        if marks and self.align in ("anchor", "marks"):
            spans = marks_to_spans(units, marks, len(x), SAMPLE_RATE)
        if spans is not None:
            pass
        elif self.align == "uniform":
            # naive baseline: every non-space character gets the same share of the audio
            ws = [Unit(u.orig, u.spoken, float(len(u.orig.strip())), u.pause) for u in units]
            spans = align_units(ws, x, SAMPLE_RATE, anchor=False)
        else:
            spans = align_units(units, x, SAMPLE_RATE, anchor=self.align == "anchor")
        chunks: list[TTSChunk] = []
        carry = ""  # text of zero-length units (quotes, spaces) rides on the next chunk
        for u, (a, b) in zip(units, spans):
            if b <= a:
                carry += u.orig
                continue
            chunks.append(TTSChunk(to_pcm16(x[a:b]), carry + u.orig))
            carry = ""
        if carry:
            if chunks:
                chunks[-1].text += carry
            else:
                chunks.append(TTSChunk(b"", carry))
        return chunks
