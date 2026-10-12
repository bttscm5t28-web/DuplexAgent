"""Local microphone and speaker via sounddevice (PortAudio). `pip install sounddevice numpy`.

There is no echo cancellation here: use headphones, or use the `browser` input/output
(the browser's WebRTC stack cancels echo for you).
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator

from duplexagent.core.interfaces import AudioInput, AudioOutput
from duplexagent.core.registry import register
from duplexagent.core.types import FRAME_SAMPLES, SAMPLE_RATE, AudioFrame, TTSChunk


def _write_wav(path: str, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> None:
    import os
    import wave

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


def resample_int16(pcm: bytes, src: int, dst: int) -> bytes:
    if src == dst or not pcm:
        return pcm
    import numpy as np

    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    n = max(1, round(len(x) * dst / src))
    y = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return y.astype(np.int16).tobytes()


@register("input", "mic")
class MicInput(AudioInput):
    def __init__(self, device: int | str | None = None, max_seconds: float | None = None, record_path: str | None = None) -> None:
        self.device = device
        self.max_seconds = max_seconds
        self.record_path = record_path  # keep the raw mic audio (aligned with the timeline) for labeling
        self._rec = bytearray()

    async def close(self) -> None:
        if self.record_path:
            _write_wav(self.record_path, bytes(self._rec))

    async def frames(self) -> AsyncIterator[AudioFrame]:
        import sounddevice as sd

        loop = asyncio.get_running_loop()
        q: asyncio.Queue[bytes] = asyncio.Queue()

        def cb(indata, frames, time_info, status) -> None:
            loop.call_soon_threadsafe(q.put_nowait, bytes(indata))

        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=FRAME_SAMPLES, channels=1, dtype="int16", device=self.device, callback=cb):
            print("[mic] listening — Ctrl+C to stop", flush=True)
            while self.max_seconds is None or self.clock() < self.max_seconds:
                pcm = await q.get()
                if self.record_path:
                    self._rec += pcm
                yield AudioFrame(pcm=pcm, ts=self.clock())


@register("output", "speaker")
class SpeakerOutput(AudioOutput):
    """Plays through a PortAudio stream; play() returns once the chunk has (almost) left the buffer,
    so the engine's played-text accounting matches what the user actually heard."""

    def __init__(self, device: int | str | None = None, sample_rate: int = SAMPLE_RATE, lead_ms: int = 60) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.lead_bytes = int(sample_rate * lead_ms / 1000) * 2
        self._buf = bytearray()
        self._lock = threading.Lock()

    async def start(self) -> None:
        import sounddevice as sd

        def cb(outdata, frames, time_info, status) -> None:
            n = len(outdata)
            with self._lock:
                take = bytes(self._buf[:n])
                del self._buf[:n]
            outdata[: len(take)] = take
            if len(take) < n:
                outdata[len(take) :] = b"\x00" * (n - len(take))

        self._stream = sd.RawOutputStream(samplerate=self.sample_rate, channels=1, dtype="int16", device=self.device, callback=cb)
        self._stream.start()

    async def play(self, chunk: TTSChunk) -> None:
        pcm = resample_int16(chunk.pcm, chunk.sample_rate, self.sample_rate)
        with self._lock:
            self._buf += pcm
        while True:
            with self._lock:
                pending = len(self._buf)
            if pending <= self.lead_bytes:
                return
            await asyncio.sleep(0.01)

    async def stop(self) -> None:
        with self._lock:
            self._buf.clear()

    async def close(self) -> None:
        self._stream.stop()
        self._stream.close()
