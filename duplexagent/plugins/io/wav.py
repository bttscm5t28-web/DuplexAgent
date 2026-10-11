"""WAV file input (16 kHz mono int16, streamed in real time) and WAV recording output."""

from __future__ import annotations

import asyncio
import wave
from collections.abc import AsyncIterator

from duplexagent.core.audio import silence
from duplexagent.core.interfaces import AudioInput, AudioOutput
from duplexagent.core.registry import register
from duplexagent.core.types import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, AudioFrame, TTSChunk


@register("input", "wav")
class WavInput(AudioInput):
    def __init__(self, path: str, tail_s: float = 10.0, realtime: bool = True) -> None:
        self.path, self.tail_s, self.realtime = path, tail_s, realtime

    async def frames(self) -> AsyncIterator[AudioFrame]:
        with wave.open(self.path, "rb") as w:
            if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
                raise ValueError(f"{self.path}: need {SAMPLE_RATE} Hz mono int16")
            data = w.readframes(w.getnframes())
        step = FRAME_SAMPLES * 2
        tail = silence() * int(self.tail_s * 1000 / FRAME_MS)
        data += tail
        loop = asyncio.get_running_loop()
        start = loop.time()
        for i in range(0, len(data) - step + 1, step):
            if self.realtime:
                delay = start + (i // step) * FRAME_MS / 1000 - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
            yield AudioFrame(pcm=data[i : i + step], ts=self.clock())


@register("output", "wav")
class WavOutput(AudioOutput):
    """Paces like a speaker and writes agent audio at its real position on the session timeline."""

    def __init__(self, path: str = "agent_out.wav") -> None:
        self.path = path
        self._buf = bytearray()

    async def play(self, chunk: TTSChunk) -> None:
        pos = int(self.clock() * SAMPLE_RATE) * 2
        if pos > len(self._buf):
            self._buf.extend(b"\x00" * (pos - len(self._buf)))
        self._buf[pos : pos + len(chunk.pcm)] = chunk.pcm
        await asyncio.sleep(chunk.duration)

    async def close(self) -> None:
        with wave.open(self.path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(bytes(self._buf))
