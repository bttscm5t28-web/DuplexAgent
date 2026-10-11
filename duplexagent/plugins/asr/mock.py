"""Mock ASR: reads the simulator's ground truth from frames. Optional synthetic errors."""

from __future__ import annotations

import random
from collections.abc import AsyncIterator

from duplexagent.core.audio import rms
from duplexagent.core.interfaces import ASR
from duplexagent.core.registry import register
from duplexagent.core.types import FRAME_MS, AudioFrame, Transcript


@register("asr", "mock")
class MockASR(ASR):
    def __init__(self, endpoint_ms: int = 160, vad_threshold: float = 0.02, error_rate: float = 0.0, seed: int = 0):
        self.endpoint_ms = endpoint_ms
        self.vad_threshold = vad_threshold
        self.error_rate = error_rate
        self._rnd = random.Random(seed)

    def _corrupt(self, ch: str) -> str:
        if self.error_rate and self._rnd.random() < self.error_rate:
            return self._rnd.choice(["", "的", ch + ch])
        return ch

    async def transcribe(self, frames: AsyncIterator[AudioFrame]) -> AsyncIterator[Transcript]:
        text = ""
        silent_ms = 0
        async for f in frames:
            if f.truth:
                text += "".join(self._corrupt(c) for c in f.truth)
                yield Transcript(text, False, f.ts)
            if rms(f.pcm) >= self.vad_threshold:
                silent_ms = 0
            else:
                silent_ms += FRAME_MS
                if text and silent_ms >= self.endpoint_ms:
                    yield Transcript(text, True, f.ts)
                    text = ""
