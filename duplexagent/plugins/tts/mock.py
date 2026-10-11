"""Mock TTS: one short tone per character at a speaking rate, with exact text alignment."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from duplexagent.core.audio import tone
from duplexagent.core.interfaces import TTS
from duplexagent.core.registry import register
from duplexagent.core.types import TTSChunk


@register("tts", "mock")
class MockTTS(TTS):
    def __init__(self, chars_per_sec: float = 6.0, first_chunk_ms: int = 80) -> None:
        self.chars_per_sec = chars_per_sec
        self.first_chunk_ms = first_chunk_ms
        self._pcm = tone(1 / chars_per_sec)

    async def synthesize(self, text: AsyncIterator[str]) -> AsyncIterator[TTSChunk]:
        first = True
        async for piece in text:
            for ch in piece:
                if first:
                    await asyncio.sleep(self.first_chunk_ms / 1000)
                    first = False
                yield TTSChunk(self._pcm, ch)
