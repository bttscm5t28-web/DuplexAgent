"""ElevenLabs streaming TTS over the `stream-input` WebSocket (text in, PCM + character alignment out).

One WebSocket per agent turn: the fast brain's text goes in as it arrives (grouped into phrases by
the shared `Segmenter`, each flushed so the first audio comes back quickly) and 16 kHz PCM comes back
with per-character timestamps, which become per-character `TTSChunk`s, so barge-in truncation is exact.
A spare connection is opened ahead of time to hide the handshake.

    [tts]
    impl = "elevenlabs"
    voice_id = "..."                 # pick a voice that speaks Chinese well
    model_id = "eleven_flash_v2_5"   # the low-latency model

Needs `pip install websockets` and the API key in $ELEVENLABS_API_KEY. No key was available when
this was written: the protocol follows the API reference and is tested against a local mock only.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from collections.abc import AsyncIterator
from urllib.parse import urlencode

from duplexagent.core.interfaces import TTS
from duplexagent.core.registry import register
from duplexagent.core.types import SAMPLE_RATE, TTSChunk
from duplexagent.plugins.tts.streaming import Segmenter


def split_by_alignment(pcm: bytes, alignment: dict | None, sr: int = SAMPLE_RATE) -> list[TTSChunk]:
    """Cut one audio message into per-character chunks using its (chunk-relative) alignment."""
    if not pcm:
        return []
    chars = (alignment or {}).get("chars") or []
    starts = (alignment or {}).get("charStartTimesMs") or []
    if not chars or len(starts) != len(chars):
        return [TTSChunk(pcm, "", sample_rate=sr)]
    n = len(pcm) // 2
    cuts = [min(n, max(0, int(ms * sr / 1000))) for ms in starts]
    cuts[0] = 0  # leading silence belongs to the first character
    cuts.append(n)
    out = []
    for i, ch in enumerate(chars):
        a, b = cuts[i], max(cuts[i], cuts[i + 1])
        if b > a or ch:
            out.append(TTSChunk(pcm[a * 2 : b * 2], ch, sample_rate=sr))
    return out


@register("tts", "elevenlabs")
class ElevenLabsTTS(TTS):
    def __init__(
        self,
        voice_id: str,
        model_id: str = "eleven_flash_v2_5",
        language_code: str | None = None,
        api_key_env: str = "ELEVENLABS_API_KEY",
        base_url: str = "wss://api.elevenlabs.io",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        speed: float = 1.0,
        chunk_length_schedule: list[int] | None = None,
        first_min: int = 4,
        first_max: int = 14,
        prewarm: bool = True,
        spare_max_age_s: float = 150.0,  # server closes idle sockets after inactivity_timeout (180 s)
    ) -> None:
        if not voice_id or voice_id.startswith("REPLACE"):
            raise ValueError("[tts] elevenlabs needs a voice_id")
        self.voice_id = voice_id
        self.model_id = model_id
        self.language_code = language_code
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.voice_settings = {"stability": stability, "similarity_boost": similarity_boost, "speed": speed}
        self.chunk_length_schedule = chunk_length_schedule or [50, 90, 120, 160]
        self.first_min, self.first_max = first_min, first_max
        self.prewarm = prewarm
        self.spare_max_age_s = spare_max_age_s
        self._spare: asyncio.Task | None = None
        self._spare_at = 0.0

    def _url(self) -> str:
        q = {"model_id": self.model_id, "output_format": f"pcm_{SAMPLE_RATE}", "sync_alignment": "true", "inactivity_timeout": 180}
        if self.language_code:
            q["language_code"] = self.language_code
        return f"{self.base_url}/v1/text-to-speech/{self.voice_id}/stream-input?{urlencode(q)}"

    async def _connect(self):
        from websockets.asyncio.client import connect

        key = os.environ.get(self.api_key_env, "")
        ws = await connect(self._url(), additional_headers={"xi-api-key": key} if key else None, max_size=2**23)
        await ws.send(json.dumps({"text": " ", "voice_settings": self.voice_settings,
                                  "generation_config": {"chunk_length_schedule": self.chunk_length_schedule}}))
        return ws

    def _arm_spare(self) -> None:
        if self.prewarm:
            self._spare = asyncio.create_task(self._connect())
            self._spare_at = time.monotonic()

    async def _take(self):
        spare, self._spare = self._spare, None
        fresh = spare is not None and time.monotonic() - self._spare_at < self.spare_max_age_s
        self._arm_spare()
        if spare is not None:
            try:
                ws = await spare
                if fresh:
                    return ws
                await ws.close()
            except Exception:
                pass
        return await self._connect()

    async def start(self) -> None:
        self._arm_spare()

    async def close(self) -> None:
        if self._spare is not None:
            self._spare.cancel()
            try:
                ws = await self._spare
                await ws.close()
            except BaseException:
                pass

    async def synthesize(self, text: AsyncIterator[str]) -> AsyncIterator[TTSChunk]:
        ws = await self._take()
        seg = Segmenter(first_min=self.first_min, first_max=self.first_max)

        async def send() -> None:
            async def push(s: str) -> None:
                await ws.send(json.dumps({"text": s + " ", "flush": True}))

            async for piece in text:
                seg.feed(piece)
                while (s := seg.pop()) is not None:
                    await push(s)
            while (s := seg.pop(final=True)) is not None:
                await push(s)
            await ws.send(json.dumps({"text": ""}))  # end of input: server finishes, sends isFinal

        sender = asyncio.create_task(send())
        try:
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("audio"):
                    for chunk in split_by_alignment(base64.b64decode(msg["audio"]), msg.get("alignment") or msg.get("normalizedAlignment")):
                        yield chunk
                if msg.get("isFinal"):
                    break
                if msg.get("error") or msg.get("message_type") == "error":
                    raise RuntimeError(f"elevenlabs: {msg}")
            if sender.done() and sender.exception():
                raise sender.exception()
        finally:
            sender.cancel()
            await ws.close()
