"""ElevenLabs Scribe v2 Realtime streaming ASR over WebSocket.

    [asr]
    impl = "elevenlabs"
    language_code = "zh"         # optional; auto-detect when omitted
    commit_strategy = "vad"      # "vad" | "manual" | "turn_prediction"

Partial transcripts become `Transcript(is_final=False)`, committed ones `is_final=True`. With
`commit_strategy = "vad"` the server endpoints segments itself; our turn detector still decides the
turn. Needs `pip install websockets` and $ELEVENLABS_API_KEY. Tested against a local mock only.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from collections.abc import AsyncIterator
from urllib.parse import urlencode

from duplexagent.core.interfaces import ASR
from duplexagent.core.registry import register
from duplexagent.core.types import FRAME_MS, SAMPLE_RATE, AudioFrame, Transcript

_ERRORS = {"error", "auth_error", "quota_exceeded", "commit_throttled", "unaccepted_terms", "rate_limited",
           "queue_overflow", "resource_exhausted", "session_time_limit_exceeded", "input_error",
           "invalid_request", "chunk_size_exceeded", "transcriber_error"}


@register("asr", "elevenlabs")
class ElevenLabsASR(ASR):
    def __init__(
        self,
        model_id: str = "scribe_v2_realtime",
        language_code: str | None = None,
        commit_strategy: str = "vad",
        vad_silence_threshold_secs: float | None = 0.3,
        keyterms: list[str] | None = None,
        api_key_env: str = "ELEVENLABS_API_KEY",
        base_url: str = "wss://api.elevenlabs.io",
        send_ms: int = 40,  # audio per message
    ) -> None:
        self.q = {"model_id": model_id, "audio_format": f"pcm_{SAMPLE_RATE}", "commit_strategy": commit_strategy}
        if language_code:
            self.q["language_code"] = language_code
        if vad_silence_threshold_secs is not None and commit_strategy == "vad":
            self.q["vad_silence_threshold_secs"] = vad_silence_threshold_secs
        self.keyterms = keyterms or []
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.frames_per_msg = max(1, send_ms // FRAME_MS)

    def _url(self) -> str:
        q = list(self.q.items()) + [("keyterms", k) for k in self.keyterms]
        return f"{self.base_url}/v1/speech-to-text/realtime?{urlencode(q)}"

    async def transcribe(self, frames: AsyncIterator[AudioFrame]) -> AsyncIterator[Transcript]:
        from websockets.asyncio.client import connect

        key = os.environ.get(self.api_key_env, "")
        last_ts = 0.0
        async with connect(self._url(), additional_headers={"xi-api-key": key} if key else None) as ws:

            async def send() -> None:
                nonlocal last_ts
                buf = b""
                n = 0
                async for f in frames:
                    last_ts = f.ts
                    buf += f.pcm
                    n += 1
                    if n >= self.frames_per_msg:
                        await ws.send(json.dumps({"message_type": "input_audio_chunk", "audio_base_64": base64.b64encode(buf).decode(),
                                                  "commit": False, "sample_rate": SAMPLE_RATE}))
                        buf, n = b"", 0
                await ws.send(json.dumps({"message_type": "input_audio_chunk", "audio_base_64": base64.b64encode(buf).decode(),
                                          "commit": True, "sample_rate": SAMPLE_RATE}))

            sender = asyncio.create_task(send())
            try:
                while True:
                    recv = asyncio.create_task(ws.recv())
                    done, _ = await asyncio.wait({recv, sender}, return_when=asyncio.FIRST_COMPLETED)
                    if recv not in done:  # input ended: drain what the server still sends, briefly
                        if sender.exception():
                            recv.cancel()
                            raise sender.exception()
                        try:
                            raw = await asyncio.wait_for(recv, timeout=2.0)
                        except (asyncio.TimeoutError, Exception):
                            return
                    else:
                        try:
                            raw = recv.result()
                        except Exception:  # connection closed by the server
                            return
                    msg = json.loads(raw)
                    kind = msg.get("message_type")
                    if kind == "partial_transcript" and msg.get("text"):
                        yield Transcript(msg["text"], False, last_ts)
                    elif kind in ("committed_transcript", "committed_transcript_with_timestamps") and msg.get("text"):
                        yield Transcript(msg["text"], True, last_ts)
                    elif kind in _ERRORS:
                        raise RuntimeError(f"elevenlabs asr {kind}: {msg.get('error')}")
            finally:
                sender.cancel()
