"""OpenAI-compatible speech endpoint (`POST {base_url}/audio/speech`), stdlib HTTP only.

Covers OpenAI itself and self-hosted servers exposing the same API, which is how GPU-heavy
candidates (CosyVoice 2/3, Fish-Speech/OpenAudio, IndexTTS-2, F5-TTS, ...) plug in without
pulling their dependencies into this process:

    [tts]
    impl = "openai_speech"
    base_url = "http://gpu-box:8000/v1"
    model = "cosyvoice2"            # whatever the server expects
    voice = "中文女"
    response_format = "pcm"         # raw s16le; set sample_rate to the server's rate
    sample_rate = 24000
    api_key_env = "TTS_API_KEY"     # optional

One request per segment (the StreamingTTS base does segmentation and alignment), so the first
segment is kept short. Status: interface verified against a local stub server only; real
servers are pending tests (no GPU / API key in this environment).
"""

from __future__ import annotations

import json
import os
import threading
import urllib.request
from typing import Any

from duplexagent.core.registry import register
from duplexagent.plugins.tts.streaming import StreamingTTS


@register("tts", "openai_speech")
class OpenAISpeechTTS(StreamingTTS):
    english_default = "keep"

    def __init__(self, base_url: str = "https://api.openai.com/v1", model: str = "gpt-4o-mini-tts", voice: str = "alloy",
                 response_format: str = "pcm", sample_rate: int = 24000, speed: float = 1.0, api_key_env: str = "OPENAI_API_KEY",
                 timeout: float = 30.0, extra_body: dict | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.url = base_url.rstrip("/") + "/audio/speech"
        self.model, self.voice, self.fmt, self.sr, self.speed = model, voice, response_format, sample_rate, speed
        self.api_key_env, self.timeout, self.extra_body = api_key_env, timeout, extra_body or {}

    def _cache_key(self) -> tuple:
        return (self.url, self.model, self.voice)

    def _load(self) -> Any:
        return None

    def _synth(self, model: Any, text: str, stop: threading.Event):
        import io
        import wave

        import numpy as np

        body = {"model": self.model, "input": text, "voice": self.voice, "response_format": self.fmt, "speed": self.speed, **self.extra_body}
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers, method="POST")
        data = bytearray()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            while not stop.is_set():
                b = r.read(8192)
                if not b:
                    break
                data += b
        if self.fmt == "wav":
            with wave.open(io.BytesIO(bytes(data))) as w:
                sr, ch = w.getframerate(), w.getnchannels()
                x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
                return (x.reshape(-1, ch).mean(axis=1) if ch > 1 else x), sr
        if self.fmt == "mp3":
            from duplexagent.plugins.tts.edge import decode_mp3

            return decode_mp3(bytes(data))
        n = len(data) // 2 * 2
        return np.frombuffer(bytes(data[:n]), dtype="<i2").astype(np.float32) / 32768, self.sr
