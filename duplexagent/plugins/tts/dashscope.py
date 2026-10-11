"""Alibaba Cloud Bailian / DashScope CosyVoice (cloud) via the `dashscope` SDK.

    [tts]
    impl = "dashscope"
    model = "cosyvoice-v2"           # or cosyvoice-v3 / cosyvoice-v3-plus
    voice = "longxiaochun_v2"
    # API key: $DASHSCOPE_API_KEY

One synthesis call per segment (the StreamingTTS base does segmentation and alignment), audio
requested as 16 kHz PCM so no resampling is needed. The service also supports bidirectional
streaming (streaming_call / streaming_complete) and word timestamps; using them would save
one connection set-up per segment and give exact alignment — left for when a key is available.

Status: written against dashscope SDK 1.2x; NOT run here (no API key) — pending tests.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from duplexagent.core.registry import register
from duplexagent.plugins.tts.streaming import StreamingTTS


@register("tts", "dashscope")
class DashScopeTTS(StreamingTTS):
    english_default = "keep"

    def __init__(self, model: str = "cosyvoice-v2", voice: str = "longxiaochun_v2", speech_rate: float = 1.0,
                 api_key_env: str = "DASHSCOPE_API_KEY", timeout_ms: int = 15000, **kw: Any) -> None:
        super().__init__(**kw)
        self.model, self.voice, self.speech_rate = model, voice, speech_rate
        self.api_key_env, self.timeout_ms = api_key_env, timeout_ms

    def _cache_key(self) -> tuple:
        return (self.model, self.voice)

    def _load(self) -> Any:
        import dashscope

        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"tts.dashscope needs ${self.api_key_env}")
        dashscope.api_key = key
        return None

    def _synth(self, model: Any, text: str, stop: threading.Event):
        import numpy as np
        from dashscope.audio.tts_v2 import AudioFormat, SpeechSynthesizer

        s = SpeechSynthesizer(model=self.model, voice=self.voice, format=AudioFormat.PCM_16000HZ_MONO_16BIT, speech_rate=self.speech_rate)
        data = s.call(text, timeout_millis=self.timeout_ms) or b""
        n = len(data) // 2 * 2
        return np.frombuffer(data[:n], dtype="<i2").astype(np.float32) / 32768, 16000
