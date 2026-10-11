"""Microsoft Edge "Read aloud" online voices via the `edge-tts` package (no API key).

    [tts]
    impl = "edge"
    voice = "zh-CN-XiaoxiaoNeural"
    rate = "+0%"

Word boundaries from the service give exact per-word timing, so TTSChunk.text alignment does
not depend on energy heuristics. Caveats: needs Internet, one websocket per segment (first
packet ~0.5-1 s), an unofficial endpoint with no SLA — a quality/alignment reference rather
than a production choice. Behind a TLS-inspecting proxy set `ca_file` (defaults to
$SSL_CERT_FILE) and `proxy` (defaults to $HTTPS_PROXY).
"""

from __future__ import annotations

import os
import threading
from typing import Any

from duplexagent.core.registry import register
from duplexagent.plugins.tts.streaming import StreamingTTS


def decode_mp3(data: bytes):
    """MP3 bytes -> (float32 mono samples, sample rate). miniaudio, else the ffmpeg CLI."""
    import numpy as np

    try:
        import miniaudio

        d = miniaudio.decode(data, output_format=miniaudio.SampleFormat.SIGNED16, nchannels=1)
        return np.frombuffer(d.samples, dtype=np.int16).astype(np.float32) / 32768, d.sample_rate
    except ImportError:
        import subprocess

        p = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", "24000", "pipe:1"],
                           input=data, capture_output=True, check=True)
        return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768, 24000


@register("tts", "edge")
class EdgeTTS(StreamingTTS):
    english_default = "keep"
    is_async = True

    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural", rate: str = "+0%", pitch: str = "+0Hz",
                 ca_file: str | None = None, proxy: str | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.voice, self.rate, self.pitch = voice, rate, pitch
        self.ca_file = ca_file or os.environ.get("SSL_CERT_FILE")
        self.proxy = proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")

    def _cache_key(self) -> tuple:
        return (self.voice,)

    def _load(self) -> Any:
        import ssl

        import edge_tts.communicate as ec

        if self.ca_file:
            ec._SSL_CTX = ssl.create_default_context(cafile=self.ca_file)
        return None

    async def _synth_async(self, model: Any, text: str, stop: threading.Event):
        import edge_tts

        c = edge_tts.Communicate(text, self.voice, rate=self.rate, pitch=self.pitch, proxy=self.proxy, boundary="WordBoundary")
        audio, marks = bytearray(), []
        async for m in c.stream():
            if stop.is_set():
                break
            if m["type"] == "audio":
                audio += m["data"]
            elif m["type"] == "WordBoundary":
                a = m["offset"] / 1e7
                marks.append((m["text"], a, a + m["duration"] / 1e7))
        x, sr = decode_mp3(bytes(audio)) if audio else (__import__("numpy").zeros(0, dtype="float32"), 24000)
        return x, sr, marks
