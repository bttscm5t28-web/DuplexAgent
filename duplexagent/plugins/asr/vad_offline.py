"""Pseudo-streaming ASR: Silero VAD segments the audio, a non-streaming model decodes each segment.

While a segment is still open, the growing audio is re-decoded every `partial_ms` (only when the
worker is idle, so a slow model degrades to fewer partials instead of falling behind). When the VAD
closes the segment (`endpoint_ms` of silence) the whole segment is decoded once more as the final.

    [asr]
    impl = "sherpa_offline"
    model = "sensevoice"        # preset below: sensevoice / paraformer-zh / firered-aed2 / firered-ctc2 / funasr-nano / ...
    endpoint_ms = 300
    partial_ms = 300

`VadSegmentASR` is reusable: subclasses only implement `_load_decoder()` and `_decode(samples)`
(see faster_whisper.py). Needs `pip install sherpa-onnx numpy huggingface_hub` (the VAD is sherpa-onnx's Silero).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

from duplexagent.core.interfaces import ASR
from duplexagent.core.registry import register
from duplexagent.core.types import AudioFrame, Transcript
from duplexagent.plugins.asr._util import Worker, cached, clean, hf_model, itn_zh, pcm_to_float

VAD_REPO = "csukuangfj/vad"  # silero_vad.onnx (v4) in the layout sherpa-onnx expects


def make_silero_vad(endpoint_ms: int, threshold: float = 0.5, min_speech_ms: int = 100, max_segment_s: float = 20.0):
    import sherpa_onnx

    d = hf_model(VAD_REPO, allow=["silero_vad.onnx"], subdir="vad")
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = os.path.join(d, "silero_vad.onnx")
    cfg.silero_vad.threshold = threshold
    cfg.silero_vad.min_silence_duration = endpoint_ms / 1000
    cfg.silero_vad.min_speech_duration = min_speech_ms / 1000
    cfg.silero_vad.max_speech_duration = max_segment_s
    cfg.silero_vad.window_size = 512
    cfg.sample_rate = 16000
    cfg.num_threads = 1
    return sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=max_segment_s + 10)


class VadSegmentASR(ASR):
    def __init__(
        self,
        endpoint_ms: int = 300,  # silence that closes a segment
        partial_ms: int = 300,  # re-decode the open segment after this much new audio (0 = no partials)
        vad_threshold: float = 0.5,
        min_speech_ms: int = 100,
        preroll_ms: int = 300,  # audio kept before the VAD onset
        max_segment_s: float = 20.0,
        num_threads: int = 2,
        itn: bool = False,
    ) -> None:
        self.endpoint_ms = endpoint_ms
        self.partial_ms = partial_ms
        self.vad_threshold = vad_threshold
        self.min_speech_ms = min_speech_ms
        self.preroll_ms = preroll_ms
        self.max_segment_s = max_segment_s
        self.num_threads = num_threads
        self.itn = itn

    # ---- subclass API -----------------------------------------------------------------

    def _cache_key(self) -> tuple:
        raise NotImplementedError

    def _load_decoder(self):
        """Blocking; runs once per process per cache key, in the worker thread."""
        raise NotImplementedError

    def _decode(self, samples) -> str:
        """Blocking; float32 16 kHz mono -> text. Runs in the worker thread."""
        raise NotImplementedError

    # ---- plumbing ---------------------------------------------------------------------

    def _make_vad(self):
        return make_silero_vad(self.endpoint_ms, self.vad_threshold, self.min_speech_ms, self.max_segment_s)

    async def start(self) -> None:
        self.worker = Worker()
        self.dec = await self.worker.run(cached, self._cache_key(), self._load_decoder)
        self.vad = await self.worker.run(self._make_vad)

    async def close(self) -> None:
        self.worker.close()

    def _run_decode(self, pcm: bytes) -> str:
        text = clean(self._decode(pcm_to_float(pcm))).lower()
        return itn_zh(text) if self.itn else text

    async def transcribe(self, frames: AsyncIterator[AudioFrame]) -> AsyncIterator[Transcript]:
        bps = 32  # bytes per ms at 16 kHz int16
        keep = (self.preroll_ms + self.min_speech_ms + 64) * bps
        partial_bytes = self.partial_ms * bps
        self.vad.reset()
        pcm = bytearray()
        in_speech, seg_start, last_launch, last_text = False, 0, 0, ""
        pending = None  # in-flight partial decode
        ts = 0.0
        async for f in frames:
            ts = f.ts
            pcm += f.pcm
            self.vad.accept_waveform(pcm_to_float(f.pcm))
            ended = not self.vad.empty()
            while not self.vad.empty():
                self.vad.pop()
            if not in_speech:
                if self.vad.is_speech_detected() or ended:
                    in_speech = True
                    seg_start = max(0, len(pcm) - keep)
                    last_launch = len(pcm) - partial_bytes  # first partial right away
                elif len(pcm) > 4 * keep:
                    del pcm[: len(pcm) - keep]
                    continue
                else:
                    continue
            if pending is not None and pending.done():
                text = pending.result() if not pending.cancelled() else ""
                pending = None
                if text and text != last_text and not ended:
                    last_text = text
                    yield Transcript(text, False, ts)
            if ended:
                if pending is not None:
                    pending.cancel()  # still queued -> dropped; already running -> result ignored
                    pending = None
                text = await self.worker.run(self._run_decode, bytes(pcm[seg_start:]))
                if text:
                    yield Transcript(text, True, ts)
                in_speech, last_text = False, ""
                del pcm[: max(0, len(pcm) - keep)]
                continue
            if self.partial_ms and pending is None and len(pcm) - last_launch >= partial_bytes:
                pending = self.worker.submit(self._run_decode, bytes(pcm[seg_start:]))
                last_launch = len(pcm)
        # input ended: close any open segment
        self.vad.flush()
        if pending is not None:
            pending.cancel()
        if in_speech:
            text = await self.worker.run(self._run_decode, bytes(pcm[seg_start:]))
            if text:
                yield Transcript(text, True, ts)
        self.vad.reset()


# ---- sherpa-onnx offline models -----------------------------------------------------------

PRESETS: dict[str, dict] = {
    # SenseVoice-Small（FunAudioLLM，中英日韩粤，非自回归，自带 ITN）
    "sensevoice": dict(repo="csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17", type="sense_voice", model="model.int8.onnx"),
    "sensevoice-2025": dict(repo="csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09", type="sense_voice", model="model.int8.onnx"),
    # Paraformer-large 离线（FunASR），非自回归
    "paraformer-zh": dict(repo="csukuangfj/sherpa-onnx-paraformer-zh-int8-2025-10-07", type="paraformer", model="model.int8.onnx"),
    # FireRedASR（小红书）。v1 AED-L 1.1B；v2 2026 AED / CTC
    "firered-aed": dict(repo="csukuangfj/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16", type="fire_red_asr", encoder="encoder.int8.onnx", decoder="decoder.int8.onnx"),
    "firered-aed2": dict(repo="csukuangfj2/sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26", type="fire_red_asr", encoder="encoder.int8.onnx", decoder="decoder.int8.onnx"),
    "firered-ctc2": dict(repo="csukuangfj2/sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25", type="fire_red_asr_ctc", model="model.int8.onnx"),
    # Qwen3-ASR-0.6B（通义千问，2026，LLM-ASR）
    "qwen3-asr": dict(
        repo="csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25", type="qwen3_asr",
        conv_frontend="conv_frontend.onnx", encoder="encoder.int8.onnx", decoder="decoder.int8.onnx", tokenizer="tokenizer",
    ),
    # Fun-ASR-Nano（通义，SenseVoice 编码器 + Qwen3-0.6B，LLM-ASR）
    "funasr-nano": dict(
        repo="csukuangfj/sherpa-onnx-funasr-nano-int8-2025-12-30", type="funasr_nano",
        encoder_adaptor="encoder_adaptor.int8.onnx", llm="llm.int8.onnx", embedding="embedding.int8.onnx", tokenizer="Qwen3-0.6B",
    ),
}


@register("asr", "sherpa_offline")
class SherpaOfflineASR(VadSegmentASR):
    def __init__(self, model: str = "sensevoice", language: str = "auto", provider: str = "cpu", files: dict | None = None, **kw) -> None:
        super().__init__(**kw)
        if model not in PRESETS and not files:
            raise ValueError(f"unknown sherpa_offline model {model!r}; presets: {sorted(PRESETS)}")
        self.spec = {**PRESETS.get(model, {}), **(files or {})}
        self.model = model
        self.language = language
        self.provider = provider

    def _cache_key(self) -> tuple:
        # ITN is a decoder option for SenseVoice / FunASR-Nano, so it is part of the key
        return ("sherpa_offline", tuple(sorted(self.spec.items())), self.num_threads, self.language, self.itn, self.provider)

    def _load_decoder(self):
        import sherpa_onnx

        s = self.spec
        names = [v for k, v in s.items() if k in ("model", "encoder", "decoder", "encoder_adaptor", "llm", "embedding", "conv_frontend")]
        allow = names + ([f"{s['tokenizer']}/*"] if "tokenizer" in s else ["tokens.txt"])
        d = hf_model(s["repo"], allow=allow)
        p = lambda k: os.path.join(d, s[k])  # noqa: E731
        tokens = os.path.join(d, "tokens.txt")
        common = dict(num_threads=self.num_threads, provider=self.provider)
        t = s["type"]
        if t == "sense_voice":
            lang = "" if self.language == "auto" else self.language
            return sherpa_onnx.OfflineRecognizer.from_sense_voice(model=p("model"), tokens=tokens, language=lang, use_itn=self.itn, **common)
        if t == "paraformer":
            return sherpa_onnx.OfflineRecognizer.from_paraformer(paraformer=p("model"), tokens=tokens, **common)
        if t == "fire_red_asr":
            return sherpa_onnx.OfflineRecognizer.from_fire_red_asr(encoder=p("encoder"), decoder=p("decoder"), tokens=tokens, **common)
        if t == "fire_red_asr_ctc":
            return sherpa_onnx.OfflineRecognizer.from_fire_red_asr_ctc(model=p("model"), tokens=tokens, **common)
        if t == "funasr_nano":
            return sherpa_onnx.OfflineRecognizer.from_funasr_nano(
                encoder_adaptor=p("encoder_adaptor"), llm=p("llm"), embedding=p("embedding"), tokenizer=p("tokenizer"), itn=self.itn, **common
            )
        if t == "qwen3_asr":
            return sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
                conv_frontend=p("conv_frontend"), encoder=p("encoder"), decoder=p("decoder"), tokenizer=p("tokenizer"), **common
            )
        raise ValueError(f"unknown sherpa_offline model type {t!r}")

    def _decode(self, samples) -> str:
        s = self.dec.create_stream()
        s.accept_waveform(16000, samples)
        self.dec.decode_stream(s)
        return s.result.text
