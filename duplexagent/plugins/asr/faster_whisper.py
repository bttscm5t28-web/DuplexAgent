"""Pseudo-streaming Whisper via faster-whisper (CTranslate2): Silero VAD segments + re-decoding.

    [asr]
    impl = "faster_whisper"
    model = "small"            # tiny / base / small / medium / large-v3 / large-v3-turbo, or a local path
    compute_type = "int8"

Needs `pip install faster-whisper sherpa-onnx numpy` (sherpa-onnx only for the shared Silero VAD).
Whisper decodes a 30 s window regardless of input length, so each partial costs a full encoder pass;
on CPU keep `partial_ms` large or use a small model.
"""

from __future__ import annotations

import os

from duplexagent.core.registry import register
from duplexagent.plugins.asr._util import MODELS_DIR
from duplexagent.plugins.asr.vad_offline import VadSegmentASR

REPOS = {"large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo"}


@register("asr", "faster_whisper")
class FasterWhisperASR(VadSegmentASR):
    def __init__(
        self,
        model: str = "small",
        compute_type: str = "int8",
        device: str = "cpu",
        language: str | None = "zh",
        beam_size: int = 1,
        initial_prompt: str | None = "以下是普通话的句子。",
        **kw,
    ) -> None:
        kw.setdefault("partial_ms", 600)
        super().__init__(**kw)
        self.model, self.compute_type, self.device = model, compute_type, device
        self.language, self.beam_size, self.initial_prompt = language, beam_size, initial_prompt

    def _cache_key(self) -> tuple:
        return ("faster_whisper", self.model, self.compute_type, self.device, self.num_threads)

    def _load_decoder(self):
        from faster_whisper import WhisperModel

        return WhisperModel(
            REPOS.get(self.model, self.model), device=self.device, compute_type=self.compute_type,
            cpu_threads=self.num_threads, download_root=os.path.join(MODELS_DIR, "asr", "faster-whisper"),
        )

    def _decode(self, samples) -> str:
        segs, _ = self.dec.transcribe(
            samples, language=self.language, beam_size=self.beam_size, initial_prompt=self.initial_prompt,
            condition_on_previous_text=False, without_timestamps=True, vad_filter=False, temperature=0.0,
        )
        return "".join(s.text for s in segs)
