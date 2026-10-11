"""FunASR native streaming Paraformer (paraformer-zh-streaming, PyTorch), endpointed by Silero VAD.

    [asr]
    impl = "funasr_streaming"
    chunk_size = [0, 10, 5]     # 600 ms chunks, 300 ms lookahead (FunASR default)
    endpoint_ms = 300

The model streams chunk by chunk inside a VAD segment (cache carried across chunks); when the VAD
closes the segment the remaining audio is decoded with is_final=True and the cache is reset.
Needs `pip install funasr torch torchaudio --index-url https://download.pytorch.org/whl/cpu` (+ sherpa-onnx for the VAD).
The same weights exported to ONNX are available as `sherpa_online` model "paraformer-bilingual" (no torch).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from duplexagent.core.registry import register
from duplexagent.core.types import AudioFrame, Transcript
from duplexagent.plugins.asr._util import clean, hf_model, itn_zh, pcm_to_float
from duplexagent.plugins.asr.vad_offline import VadSegmentASR


@register("asr", "funasr_streaming")
class FunASRStreaming(VadSegmentASR):
    def __init__(
        self,
        repo: str = "funasr/paraformer-zh-streaming",
        chunk_size: list[int] | None = None,
        encoder_chunk_look_back: int = 4,
        decoder_chunk_look_back: int = 1,
        **kw,
    ) -> None:
        super().__init__(**kw)
        self.repo = repo
        self.chunk_size = chunk_size or [0, 10, 5]
        self.enc_lb, self.dec_lb = encoder_chunk_look_back, decoder_chunk_look_back
        self.stride = self.chunk_size[1] * 960  # samples (60 ms per unit)

    def _cache_key(self) -> tuple:
        return ("funasr_streaming", self.repo, self.num_threads)

    def _load_decoder(self):
        import torch
        from funasr import AutoModel

        torch.set_num_threads(self.num_threads)
        d = hf_model(self.repo, allow=["model.pt", "config.yaml", "am.mvn", "tokens.json", "seg_dict", "configuration.json"])
        return AutoModel(model=d, disable_update=True, disable_pbar=True, log_level="ERROR")

    def _chunk(self, samples, cache: dict, final: bool) -> str:
        res = self.dec.generate(
            input=samples, cache=cache, is_final=final, chunk_size=self.chunk_size,
            encoder_chunk_look_back=self.enc_lb, decoder_chunk_look_back=self.dec_lb, disable_pbar=True,
        )
        return res[0]["text"] if res else ""

    def _post(self, text: str) -> str:
        text = clean(text).lower()
        return itn_zh(text) if self.itn else text

    async def transcribe(self, frames: AsyncIterator[AudioFrame]) -> AsyncIterator[Transcript]:
        import numpy as np

        keep = int((self.preroll_ms + self.min_speech_ms + 64) * 16)  # samples
        self.vad.reset()
        pre = np.zeros(0, np.float32)  # audio before the onset (pre-roll)
        buf = np.zeros(0, np.float32)  # in-segment audio not yet decoded
        in_speech, text, cache, ts = False, "", {}, 0.0
        async for f in frames:
            ts = f.ts
            x = pcm_to_float(f.pcm)
            self.vad.accept_waveform(x)
            ended = not self.vad.empty()
            while not self.vad.empty():
                self.vad.pop()
            if not in_speech:
                pre = np.concatenate([pre, x])[-keep:]
                if not (self.vad.is_speech_detected() or ended):
                    continue
                in_speech, buf, text, cache = True, pre, "", {}
            else:
                buf = np.concatenate([buf, x])
            while len(buf) >= self.stride and not ended:
                chunk, buf = buf[: self.stride], buf[self.stride :]
                delta = await self.worker.run(self._chunk, chunk, cache, False)
                if delta:
                    text += delta
                    yield Transcript(self._post(text), False, ts)
            if ended:
                text += await self.worker.run(self._chunk, buf, cache, True)
                if self._post(text):
                    yield Transcript(self._post(text), True, ts)
                in_speech, pre, buf = False, np.zeros(0, np.float32), np.zeros(0, np.float32)
        self.vad.flush()
        if in_speech:
            text += await self.worker.run(self._chunk, buf, cache, True)
            if self._post(text):
                yield Transcript(self._post(text), True, ts)
        self.vad.reset()
