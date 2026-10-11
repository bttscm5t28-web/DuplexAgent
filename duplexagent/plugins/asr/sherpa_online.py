"""True streaming ASR with sherpa-onnx (k2-fsa): streaming Zipformer transducer / CTC and streaming Paraformer.

    [asr]
    impl = "sherpa_online"
    model = "zipformer-zh-2025"      # a preset below, or set `repo` + `type` + file names yourself
    endpoint_ms = 400                # trailing silence (after decoded speech) that ends an utterance
    num_threads = 2

Models download from Hugging Face into models/asr/ on first use. Needs `pip install sherpa-onnx numpy huggingface_hub`.
Partials come out every decode chunk. Finals come from one of two endpointers:
  endpoint = "model"  sherpa's own rule (trailing blank frames after decoded speech); transducer/CTC only.
  endpoint = "vad"    Silero VAD closes the utterance, then the stream is flushed with tail padding.
                      Required for streaming Paraformer, which has no blank symbol (its "model" endpoint fires mid-word).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

from duplexagent.core.interfaces import ASR
from duplexagent.core.registry import register
from duplexagent.core.types import AudioFrame, Transcript
from duplexagent.plugins.asr._util import Worker, cached, clean, hf_model, itn_zh, pcm_to_float

PRESETS: dict[str, dict] = {
    # 中英双语，icefall stateless7-streaming，2023
    "zipformer-bilingual": dict(
        repo="csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20", type="transducer",
        encoder="encoder-epoch-99-avg-1.int8.onnx", decoder="decoder-epoch-99-avg-1.onnx", joiner="joiner-epoch-99-avg-1.int8.onnx",
    ),
    # 中文（数万小时），zipformer2 streaming，2025
    "zipformer-zh-2025": dict(
        repo="csukuangfj/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30", type="transducer",
        encoder="encoder.int8.onnx", decoder="decoder.onnx", joiner="joiner.int8.onnx",
    ),
    "zipformer-zh-xlarge-2025": dict(
        repo="csukuangfj/sherpa-onnx-streaming-zipformer-zh-xlarge-int8-2025-06-30", type="transducer",
        encoder="encoder.int8.onnx", decoder="decoder.onnx", joiner="joiner.int8.onnx",
    ),
    "zipformer-ctc-zh-2025": dict(repo="csukuangfj/sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30", type="zipformer2_ctc", model="model.int8.onnx"),
    # FunASR paraformer-zh-streaming（中英），导出为 onnx
    "paraformer-bilingual": dict(
        repo="csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en", type="paraformer", encoder="encoder.int8.onnx", decoder="decoder.int8.onnx",
        endpoint="vad",
    ),
}


@register("asr", "sherpa_online")
class SherpaOnlineASR(ASR):
    def __init__(
        self,
        model: str = "zipformer-zh-2025",
        endpoint_ms: int = 400,
        endpoint: str | None = None,  # "model" | "vad"; default per preset ("model" unless the preset says otherwise)
        num_threads: int = 2,
        batch_ms: int = 60,  # feed the recognizer this much audio per worker call
        itn: bool = False,  # 二十三 -> 23 (set true for the engine; false for CER against Chinese-numeral references)
        decoding_method: str = "greedy_search",
        provider: str = "cpu",
        **files: str,
    ) -> None:
        if model not in PRESETS and "repo" not in files:
            raise ValueError(f"unknown sherpa_online model {model!r}; presets: {sorted(PRESETS)}")
        self.spec = {**PRESETS.get(model, {}), **files}
        self.endpoint = endpoint or self.spec.pop("endpoint", "model")
        self.spec.pop("endpoint", None)
        if self.endpoint not in ("model", "vad"):
            raise ValueError("endpoint must be 'model' or 'vad'")
        self.model = model
        self.endpoint_ms = endpoint_ms
        self.num_threads = num_threads
        self.batch_ms = batch_ms
        self.itn = itn
        self.decoding_method = decoding_method
        self.provider = provider

    def _load(self):
        import sherpa_onnx

        s = self.spec
        names = [v for k, v in s.items() if k in ("encoder", "decoder", "joiner", "model")] + ["tokens.txt"]
        d = hf_model(s["repo"], allow=names + ["bpe.model"])
        p = lambda k: os.path.join(d, s[k])  # noqa: E731
        common = dict(
            tokens=os.path.join(d, "tokens.txt"), num_threads=self.num_threads, sample_rate=16000, feature_dim=80,
            enable_endpoint_detection=self.endpoint == "model", rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=self.endpoint_ms / 1000, rule3_min_utterance_length=30.0,
            decoding_method=self.decoding_method, provider=self.provider,
        )
        if s["type"] == "transducer":
            return sherpa_onnx.OnlineRecognizer.from_transducer(encoder=p("encoder"), decoder=p("decoder"), joiner=p("joiner"), **common)
        if s["type"] == "paraformer":
            common.pop("decoding_method")
            return sherpa_onnx.OnlineRecognizer.from_paraformer(encoder=p("encoder"), decoder=p("decoder"), **common)
        if s["type"] == "zipformer2_ctc":
            common["decoding_method"] = "greedy_search"
            return sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(model=p("model"), **common)
        raise ValueError(f"unknown sherpa_online model type {s['type']!r}")

    async def start(self) -> None:
        self.worker = Worker()
        key = ("sherpa_online", tuple(sorted(self.spec.items())), self.endpoint, self.endpoint_ms, self.num_threads, self.decoding_method, self.provider)
        self.rec = await self.worker.run(cached, key, self._load)
        self.stream = self.rec.create_stream()
        self.vad = None
        if self.endpoint == "vad":
            from duplexagent.plugins.asr.vad_offline import make_silero_vad

            self.vad = await self.worker.run(make_silero_vad, self.endpoint_ms)

    async def close(self) -> None:
        self.worker.close()

    def _step(self, samples, finish: bool = False) -> tuple[str, bool]:
        """Worker thread: feed audio, decode what is ready, report (text, endpoint)."""
        rec, s = self.rec, self.stream
        if len(samples):
            s.accept_waveform(16000, samples)
            if self.vad is not None:
                self.vad.accept_waveform(samples)
                if not self.vad.empty():
                    finish = True
                    while not self.vad.empty():
                        self.vad.pop()
        if finish:
            import numpy as np

            s.accept_waveform(16000, np.zeros(int(0.66 * 16000), dtype=np.float32))  # tail padding flushes the last chunk
            s.input_finished()
        while rec.is_ready(s):
            rec.decode_stream(s)
        text = rec.get_result(s)
        if finish:
            self.stream = rec.create_stream()
            return text, True
        ep = rec.is_endpoint(s)
        if ep:
            rec.reset(s)
        return text, ep

    def _post(self, text: str) -> str:
        text = clean(text).lower()
        return itn_zh(text) if self.itn else text

    async def transcribe(self, frames: AsyncIterator[AudioFrame]) -> AsyncIterator[Transcript]:
        import numpy as np

        if self.vad is not None:
            self.vad.reset()
        buf: list[bytes] = []
        buf_ms, last, ts = 0, "", 0.0
        async for f in frames:
            buf.append(f.pcm)
            buf_ms += f.duration * 1000
            ts = f.ts
            if buf_ms < self.batch_ms:
                continue
            samples = pcm_to_float(b"".join(buf))
            buf, buf_ms = [], 0
            text, ep = await self.worker.run(self._step, samples)
            text = self._post(text)
            if text and text != last and not ep:
                yield Transcript(text, False, ts)
                last = text
            if ep:
                if text:
                    yield Transcript(text, True, ts)
                last = ""
        text, _ = await self.worker.run(self._step, pcm_to_float(b"".join(buf)) if buf else np.zeros(0, np.float32), True)
        text = self._post(text)
        if text:
            yield Transcript(text, True, ts)
