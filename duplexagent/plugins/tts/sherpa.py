"""sherpa-onnx offline TTS models (CPU, ONNX Runtime), wrapped for streaming.

    [tts]
    impl = "sherpa"
    preset = "matcha-zh-en"          # see PRESETS
    model_root = "models/tts"        # where the model archives were extracted
    sid = 0                          # speaker id for multi-speaker models
    speed = 1.0
    num_threads = 2

Download (weights are not in git):
    python experiments/tts/download.py matcha-zh-en      # or: all
"""

from __future__ import annotations

import os
import threading
from typing import Any

from duplexagent.core.registry import register
from duplexagent.plugins.tts.streaming import StreamingTTS

RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

# preset -> (archive dirs / files to download, english handling, builder)
PRESETS: dict[str, dict[str, Any]] = {
    "matcha-zh-en": {
        "files": ["tts-models/matcha-icefall-zh-en.tar.bz2", "vocoder-models/vocos-16khz-univ.onnx"],
        "english": "keep",
        "license": "Apache-2.0 (icefall/sherpa-onnx); data: baker + LJSpeech",
    },
    "matcha-zh-baker": {
        "files": ["tts-models/matcha-icefall-zh-baker.tar.bz2", "vocoder-models/vocos-22khz-univ.onnx"],
        "english": "zh",
        "license": "Apache-2.0 code; Biaobei (baker) data is non-commercial",
    },
    "melo-zh-en": {
        "files": ["tts-models/vits-melo-tts-zh_en.tar.bz2"],
        "english": "keep",
        "license": "MIT (MeloTTS)",
    },
    "vits-aishell3": {
        "files": ["tts-models/vits-icefall-zh-aishell3.tar.bz2"],
        "english": "zh",
        "license": "Apache-2.0 code; AISHELL-3 data (Apache-2.0)",
    },
    "piper-huayan": {
        "files": ["tts-models/vits-piper-zh_CN-huayan-medium.tar.bz2"],
        "english": "zh",
        "license": "MIT (piper); huayan voice dataset license unclear",
    },
    "kokoro": {
        "files": ["tts-models/kokoro-multi-lang-v1_1.tar.bz2"],
        "english": "keep",
        "license": "Apache-2.0 (Kokoro-82M v1.1-zh)",
    },
    "kokoro-int8": {
        "files": ["tts-models/kokoro-int8-multi-lang-v1_1.tar.bz2"],
        "english": "keep",
        "license": "Apache-2.0 (Kokoro-82M v1.1-zh)",
    },
    "zipvoice": {
        "files": ["tts-models/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2", "vocoder-models/vocos_24khz.onnx"],
        "english": "keep",
        "license": "Apache-2.0 code; Emilia-trained weights (CC-BY-NC-4.0 per upstream note)",
    },
}


def _build(preset: str, root: str, num_threads: int, extra: dict) -> Any:
    import sherpa_onnx as so

    j = lambda *p: os.path.join(root, *p)  # noqa: E731
    m = so.OfflineTtsModelConfig(num_threads=num_threads, provider="cpu")
    fsts = ""
    if preset == "matcha-zh-en":
        d = j("matcha-icefall-zh-en")
        m.matcha = so.OfflineTtsMatchaModelConfig(
            acoustic_model=f"{d}/model-steps-3.onnx", vocoder=j("vocos-16khz-univ.onnx"),
            lexicon=f"{d}/lexicon.txt", tokens=f"{d}/tokens.txt", data_dir=f"{d}/espeak-ng-data",
            noise_scale=extra.get("noise_scale", 0.667), length_scale=extra.get("length_scale", 1.0),
        )
    elif preset == "matcha-zh-baker":
        d = j("matcha-icefall-zh-baker")
        m.matcha = so.OfflineTtsMatchaModelConfig(
            acoustic_model=f"{d}/model-steps-3.onnx", vocoder=j("vocos-22khz-univ.onnx"),
            lexicon=f"{d}/lexicon.txt", tokens=f"{d}/tokens.txt", dict_dir=f"{d}/dict",
            noise_scale=extra.get("noise_scale", 0.667), length_scale=extra.get("length_scale", 1.0),
        )
    elif preset == "melo-zh-en":
        d = j("vits-melo-tts-zh_en")
        m.vits = so.OfflineTtsVitsModelConfig(model=f"{d}/model.onnx", lexicon=f"{d}/lexicon.txt", tokens=f"{d}/tokens.txt", dict_dir=f"{d}/dict")
    elif preset == "vits-aishell3":
        d = j("vits-icefall-zh-aishell3")
        m.vits = so.OfflineTtsVitsModelConfig(model=f"{d}/model.onnx", lexicon=f"{d}/lexicon.txt", tokens=f"{d}/tokens.txt")
        fsts = f"{d}/new_heteronym.fst"
    elif preset == "piper-huayan":
        d = j("vits-piper-zh_CN-huayan-medium")
        m.vits = so.OfflineTtsVitsModelConfig(model=f"{d}/zh_CN-huayan-medium.onnx", tokens=f"{d}/tokens.txt", data_dir=f"{d}/espeak-ng-data")
    elif preset in ("kokoro", "kokoro-int8"):
        d = j("kokoro-multi-lang-v1_1" if preset == "kokoro" else "kokoro-int8-multi-lang-v1_1")
        m.kokoro = so.OfflineTtsKokoroModelConfig(
            model=f"{d}/model{'.int8' if preset == 'kokoro-int8' else ''}.onnx", voices=f"{d}/voices.bin", tokens=f"{d}/tokens.txt",
            lexicon=f"{d}/lexicon-us-en.txt,{d}/lexicon-zh.txt", data_dir=f"{d}/espeak-ng-data", dict_dir=f"{d}/dict",
        )
    elif preset == "zipvoice":
        d = j("sherpa-onnx-zipvoice-distill-int8-zh-en-emilia")
        m.zipvoice = so.OfflineTtsZipvoiceModelConfig(
            tokens=f"{d}/tokens.txt", encoder=f"{d}/encoder.int8.onnx", decoder=f"{d}/decoder.int8.onnx",
            vocoder=j("vocos_24khz.onnx"), data_dir=f"{d}/espeak-ng-data", lexicon=f"{d}/lexicon.txt",
        )
    else:
        raise ValueError(f"unknown sherpa preset {preset!r}; one of {sorted(PRESETS)}")
    # Our own normalizer already verbalized numbers/dates, so the zh rule FSTs are not needed
    # (they would also break the alignment between our units and the audio).
    cfg = so.OfflineTtsConfig(model=m, rule_fsts=fsts, max_num_sentences=1, silence_scale=extra.get("silence_scale", 0.2))
    if not cfg.validate():
        raise RuntimeError(f"sherpa-onnx config for {preset!r} is invalid; did you download the model into {root!r}?")
    return so.OfflineTts(cfg)


@register("tts", "sherpa")
class SherpaTTS(StreamingTTS):
    def __init__(
        self,
        preset: str = "matcha-zh-en",
        model_root: str = "models/tts",
        sid: int = 0,
        speed: float = 1.0,
        num_threads: int = 2,
        noise_scale: float | None = None,
        length_scale: float | None = None,
        prompt_wav: str | None = None,  # zipvoice: reference voice
        prompt_text: str | None = None,
        num_steps: int = 4,  # zipvoice flow-matching steps
        english: str | None = None,
        **kw: Any,
    ) -> None:
        if preset not in PRESETS:
            raise ValueError(f"unknown sherpa preset {preset!r}; one of {sorted(PRESETS)}")
        super().__init__(english=english or PRESETS[preset]["english"], **kw)
        self.preset, self.model_root, self.sid, self.speed = preset, model_root, sid, speed
        self.num_threads = num_threads
        self.extra = {k: v for k, v in (("noise_scale", noise_scale), ("length_scale", length_scale)) if v is not None}
        self.prompt_wav, self.prompt_text, self.num_steps = prompt_wav, prompt_text, num_steps
        if preset == "zipvoice" and not self.prompt_wav:
            d = os.path.join(model_root, "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia", "test_wavs")
            self.prompt_wav = os.path.join(d, "news-female-2.wav")
            self.prompt_text = "本台消息, 中共中央国务院, 近日印发关于构建数据基础制度, 更好发挥数据要素作用的意见."

    def _cache_key(self) -> tuple:
        return (self.preset, os.path.abspath(self.model_root), self.num_threads, tuple(sorted(self.extra.items())))

    def _load(self) -> Any:
        tts = _build(self.preset, self.model_root, self.num_threads, self.extra)
        prompt = None
        if self.preset == "zipvoice":
            import soundfile as sf

            samples, sr = sf.read(self.prompt_wav, dtype="float32", always_2d=False)
            prompt = (samples, sr)
        return {"tts": tts, "prompt": prompt}

    def _synth(self, model: Any, text: str, stop: threading.Event):
        import numpy as np
        import sherpa_onnx as so

        tts = model["tts"]

        def cb(_samples, _progress) -> int:
            return 0 if stop.is_set() else 1  # return 0 to stop generation

        if model["prompt"] is not None:
            g = so.GenerationConfig()
            g.reference_audio, g.reference_sample_rate = model["prompt"]
            g.reference_text = self.prompt_text
            g.num_steps, g.speed = self.num_steps, self.speed
            a = tts.generate(text, g, callback=cb)
        else:
            a = tts.generate(text, sid=self.sid, speed=self.speed, callback=cb)
        return np.asarray(a.samples, dtype=np.float32), a.sample_rate
