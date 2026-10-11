"""Simulated user with continuous background noise, for VAD robustness tests.

The stock `sim` input outputs digital silence between utterances, which flatters an energy VAD.
`noisy_sim` adds a looped background recording to every frame at a fixed level relative to the
speech (SNR over the scenario's speech RMS). The clean PCM rides along as `frame.clean_pcm`, so
the experiment ASR (`lagged_mock`) can keep acting like a noise-robust recognizer while the turn
detector only sees the noisy signal.

    [plugins] modules = ["experiments.turn.noisy_sim", "experiments.turn.lagged_asr"]
    [input] impl = "noisy_sim"  noise_wav = "experiments/turn/data/bg/babble.wav"  snr_db = 10
"""

from __future__ import annotations

import math
import struct
import wave

from duplexagent.core.registry import register
from duplexagent.core.types import FRAME_SAMPLES
from duplexagent.plugins.io.sim import SimInput


@register("input", "noisy_sim")
class NoisySimInput(SimInput):
    def __init__(self, noise_wav: str = "", snr_db: float = 10.0, speech_rms: float = 0.08, **kw) -> None:
        super().__init__(**kw)
        with wave.open(noise_wav, "rb") as w:
            raw = w.readframes(w.getnframes())
        n = len(raw) // 2
        self.noise = struct.unpack(f"<{n}h", raw[: n * 2])
        cur = math.sqrt(sum(s * s for s in self.noise) / max(1, n)) / 32768 or 1e-6
        self.gain = speech_rms * 10 ** (-snr_db / 20) / cur
        self._pos = 0

    def _frame_pcm(self, i: int, t: float) -> bytes:
        clean = super()._frame_pcm(i, t)
        c = struct.unpack(f"<{FRAME_SAMPLES}h", clean)
        n = len(self.noise)
        out = []
        for k in range(FRAME_SAMPLES):
            v = c[k] + self.gain * self.noise[(self._pos + k) % n]
            out.append(max(-32768, min(32767, int(v))))
        self._pos = (self._pos + FRAME_SAMPLES) % n
        self._last_clean = clean
        return struct.pack(f"<{FRAME_SAMPLES}h", *out)

    async def frames(self):
        async for f in super().frames():
            f.clean_pcm = self._last_clean  # experiment-only side channel for the mock ASR
            yield f
