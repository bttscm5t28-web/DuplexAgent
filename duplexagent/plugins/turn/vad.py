"""Frame-level voice activity backends for the `hybrid` turn detectors.

Each backend maps one 20 ms frame (16 kHz int16 PCM) to a speech score in [0, 1]:

  energy  RMS / threshold, stdlib only (the baseline's VAD)
  silero  Silero VAD v5/v6 ONNX (MIT) via onnxruntime; weights ship inside the `silero-vad` wheel
  ten     TEN VAD (Apache-2.0 + extra conditions) via the `ten-vad` wheel (needs libc++1 on Linux)

Optional dependencies are imported lazily so the stdlib baseline never needs them.
"""

from __future__ import annotations

import os
import threading

from duplexagent.core.audio import rms

_lock = threading.Lock()
_sessions: dict[str, object] = {}


def _ort_session(path: str, threads: int = 1):
    import onnxruntime as ort

    key = f"{path}:{threads}"
    with _lock:
        if key not in _sessions:
            so = ort.SessionOptions()
            so.inter_op_num_threads = 1
            so.intra_op_num_threads = threads
            so.log_severity_level = 3
            _sessions[key] = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        return _sessions[key]


class EnergyVAD:
    name = "energy"

    def __init__(self, threshold: float = 0.02) -> None:
        self.threshold = threshold

    def __call__(self, pcm: bytes) -> float:
        # 0.5 exactly at the threshold, saturating at 2x: keeps the [0, 1] "probability" contract
        return min(1.0, 0.5 * rms(pcm) / self.threshold)


def silero_model_path() -> str:
    env = os.environ.get("SILERO_VAD_ONNX")
    if env:
        return env
    import importlib.util

    spec = importlib.util.find_spec("silero_vad")  # locate the wheel without importing it (it imports torch)
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Silero VAD weights not found: `pip install --no-deps silero-vad` or set SILERO_VAD_ONNX")
    return os.path.join(list(spec.submodule_search_locations)[0], "data", "silero_vad.onnx")


class SileroVAD:
    """Silero needs 512-sample windows (+64 samples of context); we run it every 32 ms and hold the last score."""

    name = "silero"
    WIN, CTX = 512, 64

    def __init__(self, model_path: str | None = None) -> None:
        import numpy as np

        self.np = np
        self.sess = _ort_session(model_path or silero_model_path())
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros(self.CTX, dtype=np.float32)
        self.buf = np.zeros(0, dtype=np.float32)
        self.sr = np.array(16000, dtype=np.int64)
        self.last = 0.0

    def __call__(self, pcm: bytes) -> float:
        np = self.np
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        self.buf = np.concatenate([self.buf, x])
        while len(self.buf) >= self.WIN:
            chunk, self.buf = self.buf[: self.WIN], self.buf[self.WIN :]
            inp = np.concatenate([self.context, chunk])[None, :]
            out, self.state = self.sess.run(None, {"input": inp, "state": self.state, "sr": self.sr})
            self.context = chunk[-self.CTX :]
            self.last = float(out[0][0])
        return self.last


class TenVAD:
    """TEN VAD on 10 ms hops (two per frame); returns the max probability of the frame's hops."""

    name = "ten"

    def __init__(self, hop: int = 160) -> None:
        import numpy as np
        from ten_vad import TenVad

        self.np = np
        self.hop = hop
        self.vad = TenVad(hop_size=hop, threshold=0.5)
        self.buf = np.zeros(0, dtype=np.int16)
        self.last = 0.0

    def __call__(self, pcm: bytes) -> float:
        np = self.np
        self.buf = np.concatenate([self.buf, np.frombuffer(pcm, dtype=np.int16)])
        probs = []
        while len(self.buf) >= self.hop:
            chunk, self.buf = np.ascontiguousarray(self.buf[: self.hop]), self.buf[self.hop :]
            p, _ = self.vad.process(chunk)
            probs.append(p)
        if probs:
            self.last = max(probs)
        return self.last


def make_vad(kind: str, **kw):
    if kind == "energy":
        return EnergyVAD(kw.get("energy_threshold", 0.02))
    if kind == "silero":
        return SileroVAD(kw.get("model_path"))
    if kind == "ten":
        return TenVAD()
    raise ValueError(f"unknown vad {kind!r}; expected energy | silero | ten")
