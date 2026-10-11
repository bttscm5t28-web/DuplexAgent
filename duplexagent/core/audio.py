from __future__ import annotations

import math
import random
import struct

from duplexagent.core.types import FRAME_SAMPLES, SAMPLE_RATE


def rms(pcm: bytes) -> float:
    n = len(pcm) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", pcm[: n * 2])
    return math.sqrt(sum(s * s for s in samples) / n) / 32768.0


def silence(samples: int = FRAME_SAMPLES) -> bytes:
    return b"\x00\x00" * samples


def noise(samples: int = FRAME_SAMPLES, amplitude: float = 0.1, seed: int | None = None) -> bytes:
    """Speech-like stand-in: band-limited noise at the given amplitude."""
    rnd = random.Random(seed)
    amp = int(amplitude * 32767)
    return struct.pack(f"<{samples}h", *(rnd.randint(-amp, amp) for _ in range(samples)))


def tone(seconds: float, freq: float = 220.0, amplitude: float = 0.2, sr: int = SAMPLE_RATE) -> bytes:
    n = int(seconds * sr)
    amp = amplitude * 32767
    return struct.pack(f"<{n}h", *(int(amp * math.sin(2 * math.pi * freq * i / sr)) for i in range(n)))
