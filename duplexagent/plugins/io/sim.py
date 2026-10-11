"""Simulated user and speaker: run the whole engine without microphones or speech models.

The simulated user "speaks" scripted utterances, either from a recording (`wav = ...`, so
real ASR / VAD models can be tested) or as noise-like PCM (so energy VADs work). Ground-truth
characters ride on the frames so the mock ASR can read them.
"""

from __future__ import annotations

import asyncio
import tomllib
from collections.abc import AsyncIterator
from dataclasses import dataclass

from duplexagent.core.audio import noise, silence
from duplexagent.core.interfaces import AudioInput, AudioOutput
from duplexagent.core.registry import register
from duplexagent.core.types import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, AudioFrame, TTSChunk


def _read_wav(path: str) -> bytes:
    import wave

    with wave.open(path, "rb") as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: need {SAMPLE_RATE} Hz mono int16")
        return w.readframes(w.getnframes())


@dataclass
class Utterance:
    at: float
    text: str
    duration: float
    kind: str = "turn"  # free-form label for evaluation ("turn" | "backchannel" | "barge_in" | ...)
    pcm: bytes | None = None  # real recording of this utterance (16 kHz mono int16), if given

    @property
    def end(self) -> float:
        return self.at + self.duration


@dataclass
class Scenario:
    name: str
    utterances: list[Utterance]
    duration: float
    expect: dict

    @classmethod
    def load(cls, path: str, user_cps: float = 5.0) -> Scenario:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        return cls.from_dict(raw, user_cps)

    @classmethod
    def from_dict(cls, raw: dict, user_cps: float = 5.0) -> Scenario:
        utts = []
        for u in raw.get("turns", []):
            text = u["text"]
            pcm = _read_wav(u["wav"]) if u.get("wav") else None
            if pcm is not None:
                dur = len(pcm) / 2 / SAMPLE_RATE
            else:
                dur = u.get("duration") or max(0.2, len(text) / u.get("cps", user_cps))
            utts.append(Utterance(at=float(u["at"]), text=text, duration=dur, kind=u.get("kind", "turn"), pcm=pcm))
        utts.sort(key=lambda u: u.at)
        last = max((u.end for u in utts), default=0.0)
        return cls(raw.get("name", "scenario"), utts, float(raw.get("duration", last + 5.0)), raw.get("expect", {}))


@register("input", "sim")
class SimInput(AudioInput):
    def __init__(self, scenario: str | dict | None = None, user_cps: float = 5.0, amplitude: float = 0.1) -> None:
        if scenario is None:
            raise ValueError("[input] sim needs `scenario` (path or inline table)")
        self.scenario = (
            Scenario.from_dict(scenario, user_cps) if isinstance(scenario, dict) else Scenario.load(scenario, user_cps)
        )
        self.amplitude = amplitude

    def _truth_schedule(self) -> dict[int, str]:
        """frame index -> characters spoken in that frame."""
        sched: dict[int, str] = {}
        step = FRAME_MS / 1000
        for u in self.scenario.utterances:
            first, n = round(u.at / step), max(1, round(u.duration / step))
            for k, ch in enumerate(u.text):
                idx = first + min(n - 1, int(k * n / len(u.text)))
                sched[idx] = sched.get(idx, "") + ch
        return sched

    def _frame_pcm(self, i: int, t: float) -> bytes:
        for u in self.scenario.utterances:
            if u.at <= t < u.end:
                if u.pcm is None:
                    return noise(amplitude=self.amplitude, seed=i)
                off = int((t - u.at) * SAMPLE_RATE) * 2
                chunk = u.pcm[off : off + FRAME_SAMPLES * 2]
                return chunk + silence(FRAME_SAMPLES - len(chunk) // 2)
        return silence()

    async def frames(self) -> AsyncIterator[AudioFrame]:
        loop = asyncio.get_running_loop()
        step = FRAME_MS / 1000
        sched = self._truth_schedule()
        start = loop.time()
        total = int(self.scenario.duration / step)
        for i in range(total):
            t = i * step
            delay = start + t - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            yield AudioFrame(pcm=self._frame_pcm(i, t), ts=self.clock(), truth=sched.get(i))


@register("output", "sim")
class SimOutput(AudioOutput):
    """Pretends to play audio in real time; keeps a transcript of what was played."""

    def __init__(self, print_text: bool = False) -> None:
        self.print_text = print_text
        self.played: list[tuple[float, str]] = []

    async def play(self, chunk: TTSChunk) -> None:
        self.played.append((self.clock(), chunk.text))
        if self.print_text and chunk.text:
            print(chunk.text, end="", flush=True)
        await asyncio.sleep(chunk.duration)
