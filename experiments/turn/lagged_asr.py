"""Mock ASR with realistic streaming lag, for turn-detection experiments.

`asr=mock` hands out each character in the very frame it is "spoken" and the final
160 ms after the voice stops, which flatters text-based turn models. Real streaming
ASR (Paraformer / SenseVoice / Whisper-streaming) shows partials ~200-400 ms late and
finals a few hundred ms after the end. This variant delays the mock's transcripts.

    [plugins] modules = ["experiments.turn.lagged_asr"]
    [asr] impl = "lagged_mock"  partial_lag_ms = 300  final_lag_ms = 200
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from duplexagent.core.audio import rms
from duplexagent.core.interfaces import ASR
from duplexagent.core.registry import register
from duplexagent.core.types import FRAME_MS, AudioFrame, Transcript


@register("asr", "lagged_mock")
class LaggedMockASR(ASR):
    def __init__(
        self,
        partial_lag_ms: int = 300,
        final_lag_ms: int = 200,  # on top of endpoint_ms
        partial_every_ms: int = 100,  # real ASRs emit partials in chunks, not per character
        endpoint_ms: int = 160,
        vad_threshold: float = 0.02,
    ) -> None:
        self.partial_lag = partial_lag_ms / 1000
        self.final_lag = final_lag_ms / 1000
        self.partial_every = partial_every_ms / 1000
        self.endpoint_ms = endpoint_ms
        self.vad_threshold = vad_threshold

    async def transcribe(self, frames: AsyncIterator[AudioFrame]) -> AsyncIterator[Transcript]:
        text = ""
        silent_ms = 0
        utt = 0
        last_partial = -1e9
        pending: list[tuple[float, int, str, bool]] = []  # (due, utterance, text, is_final)
        async for f in frames:
            if f.truth:
                text += f.truth
                if f.ts - last_partial >= self.partial_every:
                    last_partial = f.ts
                    pending.append((f.ts + self.partial_lag, utt, text, False))
            if rms(f.pcm) >= self.vad_threshold:
                silent_ms = 0
            else:
                silent_ms += FRAME_MS
                if text and silent_ms >= self.endpoint_ms:
                    pending.append((f.ts + self.final_lag, utt, text, True))
                    text, utt, last_partial = "", utt + 1, -1e9
            pending.sort(key=lambda p: p[0])
            while pending and pending[0][0] <= f.ts:
                _, u, txt, final = pending.pop(0)
                yield Transcript(txt, final, f.ts)
                if final:  # partials of that utterance still in flight are stale now
                    pending = [p for p in pending if p[1] != u]
        for _, _, txt, final in pending:
            yield Transcript(txt, final, 0.0)
