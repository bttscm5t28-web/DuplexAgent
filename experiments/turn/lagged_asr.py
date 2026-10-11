"""Mock ASR with realistic streaming lag, for turn-detection experiments.

`asr=mock` hands out each character in the very frame it is "spoken" and the final
160 ms after the voice stops, which flatters text-based turn models. Real streaming
ASR (Paraformer / SenseVoice / Whisper-streaming) shows partials ~200-400 ms late and
finals a few hundred ms after the end. This variant delays the mock's transcripts.

It also fixes a mock artifact that shows up with `wav = ...` scenarios: the first truth
character lands on the recording's leading-silence frame, and since the silence counter
carries over from before the utterance the mock emits an instant one-character final.
Here an utterance is only endpointed after voice has actually been heard in it.
Use partial_lag_ms = final_lag_ms = 0 for an "ideal" ASR on wav scenarios.

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
        held = ""  # truth characters that fell on silent frames: an ASR can't hear them yet
        silent_ms = 0
        utt = 0
        heard = False  # voice seen since this utterance's first character
        last_partial = -1e9
        pending: list[tuple[float, int, str, bool]] = []  # (due, utterance, text, is_final)
        async for f in frames:
            voiced = rms(f.pcm) >= self.vad_threshold
            if f.truth:
                held += f.truth
            if voiced and held:
                text += held
                held = ""
                if f.ts - last_partial >= self.partial_every:
                    last_partial = f.ts
                    pending.append((f.ts + self.partial_lag, utt, text, False))
            if voiced:
                silent_ms = 0
                heard = heard or bool(text)
            else:
                silent_ms += FRAME_MS
                if heard and silent_ms >= self.endpoint_ms:
                    text += held  # characters scheduled on the recording's trailing pad
                    held = ""
                    pending.append((f.ts + self.final_lag, utt, text, True))
                    text, utt, last_partial, heard = "", utt + 1, -1e9, False
            pending.sort(key=lambda p: p[0])
            while pending and pending[0][0] <= f.ts:
                _, u, txt, final = pending.pop(0)
                yield Transcript(txt, final, f.ts)
                if final:  # partials of that utterance still in flight are stale now
                    pending = [p for p in pending if p[1] != u]
        for _, _, txt, final in pending:
            yield Transcript(txt, final, 0.0)
