"""Baseline turn detector: energy VAD + silence timeout + backchannel word list."""

from __future__ import annotations

import re

from duplexagent.core.audio import rms
from duplexagent.core.interfaces import TurnContext, TurnDetector
from duplexagent.core.registry import register
from duplexagent.core.types import FRAME_MS, AudioFrame, Transcript, TurnEvent, TurnEventType as T

DEFAULT_BACKCHANNELS = ["嗯", "恩", "啊", "哦", "噢", "对", "是", "好", "好的", "行", "嗯哼", "没错", "然后呢", "uh-huh", "yeah", "ok", "okay", "mm", "right"]
_PUNCT = re.compile(r"[\s,.!?;:，。！？；：、…~\-]+")


@register("turn", "rule")
class RuleTurnDetector(TurnDetector):
    def __init__(
        self,
        vad_threshold: float = 0.02,
        min_speech_ms: int = 60,  # voiced time before SPEECH_START
        hangover_ms: int = 100,  # silence before SPEECH_END
        eot_silence_ms: int = 500,  # silence after speech before TURN_END
        final_wait_ms: int = 400,  # extra wait for the ASR final
        barge_in_ms: int = 300,  # voiced time while agent talks before BARGE_IN
        barge_in_max_ms: int = 700,  # barge in even without any transcript after this long
        backchannels: list[str] | None = None,
    ) -> None:
        self.vad_threshold = vad_threshold
        self.min_speech_ms = min_speech_ms
        self.hangover_ms = hangover_ms
        self.eot_silence_ms = eot_silence_ms
        self.final_wait_ms = final_wait_ms
        self.barge_in_ms = barge_in_ms
        self.barge_in_max_ms = barge_in_max_ms
        self.backchannels = sorted(backchannels or DEFAULT_BACKCHANNELS, key=len, reverse=True)
        self._reset_utterance()
        self.in_speech = False
        self._run_ms = 0
        self._silence_ms = 0

    def _reset_utterance(self) -> None:
        self.utt_active = False
        self.utt_text = ""
        self._committed = ""  # ASR finals within this turn (one turn may span several ASR segments)
        self.utt_final = False
        self.voiced_during_agent_ms = 0
        self.started_during_agent = False
        self.barged = False

    def is_backchannel(self, text: str) -> bool:
        rest = _PUNCT.sub("", text.lower())
        if not rest:
            return False
        for b in self.backchannels:
            rest = rest.replace(b, "")
        return rest == ""

    def _maybe_barge(self, ts: float, ctx: TurnContext) -> list[TurnEvent]:
        if self.barged or not ctx.agent_speaking or not self.in_speech:
            return []
        if self.voiced_during_agent_ms < self.barge_in_ms:
            return []
        if self.utt_text and self.is_backchannel(self.utt_text):
            return []
        if not self.utt_text and self.voiced_during_agent_ms < self.barge_in_max_ms:
            return []
        self.barged = True
        return [TurnEvent(T.BARGE_IN, ts, self.utt_text)]

    def on_audio(self, frame: AudioFrame, ctx: TurnContext) -> list[TurnEvent]:
        out: list[TurnEvent] = []
        if rms(frame.pcm) >= self.vad_threshold:
            self._run_ms += FRAME_MS
            self._silence_ms = 0
            if not self.in_speech and self._run_ms >= self.min_speech_ms:
                self.in_speech = True
                if not self.utt_active:
                    self.utt_active = True
                    self.started_during_agent = ctx.agent_speaking
                out.append(TurnEvent(T.SPEECH_START, frame.ts))
            if self.in_speech and ctx.agent_speaking:
                self.voiced_during_agent_ms += FRAME_MS
                out += self._maybe_barge(frame.ts, ctx)
            return out

        self._run_ms = 0
        self._silence_ms += FRAME_MS
        if self.in_speech and self._silence_ms >= self.hangover_ms:
            self.in_speech = False
            out.append(TurnEvent(T.SPEECH_END, frame.ts - self._silence_ms / 1000))
        if self.utt_active and not self.in_speech and self._silence_ms >= self.eot_silence_ms:
            if self.utt_final or self._silence_ms >= self.eot_silence_ms + self.final_wait_ms:
                out += self._finish(frame.ts, ctx)
        return out

    def on_transcript(self, t: Transcript, ctx: TurnContext) -> list[TurnEvent]:
        if t.is_final:
            self._committed += t.text
            self.utt_text = self._committed
        else:
            self.utt_text = self._committed + t.text
        self.utt_final = t.is_final
        return self._maybe_barge(t.ts, ctx)

    def _finish(self, ts: float, ctx: TurnContext) -> list[TurnEvent]:
        text = self.utt_text.strip()
        out: list[TurnEvent] = []
        if text:
            if ctx.agent_speaking and not self.barged and self.is_backchannel(text):
                out.append(TurnEvent(T.BACKCHANNEL, ts, text))
            else:
                if ctx.agent_speaking and not self.barged:
                    out.append(TurnEvent(T.BARGE_IN, ts, text))  # short but meaningful, e.g. "停"
                out.append(TurnEvent(T.TURN_END, ts, text))
        self._reset_utterance()
        return out
