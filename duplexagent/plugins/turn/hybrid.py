"""Hybrid turn detector: pluggable VAD + fused end-of-turn scorers + lexicon-aware barge-in.

    VAD (energy | silero | ten) ──► speech / silence with hysteresis
    EOT scorers (text rules, Smart Turn audio, LM text, TEN text) ──► P(done) for the current pause
    P(done) ──► adaptive silence timeout: min_silence_ms when sure the user is done,
                max_silence_ms when sure they are mid-sentence ("我想……")
    Barge-in while the agent talks:
      stop command ("停/等一下/不对") ──► immediately
      backchannel or its prefix ("嗯/对对对/原来如此") ──► never (BACKCHANNEL at the end)
      echo of the agent's own words ──► ignored
      other words ──► after barge_in_ms of voice; no words at all ──► after barge_in_max_ms

Slow scorers run in a worker thread so `on_audio` never blocks the 20 ms frame loop; until a
result arrives the detector waits (bounded by max_silence_ms).

Registered presets (all are this class with different defaults; any kwarg can be overridden):
  hybrid / adaptive   energy VAD + text rules
  silero              Silero VAD + fixed silence (VAD swap only)
  ten_vad             TEN VAD + fixed silence (VAD swap only)
  smart_turn          Silero VAD + Smart Turn v3
  lm_eot              Silero VAD + LiveKit-style LM text EOT
  ten_turn            Silero VAD + TEN Turn Detection
  fusion              Silero VAD + Smart Turn + text rules (+ LM if enabled)
"""

from __future__ import annotations

import concurrent.futures as cf
from typing import Any

from duplexagent.core.interfaces import TurnContext, TurnDetector
from duplexagent.core.registry import register
from duplexagent.core.types import FRAME_MS, SAMPLE_RATE, AudioFrame, Transcript, TurnEvent, TurnEventType as T
from duplexagent.plugins.turn import lexicon

_pool: cf.ThreadPoolExecutor | None = None


def _executor() -> cf.ThreadPoolExecutor:
    global _pool
    if _pool is None:
        _pool = cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="turn-eot")
    return _pool


class _Done:
    """Future-like wrapper for scorers cheap enough to run inline."""

    def __init__(self, v: float) -> None:
        self.v = v

    def done(self) -> bool:
        return True

    def result(self) -> float:
        return self.v


@register("turn", "hybrid")
class HybridTurnDetector(TurnDetector):
    DEFAULTS: dict[str, Any] = {}

    def __init__(self, **kw: Any) -> None:
        o = {
            "vad": "energy",
            "vad_threshold": 0.5,  # speech score to start voice (energy: 0.5 == rms at energy_threshold)
            "vad_off_threshold": None,  # score below which voice stops (hysteresis); default = vad_threshold - 0.15
            "energy_threshold": 0.02,
            "min_speech_ms": 60,
            "hangover_ms": 100,
            "eot": ["text"],  # scorers fused into P(done); [] = plain silence timeout
            "eot_weights": {},  # name -> weight in the fused mean (default 1)
            "eot_config": {},  # name -> kwargs for that scorer
            "base_silence_ms": 500,  # timeout when no scorer has an opinion
            "min_silence_ms": 250,  # timeout when P(done) >= p_high
            "max_silence_ms": 2000,  # timeout when P(done) <= p_low
            "p_low": 0.15,
            "p_high": 0.75,
            "audio_probe_ms": 100,  # run audio scorers once the pause is this long (they need the pause)
            "final_wait_ms": 400,  # extra wait for the ASR final after the timeout
            "barge_in_ms": 300,  # voice while the agent talks (with non-backchannel words) before BARGE_IN
            "barge_in_max_ms": 700,  # barge in without any words after this much voice
            "command_barge": True,  # stop commands barge in at once
            "echo_guard": True,  # ignore "user" text that is a slice of the agent's own speech
            "extra_backchannels": [],
            "extra_commands": [],
            "adapt": True,  # learn from cut-ins: user resumed right after our TURN_END -> wait longer
            "adapt_window_ms": 1500,
            "context_turns": 2,
            "lexicon_override": True,  # bare stop command / backchannel = done, whatever the models say  # last agent/user messages handed to context-aware scorers
        }
        o.update(self.DEFAULTS)
        unknown = set(kw) - set(o)
        if unknown:
            raise TypeError(f"{type(self).__name__}: unknown options {sorted(unknown)}")
        o.update(kw)
        self.o = o
        if o["vad_off_threshold"] is None:
            o["vad_off_threshold"] = o["vad_threshold"] if o["vad"] == "energy" else max(0.05, o["vad_threshold"] - 0.15)
        self.vad = None
        self.scorers: list = []
        self.in_speech = False
        self._run_ms = 0
        self._silence_ms = 0
        self._audio = bytearray()
        self._audio_cap = 8 * SAMPLE_RATE * 2
        self._history: list[tuple[str, str]] = []  # (role, text) for context-aware scorers
        self._last_agent_text = ""
        self._scale = 1.0  # adaptive multiplier on the pause timeout
        self._last_turn_end: float | None = None
        self.latencies: list[float] = []  # seconds per blocking scorer call (diagnostics)
        self._reset_utterance()

    # ---- lifecycle ---------------------------------------------------------------------------

    async def start(self) -> None:
        import asyncio

        from duplexagent.plugins.turn.eot_models import make_scorer
        from duplexagent.plugins.turn.vad import make_vad

        loop = asyncio.get_running_loop()
        # model loading may download weights; keep it off the event loop
        self.vad = await loop.run_in_executor(None, lambda: make_vad(self.o["vad"], energy_threshold=self.o["energy_threshold"]))
        self.scorers = []
        for name in self.o["eot"]:
            cfg = self.o["eot_config"].get(name, {})
            self.scorers.append(await loop.run_in_executor(None, lambda n=name, c=cfg: make_scorer(n, c)))
        for s in self.scorers:  # warm-up so the first real call is not slow
            if s.blocking:
                await loop.run_in_executor(_executor(), lambda s=s: s("你好", bytes(3200), []))

    def _ensure_started(self) -> None:
        # unit tests may call on_audio without start(); fall back to the stdlib parts
        if self.vad is None:
            from duplexagent.plugins.turn.eot_models import TextRuleEOT
            from duplexagent.plugins.turn.vad import make_vad

            self.vad = make_vad(self.o["vad"], energy_threshold=self.o["energy_threshold"])
            self.scorers = [TextRuleEOT()] if "text" in self.o["eot"] else []

    # ---- state --------------------------------------------------------------------------------

    def _reset_utterance(self) -> None:
        self.utt_active = False
        self.utt_text = ""
        self._committed = ""
        self.utt_final = False
        self.voiced_during_agent_ms = 0
        self.barged = False
        self._reset_pause()

    def _reset_pause(self) -> None:
        self._pending: dict[str, Any] = {}  # scorer name -> future
        self._pending_key: dict[str, Any] = {}  # scorer name -> what it was computed on

    # ---- helpers ------------------------------------------------------------------------------

    def is_backchannel(self, text: str) -> bool:
        return lexicon.is_backchannel(text, self.o["extra_backchannels"])

    def _submit(self, scorer, key, *args) -> None:
        if self._pending_key.get(scorer.name) == key:
            return
        self._pending_key[scorer.name] = key
        if scorer.blocking:
            import time

            def run():
                t0 = time.perf_counter()
                try:
                    return scorer(*args)
                finally:
                    self.latencies.append(time.perf_counter() - t0)

            self._pending[scorer.name] = _executor().submit(run)
        else:
            self._pending[scorer.name] = _Done(scorer(*args))

    def _update_scores(self) -> None:
        """(Re)launch scorers for the current pause; text scorers rerun when the words change."""
        history = self._history[-self.o["context_turns"] :]
        for s in self.scorers:
            if s.needs == "audio":
                if self._silence_ms >= self.o["audio_probe_ms"]:
                    self._submit(s, "pause", self.utt_text, bytes(self._audio), history)
            else:
                self._submit(s, self.utt_text, self.utt_text, None, history)

    def p_done(self) -> tuple[float | None, bool]:
        """Fused P(done) over finished scorers, and whether any scorer is still running."""
        t = lexicon.norm(self.utt_text)
        if self.o["lexicon_override"] and t and (t in lexicon.STOP_SET or self.is_backchannel(t)):
            return 0.95, False  # "停" / "嗯嗯": interaction tokens chat LMs and prosody models don't cover
        total = wsum = 0.0
        waiting = False
        for s in self.scorers:
            f = self._pending.get(s.name)
            if f is None or not f.done():
                waiting = True
                continue
            try:
                p = f.result()
            except Exception:
                continue  # a failing model must never take the conversation down
            w = self.o["eot_weights"].get(s.name, 1.0)
            total += w * p
            wsum += w
        return (total / wsum if wsum else None), waiting

    def required_silence_ms(self) -> float:
        p, waiting = self.p_done()
        o = self.o
        if p is None:
            need = o["base_silence_ms"] if not self.scorers else o["max_silence_ms"]
        elif p >= o["p_high"]:
            need = o["min_silence_ms"]
        elif p <= o["p_low"]:
            need = o["max_silence_ms"]
        else:
            k = (p - o["p_low"]) / (o["p_high"] - o["p_low"])
            need = o["max_silence_ms"] - k * (o["max_silence_ms"] - o["min_silence_ms"])
        if waiting and p is not None:
            need = max(need, o["base_silence_ms"])  # some scorer still thinking: don't commit too early
        need = need * self._scale if o["adapt"] else need
        return min(need, o["max_silence_ms"] * (self._scale if o["adapt"] else 1.0))

    # ---- barge-in -----------------------------------------------------------------------------

    def _maybe_barge(self, ts: float, ctx: TurnContext) -> list[TurnEvent]:
        if self.barged or not ctx.agent_speaking or not self.utt_active:
            return []
        text = self.utt_text
        o = self.o
        if text and o["echo_guard"] and lexicon.is_echo(text, ctx.agent_text):
            return []
        if text and o["command_barge"] and lexicon.is_stop_command(text, o["extra_commands"]):
            return self._barge(ts)
        if not self.in_speech or self.voiced_during_agent_ms < o["barge_in_ms"]:
            return []
        if text:
            if self.is_backchannel(text) or lexicon.is_backchannel_prefix(text):
                return []
            return self._barge(ts)
        if self.voiced_during_agent_ms >= o["barge_in_max_ms"]:
            return self._barge(ts)
        return []

    def _barge(self, ts: float) -> list[TurnEvent]:
        self.barged = True
        return [TurnEvent(T.BARGE_IN, ts, self.utt_text)]

    # ---- TurnDetector -------------------------------------------------------------------------

    def on_audio(self, frame: AudioFrame, ctx: TurnContext) -> list[TurnEvent]:
        self._ensure_started()
        if ctx.agent_text:
            self._last_agent_text = ctx.agent_text
        self._audio += frame.pcm
        if len(self._audio) > self._audio_cap:
            del self._audio[: len(self._audio) - self._audio_cap]
        out: list[TurnEvent] = []
        score = self.vad(frame.pcm)
        voiced = score >= (self.o["vad_off_threshold"] if self.in_speech else self.o["vad_threshold"])
        if voiced:
            self._run_ms += FRAME_MS
            self._silence_ms = 0
            if not self.in_speech and self._run_ms >= self.o["min_speech_ms"]:
                self.in_speech = True
                if self.utt_active:
                    self._reset_pause()  # user resumed: forget the pause's verdicts
                else:
                    self.utt_active = True
                self._note_resume(frame.ts)
                out.append(TurnEvent(T.SPEECH_START, frame.ts))
            if self.in_speech and ctx.agent_speaking:
                self.voiced_during_agent_ms += FRAME_MS
                out += self._maybe_barge(frame.ts, ctx)
            return out

        self._run_ms = 0
        self._silence_ms += FRAME_MS
        if self.in_speech and self._silence_ms >= self.o["hangover_ms"]:
            self.in_speech = False
            out.append(TurnEvent(T.SPEECH_END, frame.ts - self._silence_ms / 1000))
        if self.utt_active and not self.in_speech:
            if self.scorers:
                self._update_scores()
            need = self.required_silence_ms() if self.scorers else self.o["base_silence_ms"]
            if self._silence_ms >= need:
                if self.utt_final or not self.utt_text or self._silence_ms >= need + self.o["final_wait_ms"]:
                    out += self._finish(frame.ts, ctx)
        return out

    def on_transcript(self, t: Transcript, ctx: TurnContext) -> list[TurnEvent]:
        if not self.utt_active:
            # A final for an utterance we already closed (late ASR) must not open a new turn.
            # A partial may legitimately run ahead of the VAD: keep its words for when voice starts
            # (e.g. a one-character "停"), but don't act on them yet.
            if not t.is_final:
                self.utt_text = t.text
            return []
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
            echo = self.o["echo_guard"] and ctx.agent_speaking and lexicon.is_echo(text, ctx.agent_text)
            if echo:
                pass
            elif ctx.agent_speaking and not self.barged and self.is_backchannel(text):
                out.append(TurnEvent(T.BACKCHANNEL, ts, text))
            else:
                if ctx.agent_speaking and not self.barged:
                    out.append(TurnEvent(T.BARGE_IN, ts, text))  # short but meaningful
                out.append(TurnEvent(T.TURN_END, ts, text))
                if self._last_agent_text:
                    self._history.append(("assistant", self._last_agent_text))
                    self._last_agent_text = ""
                self._history.append(("user", text))
                self._history = self._history[-8:]
                self._last_turn_end = ts
        self._reset_utterance()
        return out

    def _note_resume(self, ts: float) -> None:
        """Speech started again soon after our TURN_END: we probably cut the user off."""
        if not self.o["adapt"] or self._last_turn_end is None:
            return
        gap = ts - self._last_turn_end
        if gap * 1000 <= self.o["adapt_window_ms"]:
            self._scale = min(1.6, self._scale * 1.2)
        else:
            self._scale = max(1.0, self._scale * 0.95)
        self._last_turn_end = None


def _preset(name: str, **defaults: Any):
    cls = type(f"Hybrid_{name}", (HybridTurnDetector,), {"DEFAULTS": defaults, "__doc__": f"hybrid preset `{name}`"})
    return register("turn", name)(cls)


_preset("adaptive", vad="energy", eot=["text"])
_preset("silero", vad="silero", eot=[], command_barge=False, echo_guard=False, adapt=False)
_preset("ten_vad", vad="ten", eot=[], command_barge=False, echo_guard=False, adapt=False)
_preset("smart_turn", vad="silero", eot=["smart_turn"])
_preset("lm_eot", vad="silero", eot=["lm"])
_preset("ten_turn", vad="silero", eot=["ten_turn"])
_preset("fusion", vad="silero", eot=["smart_turn", "text"])
