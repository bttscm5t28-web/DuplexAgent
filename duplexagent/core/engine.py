"""The duplex engine: wires the swappable modules and runs the conversation state machine.

States (derived, not stored):
  LISTENING  no response task running
  THINKING   response task running, no agent audio yet
  SPEAKING   agent audio playing
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from duplexagent.core.aio import Channel
from duplexagent.core.interfaces import (
    ASR,
    TTS,
    AudioInput,
    AudioOutput,
    FastBrain,
    FusionPolicy,
    FusionState,
    SlowBrain,
    TurnContext,
    TurnDetector,
)
from duplexagent.core.orchestrator import TaskOrchestrator
from duplexagent.core.timeline import Timeline
from duplexagent.core.types import (
    AmendTask,
    AudioFrame,
    BrainContext,
    CancelTask,
    Delegate,
    Message,
    Say,
    TaskRecord,
    TurnEvent,
    TurnEventType,
)


@dataclass
class Components:
    input: AudioInput
    output: AudioOutput
    asr: ASR
    turn: TurnDetector
    fast: FastBrain
    slow: SlowBrain
    tts: TTS
    fusion: FusionPolicy

    def all(self):
        return [self.input, self.output, self.asr, self.turn, self.fast, self.slow, self.tts, self.fusion]


@dataclass
class EngineConfig:
    fusion_tick_ms: int = 50
    drain_timeout: float = 15.0  # after input ends, wait this long for tasks/responses
    cancel_on_resume: bool = True  # react when the user speaks again while the agent is still thinking
    # "hold": keep thinking but don't start audio until the turn detector decides (backchannel ->
    # release, barge-in / new turn -> drop and merge); "cancel": drop the response right away.
    resume_policy: str = "hold"
    hold_release_ms: int = 2500  # release a hold if the detector never decides (noise, cough)
    max_redeliver: int = 2  # a fused task result cut off by the user is re-queued at most this often
    merge_window_ms: int = 1000  # user resumed this soon after their turn ended: same turn, merge
    max_tasks: int = 4
    max_history: int = 40


class DuplexEngine:
    def __init__(self, c: Components, cfg: EngineConfig | None = None, timeline: Timeline | None = None) -> None:
        self.c = c
        self.cfg = cfg or EngineConfig()
        self.timeline = timeline or Timeline()
        self.history: list[Message] = []
        self._t0 = 0.0
        self._turn_id = 0
        self._response: asyncio.Task | None = None
        self._playing = False
        self._agent_text = ""
        self.user_speaking = False
        self._last_activity = 0.0
        self._last_user_speech_end: float | None = None
        self._may_speak: asyncio.Event | None = None  # cleared while a response is on hold
        self._held_since: float | None = None
        self._merge_next = False  # last user turn got no answer out: append the next turn to it
        self._last_turn_end = 0.0
        self._hold_gap: float | None = None  # turn end -> user resumed, for the hold in progress
        self._response_trigger = ""
        self._redelivered: dict[str, int] = {}
        self.orch: TaskOrchestrator | None = None

    # ---- clock ------------------------------------------------------------------------

    def now(self) -> float:
        return asyncio.get_running_loop().time() - self._t0

    def log(self, name: str, **data) -> None:
        self.timeline.log(self.now(), name, **data)

    # ---- lifecycle --------------------------------------------------------------------

    async def run(self) -> None:
        self._t0 = asyncio.get_running_loop().time()
        self.c.input.bind_clock(self.now)
        self.c.output.bind_clock(self.now)
        self.orch = TaskOrchestrator(self.c.slow, self.now, self.timeline, self.cfg.max_tasks)
        self._may_speak = asyncio.Event()
        self._may_speak.set()
        for p in self.c.all():
            await p.start()
        # Restart the session clock once every plugin is loaded: model loading can take seconds,
        # and scenario times / metrics are relative to when audio starts flowing.
        self._t0 = asyncio.get_running_loop().time()
        self.log("session_start", modules={k: getattr(self.c, k).plugin_name for k in Components.__annotations__})
        asr_in: Channel[AudioFrame] = Channel()
        asr_task = asyncio.create_task(self._asr_loop(asr_in), name="asr")
        fusion_task = asyncio.create_task(self._fusion_loop(), name="fusion")
        try:
            await self._pump(asr_in)
            await self._drain()
        finally:
            asr_in.close()
            for t in (asr_task, fusion_task, self._response):
                if t:
                    t.cancel()
            await asyncio.gather(*(t for t in (asr_task, fusion_task, self._response) if t), return_exceptions=True)
            await self.orch.close()
            for p in self.c.all():
                await p.close()
            self.log("session_end")

    async def _drain(self) -> None:
        deadline = self.now() + self.cfg.drain_timeout
        while self.now() < deadline:
            busy = (self._response and not self._response.done()) or self.orch.running() or self.orch.ready()
            if not busy:
                return
            await asyncio.sleep(0.05)
        self.log("drain_timeout")

    # ---- input side -------------------------------------------------------------------

    def _turn_ctx(self) -> TurnContext:
        # The agent holds the floor from the moment it starts a response, not only once audio plays:
        # user speech while it is thinking is then classified like speech over it (backchannel vs
        # barge-in) instead of always opening a new turn.
        return TurnContext(now=self.now(), agent_speaking=self._playing or self._response_active(),
                           agent_text=self._agent_text, agent_audible=self._playing)

    async def _pump(self, asr_in: Channel[AudioFrame]) -> None:
        async for frame in self.c.input.frames():
            asr_in.put(frame)
            for ev in self.c.turn.on_audio(frame, self._turn_ctx()):
                self._on_turn_event(ev)
        self.log("input_end")

    async def _asr_loop(self, frames: Channel[AudioFrame]) -> None:
        async for t in self.c.asr.transcribe(frames):
            self.log("asr_final" if t.is_final else "asr_partial", text=t.text)
            for ev in self.c.turn.on_transcript(t, self._turn_ctx()):
                self._on_turn_event(ev)

    def _response_active(self) -> bool:
        return self._response is not None and not self._response.done()

    def _on_turn_event(self, ev: TurnEvent) -> None:
        self.timeline.log(ev.ts, ev.type.value, **({"text": ev.text} if ev.text else {}))
        if ev.type == TurnEventType.SPEECH_START:
            self.user_speaking = True
            if self.cfg.cancel_on_resume and self._response_active() and not self._playing:
                if self.cfg.resume_policy == "cancel":
                    self._interrupt("user_resumed")
                elif self._held_since is None:
                    self._hold()
        elif ev.type == TurnEventType.SPEECH_END:
            self.user_speaking = False
            self._last_activity = ev.ts
            self._last_user_speech_end = ev.ts
        elif ev.type == TurnEventType.BACKCHANNEL:
            self._release("backchannel")
        elif ev.type == TurnEventType.BARGE_IN:
            if self._response_active():
                self._interrupt("barge_in" if self._playing else "user_resumed")
        elif ev.type == TurnEventType.TURN_END:
            if self._response_active():
                self._interrupt("new_turn")
            self._release(None)
            last = self.history[-1] if self.history else None
            if self._merge_next and last is not None and last.role == "user":
                # The previous response never got a word out (user kept talking): one turn, not two.
                last.content += ev.text
                self.log("turn_merged", text=last.content)
            else:
                self.history.append(Message("user", ev.text))
            self._merge_next = False
            self._last_turn_end = ev.ts
            self._hold_gap = None
            self._start_response("user_turn", user_text=self.history[-1].content)

    def _hold(self) -> None:
        self._held_since = self.now()
        self._hold_gap = self._held_since - self._last_turn_end
        self._may_speak.clear()
        self.log("hold")

    def _release(self, reason: str | None) -> None:
        if self._held_since is None:
            return
        self._held_since = None
        self._may_speak.set()
        if reason:
            self.log("release", reason=reason)

    def _interrupt(self, reason: str) -> None:
        self.log("interrupt", reason=reason)
        # Decided here, synchronously: the TURN_END that follows may arrive before the task unwinds.
        gap = self._hold_gap
        if not self._playing and self._response_trigger == "user_turn" and gap is not None and gap * 1000 <= self.cfg.merge_window_ms:
            self._merge_next = True
        self._response.cancel()
        asyncio.create_task(self.c.output.stop())

    # ---- response side ----------------------------------------------------------------

    def _start_response(self, trigger: str, user_text: str = "", deliver: list[TaskRecord] | None = None) -> None:
        self._turn_id += 1
        deliver = deliver or []
        for t in deliver:
            t.delivered = True
            self.history.append(Message("task", t.result or "", {"task_id": t.id, "query": t.query, "status": t.status.value}))
        self.history = self.history[-self.cfg.max_history :]
        ctx = BrainContext(
            trigger=trigger,
            history=list(self.history),
            user_text=user_text,
            tasks=list(self.orch.tasks),
            deliver=deliver,
            turn_id=self._turn_id,
        )
        self.log("respond", turn=self._turn_id, trigger=trigger, deliver=[t.id for t in deliver])
        self._response_trigger = trigger
        self._response = asyncio.create_task(self._respond(ctx), name=f"respond-{self._turn_id}")

    async def _respond(self, ctx: BrainContext) -> None:
        text_ch: Channel[str] = Channel()
        said: list[str] = []
        spoken: list[str] = []
        interrupted = False

        async def think() -> None:
            try:
                async for a in self.c.fast.respond(ctx):
                    if isinstance(a, Say):
                        if not said:
                            self.log("fast_first_text", turn=ctx.turn_id)
                        said.append(a.text)
                        self._agent_text = "".join(said)
                        text_ch.put(a.text)
                    elif isinstance(a, Delegate):
                        self.orch.spawn(a, origin_turn=ctx.turn_id)
                    elif isinstance(a, CancelTask):
                        self.orch.cancel(a.task_id)
                    elif isinstance(a, AmendTask):
                        self.orch.amend(a.task_id, a.info)
            finally:
                text_ch.close()

        async def speak() -> None:
            async for chunk in self.c.tts.synthesize(text_ch):
                if not self._playing:
                    await self._may_speak.wait()  # on hold while the user is talking
                    self._playing = True
                    ref = self._last_user_speech_end if ctx.trigger == "user_turn" else None
                    latency = round(self.now() - ref, 3) if ref is not None else None
                    self.log("agent_audio_start", turn=ctx.turn_id, latency=latency)
                await self.c.output.play(chunk)
                spoken.append(chunk.text)

        try:
            await asyncio.gather(think(), speak())
        except asyncio.CancelledError:
            interrupted = True
            raise
        except Exception as e:
            self.log("response_error", turn=ctx.turn_id, error=f"{type(e).__name__}: {e}")
        finally:
            self._playing = False
            self._agent_text = ""
            self._last_activity = self.now()
            played = "".join(spoken)
            if interrupted:
                # A result cut off mid-sentence is not delivered: queue it again (the brain sees what
                # was already said in the history), at most `max_redeliver` times per task.
                for t in ctx.deliver:
                    n = self._redelivered.get(t.id, 0)
                    if not played or n < self.cfg.max_redeliver:
                        t.delivered = False
                        self._redelivered[t.id] = n + (1 if played else 0)
            if played:
                # Only what was actually played goes into history, so the brain knows where it was cut.
                self.history.append(Message("assistant", played, {"interrupted": interrupted, "turn": ctx.turn_id}))
            self.log("agent_end", turn=ctx.turn_id, interrupted=interrupted, played=played)

    # ---- fusion -------------------------------------------------------------------------

    async def _fusion_loop(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.fusion_tick_ms / 1000)
            if self._held_since is not None and not self.user_speaking:
                if (self.now() - max(self._held_since, self._last_user_speech_end or 0)) * 1000 >= self.cfg.hold_release_ms:
                    self._release("timeout")
            ready = self.orch.ready()
            if not ready or self._response_active():
                continue
            state = FusionState(now=self.now(), user_speaking=self.user_speaking, idle_since=self._last_activity)
            chosen = self.c.fusion.select(state, ready)
            if chosen:
                self._start_response("task_result", deliver=chosen)
