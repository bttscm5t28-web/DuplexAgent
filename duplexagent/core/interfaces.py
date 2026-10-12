"""Abstract interfaces of the swappable modules.

Data flow:

  AudioInput ──frames──► ASR ──transcripts──┐
       └──────frames──► TurnDetector ◄──────┘ ──turn events──► Engine
  Engine ──BrainContext──► FastBrain ──Say/Delegate──► TTS ──chunks──► AudioOutput
                                          └──► TaskOrchestrator ──► SlowBrain
  TaskOrchestrator ──finished tasks──► FusionPolicy ──"deliver now"──► FastBrain

Every plugin gets `start()` / `close()` hooks for loading models or opening connections.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass

from duplexagent.core.types import (
    AudioFrame,
    BrainContext,
    FastAction,
    TaskRecord,
    TaskUpdate,
    Transcript,
    TTSChunk,
    TurnEvent,
)


class Plugin:
    plugin_name: str = "?"

    async def start(self) -> None:
        """Load models / open connections. Called once before the session."""

    async def close(self) -> None:
        """Release resources. Called once after the session."""


class AudioInput(Plugin, abc.ABC):
    @abc.abstractmethod
    def frames(self) -> AsyncIterator[AudioFrame]:
        """Yield frames in real time (each frame's ts set from `clock()`). Ends when input ends."""

    def bind_clock(self, clock) -> None:
        self.clock = clock


class AudioOutput(Plugin, abc.ABC):
    @abc.abstractmethod
    async def play(self, chunk: TTSChunk) -> None:
        """Play a chunk; return once it has been played (paces the speaker in real time)."""

    async def stop(self) -> None:
        """Drop anything buffered. Called on barge-in."""

    def bind_clock(self, clock) -> None:
        self.clock = clock


class ASR(Plugin, abc.ABC):
    @abc.abstractmethod
    def transcribe(self, frames: AsyncIterator[AudioFrame]) -> AsyncIterator[Transcript]:
        """Streaming recognition: yield partial transcripts, then a final one per utterance.

        `Transcript.text` is the full utterance so far. After a final, the next
        transcript starts a new utterance.
        """


@dataclass
class TurnContext:
    now: float
    agent_speaking: bool  # agent holds the floor: responding (thinking or playing audio)
    agent_text: str = ""  # what the agent is currently saying (for context-aware models)
    # Agent audio is actually playing. While it is not (agent still thinking, audio on hold), an
    # early barge-in buys nothing: detectors should decide at the end of the utterance, on the final text.
    agent_audible: bool = True


class TurnDetector(Plugin, abc.ABC):
    """Voice activity, end-of-turn and interruption decisions."""

    @abc.abstractmethod
    def on_audio(self, frame: AudioFrame, ctx: TurnContext) -> list[TurnEvent]:
        """Called for every input frame (silence included), so it can also run timers."""

    @abc.abstractmethod
    def on_transcript(self, t: Transcript, ctx: TurnContext) -> list[TurnEvent]:
        ...


class FastBrain(Plugin, abc.ABC):
    @abc.abstractmethod
    def respond(self, ctx: BrainContext) -> AsyncIterator[FastAction]:
        """Stream actions for one agent turn. Yield Say chunks as early as possible."""


class SlowBrain(Plugin, abc.ABC):
    @abc.abstractmethod
    def run(self, task: TaskRecord) -> AsyncIterator[TaskUpdate]:
        """Work on a task; yield progress/partial updates, end with a result or error.

        `task.amendments` may grow while running (user added information).
        """


class TTS(Plugin, abc.ABC):
    @abc.abstractmethod
    def synthesize(self, text: AsyncIterator[str]) -> AsyncIterator[TTSChunk]:
        """Streaming synthesis from a stream of text pieces. Set TTSChunk.text for alignment."""


@dataclass
class FusionState:
    now: float
    user_speaking: bool
    idle_since: float  # last time either side was active
    # select() is only called while the agent is not responding.


class FusionPolicy(Plugin, abc.ABC):
    @abc.abstractmethod
    def select(self, state: FusionState, ready: list[TaskRecord]) -> list[TaskRecord]:
        """Return the finished tasks to fuse into the conversation now (may be empty)."""
