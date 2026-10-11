"""Data types shared by every module. Keep this file dependency-free."""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field
from typing import Any

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000


@dataclass
class AudioFrame:
    """A short chunk of mono int16 PCM captured from the user."""

    pcm: bytes
    ts: float  # engine clock (seconds since engine start) at capture time
    sample_rate: int = SAMPLE_RATE
    # Simulation only: ground-truth text spoken in this frame. Real inputs leave it None;
    # mock components may read it so the pipeline runs without real speech models.
    truth: str | None = None

    @property
    def duration(self) -> float:
        return len(self.pcm) / 2 / self.sample_rate


@dataclass
class Transcript:
    text: str  # full text of the current utterance so far (not a delta)
    is_final: bool
    ts: float
    confidence: float | None = None


class TurnEventType(str, enum.Enum):
    SPEECH_START = "speech_start"  # user voice activity began
    SPEECH_END = "speech_end"  # user voice activity stopped (may resume)
    TURN_END = "turn_end"  # user finished a turn; the agent should respond
    BARGE_IN = "barge_in"  # user interrupts the agent; stop talking now
    BACKCHANNEL = "backchannel"  # "嗯/对/好" while the agent talks; keep talking


@dataclass
class TurnEvent:
    type: TurnEventType
    ts: float
    text: str = ""


@dataclass
class Message:
    role: str  # "user" | "assistant" | "task"
    content: str
    meta: dict[str, Any] = field(default_factory=dict)


# ---- fast brain actions -------------------------------------------------------------


@dataclass
class Say:
    """Text to speak; the engine streams it to TTS as it arrives."""

    text: str


@dataclass
class Delegate:
    """Hand a self-contained task to the slow brain."""

    query: str  # self-contained task description (the slow brain does not see the chat)
    context: str = ""  # optional extra context the fast brain chooses to pass on
    priority: int = 0


@dataclass
class CancelTask:
    task_id: str


@dataclass
class AmendTask:
    task_id: str
    info: str


FastAction = Say | Delegate | CancelTask | AmendTask


# ---- slow brain tasks ---------------------------------------------------------------


class TaskStatus(str, enum.Enum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskUpdate:
    kind: str  # "progress" | "partial" | "result" | "error"
    text: str


_task_ids = itertools.count(1)


@dataclass
class TaskRecord:
    query: str
    context: str = ""
    priority: int = 0
    origin_turn: int = 0
    id: str = field(default_factory=lambda: f"t{next(_task_ids)}")
    status: TaskStatus = TaskStatus.RUNNING
    progress: list[str] = field(default_factory=list)
    amendments: list[str] = field(default_factory=list)
    result: str | None = None
    created_at: float = 0.0
    finished_at: float | None = None
    delivered: bool = False  # result has been fused back into the conversation

    @property
    def awaiting_delivery(self) -> bool:
        return self.status in (TaskStatus.DONE, TaskStatus.FAILED) and not self.delivered


# ---- TTS ----------------------------------------------------------------------------


@dataclass
class TTSChunk:
    pcm: bytes
    text: str  # the slice of input text this audio covers ("" if unknown)
    sample_rate: int = SAMPLE_RATE

    @property
    def duration(self) -> float:
        return len(self.pcm) / 2 / self.sample_rate


# ---- context handed to the fast brain ------------------------------------------------


@dataclass
class BrainContext:
    trigger: str  # "user_turn" | "task_result"
    history: list[Message]
    user_text: str = ""  # the user turn that triggered this response
    tasks: list[TaskRecord] = field(default_factory=list)  # all tasks of this session
    deliver: list[TaskRecord] = field(default_factory=list)  # results to fuse now
    turn_id: int = 0
