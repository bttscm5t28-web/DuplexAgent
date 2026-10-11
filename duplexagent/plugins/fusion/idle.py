"""Baseline fusion: deliver finished tasks once both sides have been quiet for `idle_ms`."""

from __future__ import annotations

from duplexagent.core.interfaces import FusionPolicy, FusionState
from duplexagent.core.registry import register
from duplexagent.core.types import TaskRecord


@register("fusion", "idle")
class IdleFusion(FusionPolicy):
    def __init__(self, idle_ms: int = 800, max_per_turn: int = 2) -> None:
        self.idle_ms = idle_ms
        self.max_per_turn = max_per_turn

    def select(self, state: FusionState, ready: list[TaskRecord]) -> list[TaskRecord]:
        if state.user_speaking or (state.now - state.idle_since) * 1000 < self.idle_ms:
            return []
        return sorted(ready, key=lambda t: (-t.priority, t.created_at))[: self.max_per_turn]
