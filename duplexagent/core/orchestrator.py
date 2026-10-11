"""Slow-brain task bus: spawn / cancel / amend tasks and track their state."""

from __future__ import annotations

import asyncio

from duplexagent.core.interfaces import SlowBrain
from duplexagent.core.timeline import Timeline
from duplexagent.core.types import Delegate, TaskRecord, TaskStatus


class TaskOrchestrator:
    def __init__(self, slow: SlowBrain, clock, timeline: Timeline, max_concurrency: int = 4) -> None:
        self.slow = slow
        self.clock = clock
        self.timeline = timeline
        self.tasks: list[TaskRecord] = []
        self._runners: dict[str, asyncio.Task] = {}
        self._sem = asyncio.Semaphore(max_concurrency)

    def spawn(self, d: Delegate, origin_turn: int) -> TaskRecord:
        rec = TaskRecord(query=d.query, context=d.context, priority=d.priority, origin_turn=origin_turn)
        rec.created_at = self.clock()
        self.tasks.append(rec)
        self._runners[rec.id] = asyncio.create_task(self._run(rec), name=f"task-{rec.id}")
        self.timeline.log(rec.created_at, "task_spawn", id=rec.id, query=rec.query)
        return rec

    def get(self, task_id: str) -> TaskRecord | None:
        return next((t for t in self.tasks if t.id == task_id), None)

    def cancel(self, task_id: str) -> None:
        rec = self.get(task_id)
        if rec is None or rec.status != TaskStatus.RUNNING:
            return
        rec.status = TaskStatus.CANCELLED
        rec.delivered = True
        runner = self._runners.get(task_id)
        if runner:
            runner.cancel()
        self.timeline.log(self.clock(), "task_cancel", id=task_id)

    def amend(self, task_id: str, info: str) -> None:
        rec = self.get(task_id)
        if rec is not None and rec.status == TaskStatus.RUNNING:
            rec.amendments.append(info)
            self.timeline.log(self.clock(), "task_amend", id=task_id, info=info)

    def running(self) -> list[TaskRecord]:
        return [t for t in self.tasks if t.status == TaskStatus.RUNNING]

    def ready(self) -> list[TaskRecord]:
        return [t for t in self.tasks if t.awaiting_delivery]

    async def _run(self, rec: TaskRecord) -> None:
        partials: list[str] = []
        try:
            async with self._sem:
                async for u in self.slow.run(rec):
                    if u.kind == "progress":
                        rec.progress.append(u.text)
                    elif u.kind == "partial":
                        partials.append(u.text)
                    elif u.kind == "result":
                        rec.result = u.text
                    elif u.kind == "error":
                        rec.result = u.text
                        rec.status = TaskStatus.FAILED
                    self.timeline.log(self.clock(), f"task_{u.kind}", id=rec.id, text=u.text[:200])
            if rec.status == TaskStatus.RUNNING:
                if rec.result is None:
                    rec.result = "".join(partials) or None
                rec.status = TaskStatus.DONE if rec.result is not None else TaskStatus.FAILED
        except asyncio.CancelledError:
            rec.status = TaskStatus.CANCELLED
            raise
        except Exception as e:  # a crashing slow brain must not take the session down
            rec.status = TaskStatus.FAILED
            rec.result = f"{type(e).__name__}: {e}"
            self.timeline.log(self.clock(), "task_crash", id=rec.id, error=rec.result)
        finally:
            rec.finished_at = self.clock()
            self.timeline.log(rec.finished_at, "task_end", id=rec.id, status=rec.status.value)

    async def drain(self, timeout: float) -> None:
        pending = [r for r in self._runners.values() if not r.done()]
        if pending:
            await asyncio.wait(pending, timeout=timeout)

    async def close(self) -> None:
        for r in self._runners.values():
            r.cancel()
        await asyncio.gather(*self._runners.values(), return_exceptions=True)
