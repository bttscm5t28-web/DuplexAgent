"""Mock slow brain: waits, reports progress, evaluates arithmetic, otherwise returns a stub answer."""

from __future__ import annotations

import ast
import asyncio
import operator
import re
from collections.abc import AsyncIterator

from duplexagent.core.interfaces import SlowBrain
from duplexagent.core.registry import register
from duplexagent.core.types import TaskRecord, TaskUpdate

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def safe_eval(expr: str) -> float:
    def ev(n: ast.AST) -> float:
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported expression")

    return ev(ast.parse(expr, mode="eval").body)


@register("slow", "mock")
class MockSlowBrain(SlowBrain):
    def __init__(self, delay_s: float = 3.0, steps: int = 2) -> None:
        self.delay_s = delay_s
        self.steps = max(1, steps)

    async def run(self, task: TaskRecord) -> AsyncIterator[TaskUpdate]:
        for i in range(self.steps):
            await asyncio.sleep(self.delay_s / self.steps)
            yield TaskUpdate("progress", f"完成了第 {i + 1}/{self.steps} 步")
        m = re.search(r"[\d\.\s\+\-\*/\(\)]{3,}", task.query.replace("×", "*").replace("乘以", "*").replace("加", "+"))
        if m:
            try:
                v = safe_eval(m.group().strip())
                yield TaskUpdate("result", f"{m.group().strip()} 等于 {v:g}")
                return
            except Exception:
                pass
        yield TaskUpdate("result", f"（模拟结果）已经完成「{task.query}」")
