"""Rule-router fast brain: rich Chinese rules for delegate / cancel / amend / status, template speech.

No model. It is both a latency floor and the first stage of the `hybrid` fast brain.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from duplexagent.core.interfaces import FastBrain
from duplexagent.core.registry import register
from duplexagent.core.types import AmendTask, BrainContext, CancelTask, Delegate, FastAction, Say
from duplexagent.plugins.fast.common import Route, ack_for, fusion_text, route, running_tasks, short_query, status_text

CANNED = {
    "你好": "你好呀，有什么可以帮你？",
    "谢谢": "不客气。",
    "再见": "再见，回头聊。",
    "晚安": "晚安，好梦。",
    "你是谁": "我是你的语音助手，可以陪你聊天，复杂的事我会交给后台去办。",
    "你叫什么": "我是你的语音助手，你可以随便叫我。",
}


def delegate_context(ctx: BrainContext, limit: int = 2) -> str:
    """Recent user turns, so a follow-up like “那上海呢” stays self-contained for the slow brain."""
    prev = [m.content for m in ctx.history[:-1] if m.role == "user"][-limit:]
    return ("此前用户说过：" + "；".join(prev)) if prev else ""


def structured_actions(r: Route, ctx: BrainContext) -> tuple[list, str] | None:
    """Actions + speech for the routes that need no language model (None for chat)."""
    running = running_tasks(ctx)
    if r.kind == "cancel" and r.task:
        return [CancelTask(r.task.id)], f"好的，「{short_query(r.task.query)}」已经取消了。"
    if r.kind == "status":
        return [], status_text(r.task, running)
    if r.kind == "amend" and r.task:
        return [AmendTask(r.task.id, ctx.user_text)], "好的，我把这个要求告诉后台了。"
    if r.kind == "delegate":
        return [Delegate(query=ctx.user_text, context=delegate_context(ctx))], ack_for(r.topic)
    if r.kind == "clarify":
        return [], "你想让我帮你做什么呢？可以说具体一点。"
    return None


@register("fast", "router")
class RouterFastBrain(FastBrain):
    def __init__(self, think_ms: int = 0, chunk_chars: int = 6) -> None:
        self.think_ms = think_ms
        self.chunk_chars = chunk_chars

    async def _say(self, text: str) -> AsyncIterator[Say]:
        for i in range(0, len(text), self.chunk_chars):
            yield Say(text[i : i + self.chunk_chars])
            await asyncio.sleep(0)

    async def respond(self, ctx: BrainContext) -> AsyncIterator[FastAction]:
        if self.think_ms:
            await asyncio.sleep(self.think_ms / 1000)
        if ctx.trigger == "task_result":
            for t in ctx.deliver:
                async for s in self._say(fusion_text(ctx, t)):
                    yield s
            return
        r = route(ctx)
        got = structured_actions(r, ctx)
        if got is None:
            text = ctx.user_text
            reply = next((v for k, v in CANNED.items() if k in text), None) or "嗯，我明白了，你接着说。"
            actions, speech = [], reply
        else:
            actions, speech = got
        # Act first: spawning is instant, and a barge-in during the receipt must not lose the task.
        for a in actions:
            yield a
        async for s in self._say(speech):
            yield s
