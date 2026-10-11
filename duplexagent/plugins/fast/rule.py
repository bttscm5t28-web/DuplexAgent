"""Rule-based fast brain: keyword delegation, canned chit-chat, task status and result fusion.

No model needed; it exists so the skeleton runs end to end and as a latency floor.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from duplexagent.core.interfaces import FastBrain
from duplexagent.core.registry import register
from duplexagent.core.types import BrainContext, CancelTask, Delegate, FastAction, Say, TaskStatus

DELEGATE_KEYWORDS = ["查", "搜", "计算", "算一下", "分析", "规划", "写一", "总结", "调研", "比较", "翻译", "search", "calculate", "analyze", "plan"]
STATUS_KEYWORDS = ["好了吗", "查到了吗", "进度", "怎么样了", "还要多久"]
CANCEL_KEYWORDS = ["算了", "不用查了", "不用了", "取消"]
CANNED = {"你好": "你好呀，有什么可以帮你？", "谢谢": "不客气。", "再见": "再见，回头聊。", "你是谁": "我是你的语音助手。"}


@register("fast", "rule")
class RuleFastBrain(FastBrain):
    def __init__(self, think_ms: int = 150, chunk_chars: int = 4, delegate_keywords: list[str] | None = None):
        self.think_ms = think_ms
        self.chunk_chars = chunk_chars
        self.delegate_keywords = delegate_keywords or DELEGATE_KEYWORDS

    async def _say(self, text: str) -> AsyncIterator[Say]:
        for i in range(0, len(text), self.chunk_chars):
            yield Say(text[i : i + self.chunk_chars])
            await asyncio.sleep(0.01)

    async def respond(self, ctx: BrainContext) -> AsyncIterator[FastAction]:
        await asyncio.sleep(self.think_ms / 1000)
        if ctx.trigger == "task_result":
            for t in ctx.deliver:
                if t.status == TaskStatus.DONE:
                    msg = f"关于你刚才问的「{t.query}」，结果是：{t.result}。"
                else:
                    msg = f"抱歉，「{t.query}」没能完成：{t.result}。"
                async for s in self._say(msg):
                    yield s
            return

        text = ctx.user_text
        running = [t for t in ctx.tasks if t.status == TaskStatus.RUNNING]
        if running and any(k in text for k in CANCEL_KEYWORDS):
            yield CancelTask(running[-1].id)
            async for s in self._say("好的，已经取消了。"):
                yield s
        elif running and any(k in text for k in STATUS_KEYWORDS):
            t = running[-1]
            p = t.progress[-1] if t.progress else "刚开始"
            async for s in self._say(f"「{t.query}」还在处理，{p}。"):
                yield s
        elif any(k in text for k in self.delegate_keywords):
            async for s in self._say("好的，我让后台去处理，你可以先聊点别的。"):
                yield s
            yield Delegate(query=text)
        else:
            reply = next((v for k, v in CANNED.items() if k in text), None) or f"嗯，你说的是「{text}」，我明白了。"
            async for s in self._say(reply):
                yield s
