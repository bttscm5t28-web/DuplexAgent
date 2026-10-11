"""LLM slow brain over any OpenAI-compatible endpoint (no tools yet; one reasoning call)."""

from __future__ import annotations

from collections.abc import AsyncIterator

from duplexagent.core.interfaces import SlowBrain
from duplexagent.core.registry import register
from duplexagent.core.types import TaskRecord, TaskUpdate

SYSTEM_PROMPT = "你是语音助手的后台“慢脑”。认真推理后给出准确、完整但精炼的结论（结论会被转述给用户听），先给结论再给关键依据。"


@register("slow", "openai_compat")
class OpenAICompatSlowBrain(SlowBrain):
    def __init__(self, base_url: str, model: str, api_key_env: str = "OPENAI_API_KEY", system_prompt: str = SYSTEM_PROMPT, **params):
        self.cfg = dict(base_url=base_url, model=model, api_key_env=api_key_env, **params)
        self.system_prompt = system_prompt

    async def start(self) -> None:
        from duplexagent.core.llm import ChatClient

        self.client = ChatClient(**self.cfg)

    async def close(self) -> None:
        await self.client.close()

    async def run(self, task: TaskRecord) -> AsyncIterator[TaskUpdate]:
        yield TaskUpdate("progress", "开始推理")
        user = task.query + (f"\n\n补充上下文：{task.context}" if task.context else "")
        if task.amendments:
            user += "\n\n用户补充：" + "；".join(task.amendments)
        msg = await self.client.complete([{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user}])
        yield TaskUpdate("result", msg.get("content") or "")
