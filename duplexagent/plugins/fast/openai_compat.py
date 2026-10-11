"""LLM fast brain over any OpenAI-compatible endpoint, using tools for delegate / cancel."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from duplexagent.core.interfaces import FastBrain
from duplexagent.core.registry import register
from duplexagent.core.types import BrainContext, CancelTask, Delegate, FastAction, Say, TaskStatus

SYSTEM_PROMPT = """你是一个实时语音助手的“快脑”。你的回复会被直接念出来，所以：
- 口语化、简短（通常一两句话），不要用列表、Markdown 或表情。
- 需要查资料、用工具、多步推理、精确计算或写长内容时，调用 delegate 交给后台“慢脑”，同时用一句话告诉用户你已经交给后台、可以继续聊别的。
- 已经交给后台的问题，在结果回来前不要自己猜答案。
- 用户说不需要了，调用 cancel_task。
- 收到后台结果时，用口语转述要点，并点明对应的是之前哪个问题；如果和你之前说的冲突，明确更正。"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "delegate",
            "description": "把复杂任务交给后台慢脑异步处理。query 必须自包含（慢脑看不到对话历史）。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "context": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_task",
            "description": "取消一个正在运行的后台任务。",
            "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]},
        },
    },
]


def build_messages(ctx: BrainContext, system_prompt: str) -> list[dict]:
    running = [t for t in ctx.tasks if t.status == TaskStatus.RUNNING]
    status = "\n".join(f"- [{t.id}] {t.query}（进度：{t.progress[-1] if t.progress else '刚开始'}）" for t in running)
    msgs = [{"role": "system", "content": system_prompt + (f"\n\n正在运行的后台任务：\n{status}" if status else "")}]
    for m in ctx.history:
        if m.role == "task":
            msgs.append({"role": "system", "content": f"后台任务结果（{m.meta.get('task_id')}，问题：{m.meta.get('query')}）：{m.content}"})
        elif m.role == "assistant" and m.meta.get("interrupted"):
            msgs.append({"role": "assistant", "content": m.content + "……（被用户打断）"})
        else:
            msgs.append({"role": m.role, "content": m.content})
    if ctx.trigger == "task_result":
        msgs.append({"role": "system", "content": "后台任务刚完成，请现在把结果自然地告诉用户。"})
    return msgs


@register("fast", "openai_compat")
class OpenAICompatFastBrain(FastBrain):
    def __init__(self, base_url: str, model: str, api_key_env: str = "OPENAI_API_KEY", system_prompt: str = SYSTEM_PROMPT, **params):
        self.cfg = dict(base_url=base_url, model=model, api_key_env=api_key_env, **params)
        self.system_prompt = system_prompt

    async def start(self) -> None:
        from duplexagent.core.llm import ChatClient

        self.client = ChatClient(**self.cfg)

    async def close(self) -> None:
        await self.client.close()

    async def respond(self, ctx: BrainContext) -> AsyncIterator[FastAction]:
        calls: dict[int, dict] = {}
        async for delta in self.client.stream(build_messages(ctx, self.system_prompt), TOOLS):
            if delta.get("content"):
                yield Say(delta["content"])
            for tc in delta.get("tool_calls") or []:
                c = calls.setdefault(tc.get("index", 0), {"name": "", "args": ""})
                fn = tc.get("function") or {}
                c["name"] += fn.get("name") or ""
                c["args"] += fn.get("arguments") or ""
        for c in calls.values():
            try:
                args = json.loads(c["args"] or "{}")
            except json.JSONDecodeError:
                continue
            if c["name"] == "delegate" and args.get("query"):
                yield Delegate(query=args["query"], context=args.get("context", ""))
            elif c["name"] == "cancel_task" and args.get("task_id"):
                yield CancelTask(args["task_id"])
