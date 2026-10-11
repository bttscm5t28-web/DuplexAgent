"""Claude slow brains.

- `claude`: the shared tool loop (agent.py) over the Anthropic Messages API via the official
  `anthropic` SDK. Our ToolBox (and any MCP tools) are client tools; tool calls run locally in
  the sandbox. Optional `server_web=true` adds Anthropic's server-side web search/fetch tools.
- `claude_agent`: Claude Agent SDK (`claude-agent-sdk`, the Claude Code harness). Our ToolBox is
  exposed as an in-process MCP server and the built-in file/shell tools are disabled, so the agent
  can only use the sandboxed tools (plus optional WebSearch/WebFetch).

Both need ANTHROPIC_API_KEY (or another credential the SDK resolves). Not yet measured against
the real API in this repo — see experiments/slow/REPORT.md.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

from duplexagent.core.registry import register
from duplexagent.core.types import TaskRecord, TaskUpdate
from duplexagent.plugins.slow.agent import AgentSlowBrain, ModelTurn, ToolCall, clip, split_answer
from duplexagent.plugins.slow.toolbox import ToolResult

FALLBACK_BETA = "server-side-fallback-2026-07-01"
UPDATES_BETA = "thinking-display-updates-2026-08-18"
_AGENT_KEYS = {"tools", "web", "max_steps", "time_budget_s", "system_prompt", "preempt_on_amend", "timezone", "fixed_now", "mcp_servers", "amend_poll_s"}


@register("slow", "claude")
class ClaudeSlowBrain(AgentSlowBrain):
    tool_style = "anthropic"

    def __init__(self, model: str = "claude-opus-5-5", effort: str | None = "medium", max_tokens: int = 16000,
                 base_url: str | None = None, api_key_env: str = "ANTHROPIC_API_KEY", fallbacks: bool = True,
                 progress_notes: bool = True, server_web: bool = False, timeout: float = 120.0, **kw):
        unknown = set(kw) - _AGENT_KEYS
        if unknown:
            raise TypeError(f"unknown option(s) for slow/claude: {sorted(unknown)}")
        super().__init__(**kw)
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.fallbacks = fallbacks
        self.progress_notes = progress_notes
        self.server_web = server_web
        self.timeout = timeout
        self.usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0}

    async def start(self) -> None:
        import os

        import anthropic  # optional dependency: pip install anthropic

        await super().start()
        kwargs = {"timeout": self.timeout, "max_retries": 2}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if os.environ.get(self.api_key_env):
            kwargs["api_key"] = os.environ[self.api_key_env]
        self.client = anthropic.AsyncAnthropic(**kwargs)
        self._anthropic = anthropic

    async def close(self) -> None:
        await super().close()
        await self.client.close()

    def _init_messages(self, user: str) -> list:
        return [{"role": "user", "content": user}]

    def _tools(self, allow_tools: bool) -> list[dict]:
        if not allow_tools:
            return []
        tools = self.toolbox.specs("anthropic")
        if self.server_web:
            tools += [{"type": "web_search_20260209", "name": "web_search", "max_uses": 3},
                      {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 3}]
        return tools

    async def _chat(self, messages: list, allow_tools: bool) -> ModelTurn:
        betas, extra = [], {}
        if self.fallbacks:
            betas.append(FALLBACK_BETA)
            extra["fallbacks"] = "default"
        if self.progress_notes:
            betas.append(UPDATES_BETA)
            extra["thinking"] = {"type": "adaptive", "display": "updates"}
        if self.effort:
            extra["output_config"] = {"effort": self.effort}
        tools = self._tools(allow_tools)
        if tools:
            extra["tools"] = tools
        elif any(isinstance(m.get("content"), list) for m in messages):
            # History holds tool_use blocks: tools must stay declared; tool_choice none forbids new calls.
            extra["tools"] = self.toolbox.specs("anthropic")
            extra["tool_choice"] = {"type": "none"}
        req = dict(model=self.model, max_tokens=self.max_tokens, system=self._system(), messages=messages, **extra)
        try:
            resp = await self.client.beta.messages.create(betas=betas, **req) if betas else await self.client.messages.create(**req)
        except self._anthropic.BadRequestError as e:
            if not (self.progress_notes and "display" in str(e)):
                raise
            self.progress_notes = False  # account/model without the progress-updates beta: degrade once
            return await self._chat(messages, allow_tools)
        self.usage["requests"] += 1
        u = getattr(resp, "usage", None)
        if u is not None:
            self.usage["input_tokens"] += getattr(u, "input_tokens", 0) or 0
            self.usage["output_tokens"] += getattr(u, "output_tokens", 0) or 0
        texts, notes, calls = [], [], []
        for b in resp.content:
            t = getattr(b, "type", "")
            if t == "text":
                texts.append(b.text)
            elif t == "thinking" and getattr(b, "thinking", ""):
                notes.append(b.thinking)
            elif t == "tool_use":
                calls.append(ToolCall(b.id, b.name, dict(b.input or {})))
        stop = resp.stop_reason or ""
        return ModelTurn("\n".join(texts), calls, resp.content, stop if stop in ("refusal", "max_tokens", "pause_turn") else "", notes)

    def _append_assistant(self, messages: list, turn: ModelTurn) -> None:
        # Append the full content (thinking + tool_use blocks) unchanged: required for preserved thinking.
        messages.append({"role": "assistant", "content": turn.raw if turn.raw is not None else turn.text})

    def _append_tool_results(self, messages: list, results: list[tuple[ToolCall, ToolResult]]) -> None:
        blocks = [{"type": "tool_result", "tool_use_id": c.id, "content": r.text, **({} if r.ok else {"is_error": True})}
                  for c, r in results]
        if blocks:  # all results of one assistant turn go in ONE user message
            messages.append({"role": "user", "content": blocks})

    def _append_user(self, messages: list, text: str) -> None:
        if messages and messages[-1]["role"] == "user":
            last = messages[-1]["content"]
            messages[-1]["content"] = (last if isinstance(last, list) else [{"type": "text", "text": last}]) + [{"type": "text", "text": text}]
        else:
            messages.append({"role": "user", "content": text})


@register("slow", "claude_agent")
class ClaudeAgentSDKSlowBrain(AgentSlowBrain):
    """Claude Agent SDK backend. One ClaudeSDKClient session per task; amendments are sent as
    follow-up user messages after interrupting the current turn; cancellation disconnects."""

    def __init__(self, model: str = "claude-opus-5-5", effort: str | None = "medium", max_turns: int = 12,
                 builtin_web: bool = False, cli_path: str | None = None, **kw):
        unknown = set(kw) - _AGENT_KEYS
        if unknown:
            raise TypeError(f"unknown option(s) for slow/claude_agent: {sorted(unknown)}")
        super().__init__(**kw)
        self.model = model
        self.effort = effort
        self.max_turns = max_turns
        self.builtin_web = builtin_web
        self.cli_path = cli_path
        self.cost_usd = 0.0

    def _mcp_server(self, task_id: str):
        from claude_agent_sdk import create_sdk_mcp_server, tool

        trace = self.traces.setdefault(task_id, [])
        sdk_tools = []
        for t in self.toolbox.tools.values():
            def make(t=t):
                @tool(t.name, t.description, t.parameters)
                async def handler(args: dict) -> dict:
                    res = await self.toolbox.call(t.name, args)
                    trace.append({"tool": t.name, "args": args, "ok": res.ok, "out": res.text[:200]})
                    return {"content": [{"type": "text", "text": res.text}], **({} if res.ok else {"is_error": True})}
                return handler
            sdk_tools.append(make())
        return create_sdk_mcp_server("slowtools", tools=sdk_tools)

    def _options(self, task_id: str):
        from claude_agent_sdk import ClaudeAgentOptions

        names = [f"mcp__slowtools__{n}" for n in self.toolbox.tools]
        builtin = ["WebSearch", "WebFetch"] if self.builtin_web else []
        opts = dict(
            model=self.model,
            system_prompt=self.system_prompt,
            mcp_servers={"slowtools": self._mcp_server(task_id)},
            tools=builtin,  # no Bash/Read/Write/Edit: only sandboxed tools
            allowed_tools=names + builtin,
            max_turns=self.max_turns,
            permission_mode="dontAsk",
        )
        if self.effort:
            opts["effort"] = self.effort
        if self.cli_path:
            opts["cli_path"] = self.cli_path
        try:
            return ClaudeAgentOptions(**opts)
        except TypeError:  # older SDKs: drop fields they don't know
            for k in ("effort", "tools", "permission_mode"):
                opts.pop(k, None)
            return ClaudeAgentOptions(**opts)

    async def run(self, task: TaskRecord) -> AsyncIterator[TaskUpdate]:
        from claude_agent_sdk import AssistantMessage, ClaudeSDKClient, ResultMessage, TextBlock, ToolUseBlock

        yield TaskUpdate("progress", f"开始处理：{clip(task.query, 30)}")
        t0 = time.monotonic()
        seen = len(task.amendments)
        prompt = self.task_prompt(task)
        if task.amendments:
            prompt += "\n用户补充：" + "；".join(task.amendments)
        client = ClaudeSDKClient(options=self._options(task.id))
        await client.connect()
        try:
            await client.query(prompt)
            while True:
                final_text, restart = "", False
                stream = client.receive_response().__aiter__()
                while True:
                    nxt = asyncio.ensure_future(stream.__anext__())
                    while not nxt.done():
                        await asyncio.wait({nxt}, timeout=self.amend_poll_s)
                        if len(task.amendments) > seen and self.preempt_on_amend and not nxt.done():
                            restart = True
                            break
                        if time.monotonic() - t0 > self.time_budget_s and not nxt.done():
                            await client.interrupt()
                    if restart:
                        nxt.cancel()
                        await asyncio.gather(nxt, return_exceptions=True)
                        await client.interrupt()
                        break
                    try:
                        msg = nxt.result()
                    except StopAsyncIteration:
                        break
                    if isinstance(msg, AssistantMessage):
                        for b in msg.content:
                            if isinstance(b, ToolUseBlock):
                                name = b.name.rsplit("__", 1)[-1]
                                yield TaskUpdate("progress", f"正在{self.toolbox.label(name, b.input)}")
                            elif isinstance(b, TextBlock) and b.text.strip():
                                final_text = b.text
                    elif isinstance(msg, ResultMessage):
                        self.cost_usd += msg.total_cost_usd or 0.0
                        if msg.result:
                            final_text = msg.result
                        if msg.is_error and not final_text:
                            yield TaskUpdate("error", f"后台出错：{msg.subtype}")
                            return
                if len(task.amendments) > seen:
                    new = task.amendments[seen:]
                    seen = len(task.amendments)
                    yield TaskUpdate("progress", f"收到补充「{clip('；'.join(new), 30)}」，按新条件重新考虑")
                    await client.query("用户补充：" + "；".join(new) + "\n（请以补充为准重新考虑，最终答案要体现补充的条件。）")
                    continue
                spoken, detail = split_answer(final_text)
                if not spoken:
                    yield TaskUpdate("error", "后台没有得出结论")
                    return
                if detail:
                    self.details[task.id] = detail
                    yield TaskUpdate("partial", f"详细：{detail}")
                yield TaskUpdate("result", spoken)
                return
        finally:
            await client.disconnect()
