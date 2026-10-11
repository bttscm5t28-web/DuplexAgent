"""Tool-using agent slow brain: a backend-agnostic loop plus the OpenAI-compatible backend.

The loop is what makes the slow brain conversation-friendly:

- **Progress**: an immediate "on it" update, then one per tool call saying what is being done
  and what was found ("用计算器算 23*47，得到 1081"), so the fast brain always has something
  true to relay when the user asks "好了吗".
- **Amendments**: `task.amendments` is watched while the model is thinking. A new amendment
  pre-empts the in-flight model call (it would be answering the old question) and is injected
  as a user message; it is also checked once more right before the result is returned.
- **Cancellation**: every await is cancellable; HTTP requests and sandboxed subprocesses are
  torn down in `finally` blocks, so cancel latency is bounded by the event loop, not the model.
- **Spoken + detailed result**: the model answers as `结论：…` (short, speakable) and
  `详细：…`. The result update carries only the spoken part; the detailed part is sent as a
  `partial` update prefixed with `详细：` and kept on `task.detail` / `self.details[task.id]`.
- **Budget**: after `max_steps` tool rounds or `time_budget_s`, the model must answer with what it has.

Backends implement `_chat` and the message-append helpers; see `OpenAICompatAgent` below and
`claude.py` for the Anthropic Messages API.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from duplexagent.core.interfaces import SlowBrain
from duplexagent.core.registry import register
from duplexagent.core.types import TaskRecord, TaskUpdate
from duplexagent.plugins.slow.toolbox import ToolBox, ToolResult

SYSTEM_PROMPT = """你是语音助手的后台“慢脑”，负责认真完成前台转交的任务。前台会把你的结论念给用户听，用户无法和你直接对话。

工作方式：
- 需要计算、日期推算、单位换算、枚举统计时，一定调用工具，不要心算。应用题先列式再用工具算；数列、计数类问题用 python 写代码并打印前几项核对下标。
- 问题涉及“今天、明天、现在、距今”等相对时间时，先调用 now 工具获取当前日期。
- 每次工具返回后检查结果是否合理；出错就换个办法。
- 信息不全时不要反问，按常见情况做合理假设，并在详细里说明假设。
- 如果收到“用户补充”，以补充为准重新考虑，最终答案必须体现补充的条件。
- 不确定的事实要说明不确定，不要编造。

完成后按下面格式给出最终答复（这时不要再调用工具）：
结论：一到两句口语化的话，直接给出答案和关键数字，60字以内，不要用 markdown、列表、公式符号。规划类任务在这里概括两三个要点。
详细：关键依据、步骤或完整方案（可选，可以稍长）。"""

PROMPT_TOOLS_TEMPLATE = """

# 工具
你可以调用下面的工具（JSON Schema 描述）：
<tools>
{tools}
</tools>
需要调用工具时，只输出一行或多行：
<tool_call>{{"name": "工具名", "arguments": {{参数}}}}</tool_call>
然后等待 <tool_response> 返回结果再继续。不需要工具时直接给出最终答复。"""


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class ModelTurn:
    text: str
    calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None  # backend-specific assistant message to append to history
    stop: str = ""  # "refusal", "max_tokens", "pause_turn" when relevant
    notes: list[str] = field(default_factory=list)  # model's own progress notes (e.g. Claude display="updates")


# ---- answer formatting ----------------------------------------------------------------

_MD = [(r"\*\*(.+?)\*\*", r"\1"), (r"`([^`]*)`", r"\1"), (r"^#+\s*", ""), (r"^\s*[-*•]\s+", ""), (r"\\[()\[\]]", ""),
       (r"\\times", "×"), (r"\\div", "÷"), (r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2"), (r"\$", "")]


_SUBSUP = str.maketrans("₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹", "01234567890123456789")


def strip_markdown(s: str) -> str:
    for a, b in _MD:
        s = re.sub(a, b, s, flags=re.M)
    return s.translate(_SUBSUP).strip()


def strip_think(s: str) -> str:
    """Remove <think>…</think> reasoning that some local models (Qwen3, DeepSeek-R1) emit inline."""
    s = re.sub(r"(?s)<think>.*?</think>", "", s or "")
    return re.sub(r"(?s)^.*?</think>", "", s).strip()


def split_answer(text: str, max_spoken: int = 120) -> tuple[str, str]:
    """Return (spoken, detail) from the model's final text."""
    text = strip_think(text)
    m = re.search(r"结论\s*[:：]\s*(.+?)(?=\n\s*详细\s*[:：]|\Z)", text, re.S)
    d = re.search(r"详细\s*[:：]\s*(.+)\Z", text, re.S)
    if m:
        spoken = strip_markdown(m.group(1))
        detail = d.group(1).strip() if d else ""
    else:
        body = strip_markdown(text)
        sentences = re.split(r"(?<=[。！？!?；;])\s*|\n+", body)
        spoken = ""
        for s in sentences:
            if not s.strip():
                continue
            if spoken and len(spoken) + len(s) > max_spoken:
                break
            spoken += s.strip()
        detail = body if len(body) > len(spoken) + 10 else ""
    spoken = re.sub(r"\s*\n\s*", "，", spoken).strip()
    if len(spoken) > max_spoken * 2:
        spoken = spoken[: max_spoken * 2] + "…"
    return spoken, detail


def clip(s: str, n: int = 40) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[:n] + "…"


def parse_prompt_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Parse Hermes/Qwen style `<tool_call>{json}</tool_call>` (also tolerates a missing close tag)."""
    calls = []
    dec = json.JSONDecoder()
    blocks = re.findall(r"(?s)<tool_call>(.*?)(?:</tool_call>|$)", text)
    if not blocks:  # bare JSON / fenced JSON call without tags
        blocks = re.findall(r'(?s)```(?:json)?\s*(\{\s*"name".*?\})\s*```', text) or ([text] if re.match(r'\s*\{\s*"name"', text) else [])
    for b in blocks:
        i = b.find('{"')
        if i < 0:
            continue
        try:
            obj, _ = dec.raw_decode(b[i:])
        except json.JSONDecodeError:
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if obj.get("name"):
            calls.append(ToolCall(f"call_{uuid.uuid4().hex[:8]}", obj["name"], args if isinstance(args, dict) else {}))
    cleaned = re.sub(r"(?s)<tool_call>.*?(</tool_call>|$)", "", text).strip() if calls else text
    return cleaned, calls


# ---- the loop ---------------------------------------------------------------------------


class AgentSlowBrain(SlowBrain):
    """Backend-agnostic tool loop. Subclasses implement the model-specific methods."""

    tool_style = "openai"

    def __init__(self, tools: list[str] | None = None, web: bool = False, max_steps: int = 6, time_budget_s: float = 60.0,
                 system_prompt: str = SYSTEM_PROMPT, preempt_on_amend: bool = True, timezone: str = "Asia/Shanghai",
                 fixed_now: str | None = None, mcp_servers: list[dict] | None = None, amend_poll_s: float = 0.05):
        self.toolbox = ToolBox(enabled=tools, web=web, timezone=timezone, fixed_now=fixed_now)
        self.max_steps = max_steps
        self.time_budget_s = time_budget_s
        self.system_prompt = system_prompt
        self.preempt_on_amend = preempt_on_amend
        self.mcp_servers = mcp_servers or []
        self.amend_poll_s = amend_poll_s
        self._mcp = None
        self.details: dict[str, str] = {}
        self.traces: dict[str, list[dict]] = {}  # task id -> tool calls (for benchmarks)

    async def start(self) -> None:
        if self.mcp_servers:
            from duplexagent.plugins.slow.mcp_tools import MCPToolSource

            self._mcp = MCPToolSource(self.mcp_servers)
            for t in await self._mcp.start():
                self.toolbox.add(t)

    async def close(self) -> None:
        if self._mcp is not None:
            await self._mcp.close()

    # -- backend hooks --------------------------------------------------------------------

    def _system(self) -> str:
        return self.system_prompt

    def _init_messages(self, user: str) -> list:
        raise NotImplementedError

    async def _chat(self, messages: list, allow_tools: bool) -> ModelTurn:
        raise NotImplementedError

    def _append_assistant(self, messages: list, turn: ModelTurn) -> None:
        raise NotImplementedError

    def _append_tool_results(self, messages: list, results: list[tuple[ToolCall, ToolResult]]) -> None:
        raise NotImplementedError

    def _append_user(self, messages: list, text: str) -> None:
        raise NotImplementedError

    # -- loop -----------------------------------------------------------------------------

    @staticmethod
    def task_prompt(task: TaskRecord) -> str:
        user = f"任务：{task.query}"
        if task.context:
            user += f"\n上下文：{task.context}"
        return user

    async def _chat_watch(self, messages: list, allow_tools: bool, task: TaskRecord, seen: int) -> ModelTurn | None:
        """Run one model call; return None if an amendment arrived meanwhile (call pre-empted)."""
        call = asyncio.ensure_future(self._chat(messages, allow_tools))
        try:
            while True:
                done, _ = await asyncio.wait({call}, timeout=self.amend_poll_s)
                if done:
                    return call.result()
                if self.preempt_on_amend and len(task.amendments) > seen:
                    return None
        finally:
            if not call.done():
                call.cancel()
                await asyncio.gather(call, return_exceptions=True)

    async def run(self, task: TaskRecord) -> AsyncIterator[TaskUpdate]:
        t0 = time.monotonic()
        trace = self.traces.setdefault(task.id, [])
        yield TaskUpdate("progress", f"开始处理：{clip(task.query, 30)}")
        messages = self._init_messages(self.task_prompt(task))
        seen, steps, empty = 0, 0, 0
        while True:
            if len(task.amendments) > seen:
                new = task.amendments[seen:]
                seen = len(task.amendments)
                self._append_user(messages, "用户补充：" + "；".join(new) + "\n（请以补充为准重新考虑，最终答案要体现补充的条件。）")
                yield TaskUpdate("progress", f"收到补充「{clip('；'.join(new), 30)}」，按新条件重新考虑")
            final_only = steps >= self.max_steps or time.monotonic() - t0 > self.time_budget_s
            if final_only and steps:
                self._append_user(messages, "时间有限，请不要再调用工具，根据已有信息直接给出最终答复。")
            try:
                turn = await self._chat_watch(messages, not final_only, task, seen)
            except Exception as e:  # model/backend failure: report it in a speakable way
                self.details[task.id] = f"{type(e).__name__}: {e}"
                yield TaskUpdate("error", f"后台模型出错了（{type(e).__name__}），这次没能完成")
                return
            if turn is None:  # pre-empted by an amendment
                continue
            if turn.stop == "refusal":
                yield TaskUpdate("error", "这个请求后台模型拒绝处理")
                return
            for note in turn.notes:
                yield TaskUpdate("progress", f"思路：{clip(note, 40)}")
            if turn.stop == "bad_tool_call":
                empty += 1
                if empty <= 2:
                    self._append_user(messages, "上一条工具调用不完整。请输出完整的工具调用，或者直接按“结论：…”给出最终答复。")
                    continue
                turn.text = re.sub(r"(?s)<tool_call>.*", "", turn.text)
            if turn.stop == "pause_turn" and not turn.calls:  # server-side tool loop paused: resend to continue
                self._append_assistant(messages, turn)
                steps += 1
                continue
            if turn.calls and not final_only:
                self._append_assistant(messages, turn)
                if turn.text.strip():
                    yield TaskUpdate("progress", f"思路：{clip(strip_markdown(strip_think(turn.text)), 40)}")
                results = []
                for c in turn.calls:
                    label = self.toolbox.label(c.name, c.args)
                    yield TaskUpdate("progress", f"正在{label}")
                    res = await self.toolbox.call(c.name, c.args)
                    trace.append({"tool": c.name, "args": c.args, "ok": res.ok, "out": res.text[:200]})
                    results.append((c, res))
                    if res.ok:
                        yield TaskUpdate("partial", f"{label}：{clip(res.text, 80)}")
                        yield TaskUpdate("progress", f"{label}，得到 {clip(res.text, 30)}")
                    else:
                        yield TaskUpdate("progress", f"{label}没成功，换个办法")
                self._append_tool_results(messages, results)
                steps += 1
                continue
            if len(task.amendments) > seen:  # arrived while the answer was being written
                if turn.text.strip():
                    self._append_assistant(messages, ModelTurn(turn.text, [], turn.raw))
                continue
            spoken, detail = split_answer(turn.text)
            if not spoken:
                empty += 1
                if empty <= 1 and not final_only:
                    self._append_user(messages, "请按“结论：…”的格式给出最终答复。")
                    continue
                yield TaskUpdate("error", "后台没有得出结论")
                return
            if detail:
                self.details[task.id] = detail
                try:
                    task.detail = detail  # optional attribute; see REPORT 接口变更请求
                except Exception:
                    pass
                yield TaskUpdate("partial", f"详细：{detail}")
            yield TaskUpdate("result", spoken)
            return


# ---- OpenAI-compatible backend ------------------------------------------------------------


@register("slow", "agent")
class OpenAICompatAgent(AgentSlowBrain):
    """Tool-using agent over any OpenAI-compatible chat endpoint (vLLM, llama.cpp server, Ollama, DeepSeek, Qwen…).

    tool_mode = "native": OpenAI `tools` / `tool_calls` (needs server support, e.g. vLLM
                          `--enable-auto-tool-choice`, llama.cpp `--jinja`).
    tool_mode = "prompt": tools described in the system prompt, calls parsed from
                          `<tool_call>{…}</tool_call>` text — works with any chat endpoint.
    """

    def __init__(self, base_url: str, model: str, api_key_env: str = "OPENAI_API_KEY", tool_mode: str = "native",
                 timeout: float = 120.0, params: dict | None = None, stream: bool = False, **kw):
        agent_keys = {"tools", "web", "max_steps", "time_budget_s", "system_prompt", "preempt_on_amend", "timezone", "fixed_now", "mcp_servers", "amend_poll_s"}
        super().__init__(**{k: v for k, v in kw.items() if k in agent_keys})
        self.params = dict(params or {}, **{k: v for k, v in kw.items() if k not in agent_keys})
        self.cfg = dict(base_url=base_url, model=model, api_key_env=api_key_env, timeout=timeout)
        if tool_mode not in ("native", "prompt"):
            raise ValueError("tool_mode must be 'native' or 'prompt'")
        self.tool_mode = tool_mode
        # stream=True: when a call is cancelled/pre-empted, closing the stream makes the server stop
        # generating (non-streamed requests keep a local server busy until they finish).
        self.stream = stream

    async def start(self) -> None:
        from duplexagent.core.llm import ChatClient

        await super().start()
        self.client = ChatClient(**self.cfg, **self.params)

    async def close(self) -> None:
        await super().close()
        await self.client.close()

    def _system(self) -> str:
        if self.tool_mode == "prompt":
            tools = "\n".join(json.dumps(s["function"], ensure_ascii=False) for s in self.toolbox.specs("openai"))
            return self.system_prompt + PROMPT_TOOLS_TEMPLATE.format(tools=tools)
        return self.system_prompt

    def _init_messages(self, user: str) -> list:
        return [{"role": "system", "content": self._system()}, {"role": "user", "content": user}]

    async def _complete(self, messages: list, tools: list | None, retries: int = 2) -> dict:
        import httpx

        for attempt in range(retries + 1):
            try:
                if self.stream:
                    return await self._complete_stream(messages, tools)
                return await self.client.complete(messages, tools=tools)
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                status = getattr(getattr(e, "response", None), "status_code", 0)
                if attempt == retries or (status and status < 500 and status != 429):
                    raise
                await asyncio.sleep(0.5 * 2**attempt)
        raise RuntimeError("unreachable")

    async def _complete_stream(self, messages: list, tools: list | None) -> dict:
        content, calls = [], {}
        async for d in self.client.stream(messages, tools=tools):
            if d.get("content"):
                content.append(d["content"])
            for tc in d.get("tool_calls") or []:
                c = calls.setdefault(tc.get("index", len(calls)), {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                c["id"] = tc.get("id") or c["id"]
                fn = tc.get("function") or {}
                c["function"]["name"] += fn.get("name") or ""
                c["function"]["arguments"] += fn.get("arguments") or ""
        msg = {"role": "assistant", "content": "".join(content)}
        if calls:
            msg["tool_calls"] = [calls[k] for k in sorted(calls)]
        return msg

    async def _chat(self, messages: list, allow_tools: bool) -> ModelTurn:
        native = self.tool_mode == "native" and allow_tools
        msg = await self._complete(messages, self.toolbox.specs("openai") if native else None)
        text = msg.get("content") or ""
        if self.tool_mode == "prompt" or not msg.get("tool_calls"):
            # Some servers return text-form tool calls even in native mode; parse them either way.
            cleaned, calls = parse_prompt_tool_calls(strip_think(text)) if allow_tools else (text, [])
            if calls:
                return ModelTurn(cleaned, calls, {"role": "assistant", "content": text})
            bad = "<tool_call>" in text  # a tool call the model started but never completed
            return ModelTurn(text, [], {"role": "assistant", "content": text}, "bad_tool_call" if bad else "")
        calls = []
        for tc in msg["tool_calls"]:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            calls.append(ToolCall(tc.get("id") or f"call_{uuid.uuid4().hex[:8]}", fn.get("name", ""), args))
        raw = {"role": "assistant", "content": text, "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.args, ensure_ascii=False)}} for c in calls]}
        return ModelTurn(text, calls, raw)

    def _append_assistant(self, messages: list, turn: ModelTurn) -> None:
        messages.append(turn.raw or {"role": "assistant", "content": turn.text})

    def _append_tool_results(self, messages: list, results: list[tuple[ToolCall, ToolResult]]) -> None:
        if self.tool_mode == "native":
            for c, r in results:
                messages.append({"role": "tool", "tool_call_id": c.id, "content": r.text if r.ok else f"错误：{r.text}"})
        else:
            body = "\n".join(f"<tool_response>\n{c.name}: {r.text if r.ok else '错误：' + r.text}\n</tool_response>" for c, r in results)
            messages.append({"role": "user", "content": body})

    def _append_user(self, messages: list, text: str) -> None:
        messages.append({"role": "user", "content": text})
