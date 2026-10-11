"""Small-LLM fast brain (local llama.cpp in-process, or any OpenAI-compatible server).

Instead of JSON tool calls (unreliable on 0.5B–4B models and slow to stream), the model
answers in a one-tag text protocol, and the tag decides the action:

    [说] 直接回答……
    [委派] 给用户的一句回执 [任务] 自包含的任务描述
    [取消 t1] 回执
    [补充 t1] 回执 [信息] 补充内容

The receipt comes *before* the task description, so speech starts as soon as the tag is
decided. The static system prompt (with few-shot examples) is a fixed prefix, so the
llama.cpp KV cache makes it free after the first turn; the per-turn task table is placed
right before the last user message to keep the history prefix cacheable.

mode = "llm"     the model makes every decision
mode = "hybrid"  the rule router (common.route) takes confident cancel / status / amend /
                 delegate / clarify decisions with template speech (≈0 ms); the model handles
                 chat, ambiguous turns and result fusion.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from collections.abc import AsyncIterator

from duplexagent.core.interfaces import FastBrain
from duplexagent.core.registry import register
from duplexagent.core.types import AmendTask, BrainContext, CancelTask, Delegate, FastAction, Say, TaskStatus
from duplexagent.plugins.fast.common import (
    INFO_SEP,
    TASK_SEP,
    PhraseChunker,
    conflicts,
    earlier_claim,
    fusion_text,
    parse_head,
    pick_task,
    route,
    running_tasks,
    norm,
    short_query,
    speakable,
    spoken_result,
    strip_markdown,
)
from duplexagent.plugins.fast.router import delegate_context, structured_actions

SYSTEM_PROMPT = """你是语音助手的“快脑”，你说的每个字都会被直接念给用户听。后台还有一个“慢脑”，能查资料、用工具、精确计算、做规划、写长内容，但要花几秒到几分钟。

每次回复必须以一个标记开头：
[说] 你直接回答。口语化、简短（一两句），不用列表、Markdown、表情。
[委派] 一句简短回执 [任务] 自包含的任务描述（慢脑看不到对话，要写清楚时间地点数字等）
[取消 t编号] 一句回执 —— 用户不要某个正在运行的任务了
[补充 t编号] 一句回执 [信息] 补充内容 —— 用户给正在运行的任务加条件或改条件

什么时候用[委派]：需要实时或外部信息（天气、新闻、股价、汇率、比赛、附近的店）；需要用户的私人数据或执行动作（日程、闹钟、提醒、发消息、订票、开关电器）；多位数乘除、利率、换算等精确计算；规划、比较、调研；写邮件、文章、代码、长翻译。
什么时候用[说]：闲聊、安慰、观点、稳定的常识、一步心算、简短词语翻译；问正在运行的任务进度（只说还在处理，绝对不要猜结果）；用户话没说完时追问。
已经交给后台的问题，在结果回来之前，绝不自己猜答案。

示例：
用户：今天好累啊
助手：[说] 辛苦啦，早点休息，要不要听点轻松的？
用户：帮我查一下深圳明天会不会下雨
助手：[委派] 好，我让后台查一下深圳明天的天气。[任务] 查询深圳明天的天气，是否下雨
用户：37乘以58等于多少
助手：[委派] 好，我让后台精确算一下。[任务] 计算 37 乘以 58
用户：二加二等于几
助手：[说] 等于四。
用户：法国的首都是哪里
助手：[说] 是巴黎。
用户：给我写个周报
助手：[委派] 没问题，我让后台来写，好了告诉你。[任务] 帮用户写一份本周工作周报
用户：（后台任务 t1 订机票 进行中）机票不用订了
助手：[取消 t1] 好的，订机票已经取消了。
用户：（后台任务 t1 写周报 进行中）周报里加上项目延期的事
助手：[补充 t1] 好，我告诉后台加上。[信息] 周报里要提到项目延期
用户：（后台任务 t1 查股价 进行中）查到了吗
助手：[说] 还在查，结果出来我马上告诉你。
用户：帮我那个
助手：[说] 你想让我帮你做什么呢？"""

FUSION_INSTRUCTION = (
    "后台任务刚完成，你已经开口说了：“{lead}”。现在直接接着往下说，用口语讲出最关键的结果，"
    "不要重复开头，不要列表，不要念编号和符号，总共{limit}字以内。{extra}"
)


def task_table(ctx: BrainContext) -> str:
    running = running_tasks(ctx)
    if not running:
        return "当前没有正在运行的后台任务。"
    lines = [f"[{t.id}] {t.query}（进度：{t.progress[-1] if t.progress else '刚开始'}）" for t in running]
    return "当前正在运行的后台任务：\n" + "\n".join(lines)


def topic_of(query: str) -> str:
    q = re.sub(r"^(帮用户|帮我|请|给用户)?(查询|查一下|查|计算|搜索|搜|撰写|写|规划|总结|翻译)(一下)?", "", query.strip())
    q = q.replace("用户的", "你的").replace("用户", "你")
    return short_query(q or query, 20)


def fusion_lead(ctx: BrainContext, idx: int = 0, total: int = 1) -> str:
    """Template opening for one delivered task (ctx.deliver[0]); the model continues from it."""
    t = ctx.deliver[0]
    corr = "更正一下我刚才说的，" if _needs_correction(ctx, t) else ""
    topic = topic_of(t.query)
    if total > 1:
        head = f"你之前问的{total}件事都有结果了。" if idx == 0 else "另外，"
        return f"{head}{corr}关于{topic}，"
    return f"{corr}关于{topic}，"


def failure_text(t) -> str:
    return f"抱歉，{topic_of(t.query)}没办成，原因是{spoken_result(t.result or '不明', 30).rstrip('。')}。"


def _needs_correction(ctx: BrainContext, t) -> bool:
    claim = earlier_claim(ctx.history, t)
    return bool(claim) and t.status == TaskStatus.DONE and conflicts(claim, t.result or "")


def build_messages(ctx: BrainContext, system_prompt: str, fusion_limit: int = 50, max_turns: int = 12, lead: str = "") -> list[dict]:
    msgs = [{"role": "system", "content": system_prompt}]
    start = max(0, len(ctx.history) - max_turns)
    start -= start % 6  # slide the window in steps so the cached prefix survives most turns
    hist = ctx.history[start:]
    last_user = max((i for i, m in enumerate(hist) if m.role == "user"), default=-1) if ctx.trigger == "user_turn" else -1
    for i, m in enumerate(hist):
        if m.role == "task":
            status = "失败" if m.meta.get("status") == "failed" else "完成"
            msgs.append({"role": "system", "content": f"后台任务 {m.meta.get('task_id')}（问题：{m.meta.get('query')}）{status}，结果：{m.content}"})
        elif m.role == "assistant":
            cut = "……（被用户打断）" if m.meta.get("interrupted") else ""
            msgs.append({"role": "assistant", "content": f"[说] {m.content}{cut}"})
        else:
            msgs.append({"role": "user", "content": m.content})
        if i == last_user:  # after the user turn: everything before it stays a cacheable prefix
            msgs.append({"role": "system", "content": task_table(ctx)})
    if ctx.trigger == "task_result":
        msgs.append({"role": "system", "content": FUSION_INSTRUCTION.format(lead=lead or fusion_lead(ctx), limit=fusion_limit, extra="")})
    return msgs


def chatml(msgs: list[dict]) -> str:
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs) + "<|im_start|>assistant\n"


# ---- backends ---------------------------------------------------------------------------

_MODELS: dict[tuple, object] = {}
_MODELS_LOCK = threading.Lock()


class LlamaCppBackend:
    """In-process llama.cpp. Generation runs in a worker thread; tokens are bridged to asyncio."""

    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = 4, n_batch: int = 512, **_):
        self.key = (model_path, n_ctx, n_threads, n_batch)
        self.lock = threading.Lock()  # one generation at a time per model

    async def start(self) -> None:
        def load():
            with _MODELS_LOCK:
                if self.key not in _MODELS:
                    from llama_cpp import Llama  # optional dependency

                    path, n_ctx, n_threads, n_batch = self.key
                    _MODELS[self.key] = (Llama(model_path=path, n_ctx=n_ctx, n_threads=n_threads, n_batch=n_batch, verbose=False), threading.Lock())
                return _MODELS[self.key]

        self.llm, self.lock = await asyncio.to_thread(load)

    async def stream(self, msgs: list[dict], prefill: str, max_tokens: int, temperature: float) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()
        prompt = chatml(msgs) + prefill

        def work():
            try:
                with self.lock:
                    for out in self.llm.create_completion(prompt, max_tokens=max_tokens, temperature=temperature, top_p=0.9,
                                                          stop=["<|im_end|>", "<|im_start|>", "<|endoftext|>"], stream=True):
                        if stop.is_set():
                            break
                        loop.call_soon_threadsafe(q.put_nowait, out["choices"][0]["text"])
            except Exception as e:  # surfaced to the caller
                loop.call_soon_threadsafe(q.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        fut = loop.run_in_executor(None, work)
        try:
            while (item := await q.get()) is not None:
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop.set()  # cancelled (barge-in) or done: the worker stops at the next token
            if not fut.done():
                fut.add_done_callback(lambda f: f.exception())


class OpenAIBackend:
    """Any OpenAI-compatible chat server (llama.cpp server, vLLM, Ollama, cloud). No prefill."""

    def __init__(self, base_url: str, model: str, api_key_env: str = "OPENAI_API_KEY", **params):
        self.cfg = dict(base_url=base_url, model=model, api_key_env=api_key_env, **params)

    async def start(self) -> None:
        from duplexagent.core.llm import ChatClient

        self.client = ChatClient(**self.cfg)

    async def close(self) -> None:
        await self.client.close()

    async def stream(self, msgs: list[dict], prefill: str, max_tokens: int, temperature: float) -> AsyncIterator[str]:
        self.client.params.update(max_tokens=max_tokens, temperature=temperature)
        async for delta in self.client.stream(msgs):
            if delta.get("content"):
                yield delta["content"]


# ---- the fast brain ------------------------------------------------------------------------


@register("fast", "llamacpp")
class LlamaCppFastBrain(FastBrain):
    def __init__(
        self,
        model_path: str = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        mode: str = "hybrid",
        backend: str = "llamacpp",
        system_prompt: str = SYSTEM_PROMPT,
        max_tokens: int = 80,
        temperature: float = 0.3,
        chunking: str = "phrase",
        no_think: bool = False,
        fusion: str = "llm",  # "llm" | "template"
        fusion_limit: int = 50,
        rewrite_query: bool = True,
        warmup: bool = True,
        llm_decides: bool = True,  # hybrid: may the model still delegate / cancel when the router says "chat"?
        **backend_kw,
    ) -> None:
        if mode not in ("llm", "hybrid"):
            raise ValueError("mode must be 'llm' or 'hybrid'")
        self.mode = mode
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.chunking = chunking
        self.no_think = no_think
        self.fusion = fusion
        self.fusion_limit = fusion_limit
        self.rewrite_query = rewrite_query
        self.warmup = warmup
        self.llm_decides = llm_decides
        if backend == "llamacpp":
            self.backend = LlamaCppBackend(model_path, **backend_kw)
        elif backend == "openai":
            self.backend = OpenAIBackend(**backend_kw)
        else:
            raise ValueError(f"unknown backend {backend!r}")
        self.stats: list[dict] = []

    async def start(self) -> None:
        await self.backend.start()
        if self.warmup:  # fill the KV cache with the static system prompt
            async for _ in self.backend.stream([{"role": "system", "content": self.system_prompt}], "", 1, 0.0):
                pass

    async def close(self) -> None:
        if hasattr(self.backend, "close"):
            await self.backend.close()

    async def prime(self, ctx: BrainContext) -> None:
        """Evaluate the prompt of the previous turns (as a live session would have cached)."""
        prev = BrainContext(trigger="user_turn", history=ctx.history[:-1], tasks=ctx.tasks)
        msgs = build_messages(prev, self.system_prompt)
        async for _ in self.backend.stream(msgs, "", 1, 0.0):
            pass

    def _prefill(self, head: str) -> str:
        return ("<think>\n\n</think>\n\n" if self.no_think else "") + head

    async def _speak(self, pieces: AsyncIterator[str]) -> AsyncIterator[Say]:
        if self.chunking == "token":
            async for p in pieces:
                yield Say(p)
            return
        ch = PhraseChunker()
        async for p in pieces:
            for out in ch.feed(p):
                if out := speakable(out):
                    yield Say(out)
        for out in ch.flush():
            if out := speakable(out):
                yield Say(out)

    async def respond(self, ctx: BrainContext) -> AsyncIterator[FastAction]:
        if ctx.trigger == "task_result":
            async for a in self._fuse(ctx):
                yield a
            return
        if self.mode == "hybrid":
            r = route(ctx)
            if r.confident and r.kind != "chat":
                actions, speech = structured_actions(r, ctx)
                for a in actions:
                    yield a
                yield Say(speech)
                return
            if not self.llm_decides:
                async for a in self._llm_turn(ctx, force="说"):
                    yield a
                return
        async for a in self._llm_turn(ctx):
            yield a

    async def _fuse(self, ctx: BrainContext) -> AsyncIterator[FastAction]:
        if self.fusion == "template":
            for t in ctx.deliver:
                yield Say(fusion_text(ctx, t))
            return
        total = len(ctx.deliver)
        for idx, t in enumerate(ctx.deliver):  # one task at a time: small models drop the second one otherwise
            if t.status != TaskStatus.DONE:
                yield Say(failure_text(t))
                continue
            sub = BrainContext(trigger="task_result", history=ctx.history, tasks=ctx.tasks, deliver=[t], turn_id=ctx.turn_id)
            lead = fusion_lead(sub, idx, total)
            yield Say(lead)  # spoken at once; the model continues from it (prefill) while it plays
            msgs = build_messages(sub, self.system_prompt, self.fusion_limit, lead=lead)
            prefill = self._prefill("[说] " + lead) if isinstance(self.backend, LlamaCppBackend) else ""
            gen = self.backend.stream(msgs, prefill, self.max_tokens, self.temperature)
            said = ""
            try:
                async for s in self._speak(_cap(_clean_speech(gen), self.fusion_limit)):
                    said += s.text
                    yield s
            finally:
                await gen.aclose()
            if len(norm(said)) < 2:  # model produced nothing usable: fall back to the result itself
                yield Say(spoken_result(t.result or "", self.fusion_limit))

    async def _llm_turn(self, ctx: BrainContext, force: str = "") -> AsyncIterator[FastAction]:
        msgs = build_messages(ctx, self.system_prompt)
        prefill = self._prefill(f"[{force}]" if force else "[")
        gen = self.backend.stream(msgs, prefill, self.max_tokens, self.temperature)
        buf = f"[{force}]" if force else ("[" if prefill.endswith("[") else "")
        if force and not isinstance(self.backend, LlamaCppBackend):
            buf = ""  # no prefill on chat APIs: the model writes the tag itself
        head = parse_head(buf) if buf.endswith("]") else None
        try:
            async for piece in (gen if head is None else _empty()):
                buf += piece
                head = parse_head(buf)
                if head:
                    break
        except BaseException:
            await gen.aclose()
            raise
        if head is None:
            head = ("说", None, buf.lstrip("[ "))  # short reply that never formed a tag
        tag, tid, rest = head
        if force:
            tag = force
        running = running_tasks(ctx)

        if tag in ("取消", "补充") and not running:
            tag = "说"  # nothing to cancel / amend: just speak
        task = None
        if tag in ("取消", "补充"):
            task = next((t for t in running if t.id == tid), None) or pick_task(ctx.user_text, running)
        if tag == "取消":
            yield CancelTask(task.id)

        sep = {"委派": TASK_SEP, "补充": INFO_SEP}.get(tag)
        tail: list[str] = []

        async def speech() -> AsyncIterator[str]:
            """Stream the spoken part; stash whatever follows the separator in `tail`."""
            async for p in _hold_brackets(_chain(rest, gen)):
                if tail:
                    tail.append(p)
                    continue
                if sep and sep in p:
                    before, _, after = p.partition(sep)
                    if before:
                        yield before
                    tail.append(after)
                    if tag == "委派" and not self.rewrite_query:
                        return
                    continue
                yield p

        try:
            async for s in self._speak(_clean_speech(speech())):
                yield s
        finally:
            await gen.aclose()  # stop the worker thread now (barge-in / early return), not at GC
        extra = "".join(tail).strip()
        if tag == "委派":
            query = extra if (self.rewrite_query and len(extra) >= 4) else ctx.user_text
            yield Delegate(query=query, context=delegate_context(ctx))
        elif tag == "补充" and task is not None:
            yield AmendTask(task.id, extra or ctx.user_text)


async def _empty() -> AsyncIterator[str]:
    return
    yield


async def _chain(first: str, rest: AsyncIterator[str]) -> AsyncIterator[str]:
    if first:
        yield first
    async for p in rest:
        yield p


async def _hold_brackets(pieces: AsyncIterator[str]) -> AsyncIterator[str]:
    """Merge pieces so a separator like "[任务]" is never split across two pieces."""
    buf = ""
    async for p in pieces:
        buf += p
        i = buf.rfind("[")
        if i >= 0 and "]" not in buf[i:]:
            if buf[:i]:
                yield buf[:i]
            buf = buf[i:]
            if len(buf) > 8:  # not a tag after all
                yield buf
                buf = ""
            continue
        yield buf
        buf = ""
    if buf:
        yield buf


async def _cap(pieces: AsyncIterator[str], limit: int) -> AsyncIterator[str]:
    """Stop at the first sentence end after `limit` chars (hard stop at 1.5x)."""
    n = 0
    async for p in pieces:
        if n >= limit:
            m = re.search(r"[。！？；!?;]", p)
            if m:
                yield p[: m.end()]
                return
        if n >= limit * 1.5:
            return
        n += len(p)
        yield p


async def _clean_speech(pieces: AsyncIterator[str]) -> AsyncIterator[str]:
    """Drop stray protocol tags / markdown from spoken text."""
    started = False
    async for p in pieces:
        p = re.sub(r"\[(说|委派|任务|信息|取消|补充)[^\]]*\]", "", p)
        p = strip_markdown(p)
        if not started:
            p = p.lstrip()
        if p:
            started = True
            yield p
