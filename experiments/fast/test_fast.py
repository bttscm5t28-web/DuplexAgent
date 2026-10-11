"""快脑实验插件的单元测试：python -m pytest -q experiments/fast/test_fast.py"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from duplexagent.core.registry import create  # noqa: E402
from duplexagent.core.types import AmendTask, BrainContext, CancelTask, Delegate, Message, Say, TaskRecord, TaskStatus  # noqa: E402
from duplexagent.plugins.fast.common import PhraseChunker, cn_to_number, conflicts, parse_head, route, spoken_result  # noqa: E402


def ctx(text, tasks=(), history=()):
    return BrainContext(trigger="user_turn", history=[*history, Message("user", text)], user_text=text, tasks=list(tasks))


def running(q, tid="t1"):
    return TaskRecord(query=q, id=tid)


def test_cn_numbers():
    assert cn_to_number("一百二十八") == 128
    assert cn_to_number("二点五") == 2.5
    assert cn_to_number("一万二") == 10002 or cn_to_number("一万二") == 12000  # colloquial, either is fine
    assert cn_to_number("3847") == 3847


def test_route_basic():
    assert route(ctx("你好")).kind == "chat"
    assert route(ctx("帮我查一下明天北京的天气")).kind == "delegate"
    assert route(ctx("三加五等于几")).kind == "chat"
    assert route(ctx("3847 乘以 296 等于多少")).kind == "delegate"
    assert route(ctx("我刚查了一下午资料头都大了")).kind == "chat"
    assert route(ctx("算了不说这个了")).kind == "chat"  # no running task -> not a cancel


def test_route_with_tasks():
    tasks = [running("查询明天北京的天气"), running("五千元预算配一台游戏电脑", "t2")]
    assert route(ctx("不用查了", tasks)).kind == "cancel"
    r = route(ctx("电脑那个不要了", tasks))
    assert r.kind == "cancel" and r.task.id == "t2"
    assert route(ctx("好了吗", tasks)).kind == "status"
    assert route(ctx("对了 显卡要 N 卡", tasks)).task.id == "t2"
    assert route(ctx("再帮我查下上海明天的天气", tasks)).kind == "delegate"


def test_phrase_chunker():
    c = PhraseChunker(first_chars=4, max_chars=8)
    out = []
    for piece in ["你好", "呀，今天", "天气不错，适合出去走走。"]:
        out += c.feed(piece)
    out += c.flush()
    assert "".join(out) == "你好呀，今天天气不错，适合出去走走。"
    assert len(out[0]) <= 4


def test_protocol_head():
    assert parse_head("[委") is None
    assert parse_head("[委派] 好的") == ("委派", None, "好的")
    assert parse_head("[取消 t2] 好") == ("取消", "t2", "好")
    assert parse_head("你好呀，")[0] == "说"


def test_fusion_helpers():
    assert conflicts("大概一百万左右吧", "3847 * 296 等于 1138712")
    assert not conflicts("大约一千出头", "23 * 47 等于 1081")
    assert "\n" not in spoken_result("1. 西湖\n2. 灵隐寺\n3. 西溪")


def _run(fast, c):
    async def go():
        await fast.start()
        return [a async for a in fast.respond(c)]

    return asyncio.run(go())


def test_router_plugin_actions():
    fast = create("fast", {"impl": "router"})
    acts = _run(fast, ctx("帮我写一封请假邮件"))
    assert isinstance(acts[0], Delegate) and any(isinstance(a, Say) for a in acts)
    acts = _run(fast, ctx("取消吧", [running("订一张周五去上海的高铁票")]))
    assert isinstance(acts[0], CancelTask)
    acts = _run(fast, ctx("要二等座 下午出发的", [running("订一张周五去上海的高铁票")]))
    assert isinstance(acts[0], AmendTask)
    t = running("计算 3847 乘以 296")
    said = "".join(a.text for a in _run(fast, ctx("结果是多少", [t])) if isinstance(a, Say))
    assert "1138712" not in said and "还在处理" in said


def test_router_fusion_corrects():
    fast = create("fast", {"impl": "router"})
    t = TaskRecord(query="计算 3847 乘以 296", id="t1", status=TaskStatus.DONE, result="3847 * 296 等于 1138712")
    hist = [Message("user", "帮我算一下 3847 乘以 296"), Message("assistant", "大概一百万左右吧，我让后台精确算一下。"),
            Message("task", t.result, {"task_id": "t1", "query": t.query})]
    said = "".join(a.text for a in _run(fast, BrainContext("task_result", hist, tasks=[t], deliver=[t])) if isinstance(a, Say))
    assert "更正" in said and "1138712" in said


class FakeBackend:
    """Replays a scripted completion token by token (no model)."""

    def __init__(self, text):
        self.text = text
        self.prompts = []

    async def start(self):
        pass

    async def stream(self, msgs, prefill, max_tokens, temperature):
        self.prompts.append((msgs, prefill))
        out = self.text[len(prefill.split("</think>")[-1].strip()):] if self.text.startswith(prefill.strip()) else self.text
        for i in range(0, len(out), 2):
            yield out[i : i + 2]


def _llm(text, mode="llm", **kw):
    from duplexagent.plugins.fast.llamacpp import LlamaCppFastBrain

    fast = LlamaCppFastBrain(mode=mode, warmup=False, **kw)
    fast.backend = FakeBackend(text)
    return fast


def test_llm_protocol_delegate():
    acts = _run(_llm("[委派] 好，我让后台查一下。[任务] 查询深圳明天的天气"), ctx("深圳明天下雨吗"))
    said = "".join(a.text for a in acts if isinstance(a, Say))
    d = [a for a in acts if isinstance(a, Delegate)]
    assert said == "好，我让后台查一下。" and d and d[0].query == "查询深圳明天的天气"


def test_llm_protocol_cancel_and_amend():
    t = running("订一张周五去上海的高铁票")
    acts = _run(_llm("[取消 t1] 好的，已经取消了。"), ctx("票不订了", [t]))
    assert isinstance(acts[0], CancelTask) and acts[0].task_id == "t1"
    acts = _run(_llm("[补充 t9] 好的。[信息] 要二等座"), ctx("要二等座", [t]))  # unknown id -> best match
    am = [a for a in acts if isinstance(a, AmendTask)]
    assert am and am[0].task_id == "t1" and am[0].info == "要二等座"


def test_llm_protocol_say_and_no_protocol():
    acts = _run(_llm("[说] 你好呀，**今天**怎么样？"), ctx("你好"))
    assert "".join(a.text for a in acts) == "你好呀，今天怎么样？"
    acts = _run(_llm("你好呀"), ctx("你好"))  # model ignored the protocol
    assert "".join(a.text for a in acts) == "你好呀"


def test_hybrid_router_shortcut_and_fusion_lead():
    fast = _llm("[说] 不该被调用", mode="hybrid")
    acts = _run(fast, ctx("帮我查一下明天北京的天气"))
    assert isinstance(acts[0], Delegate) and not fast.backend.prompts
    t = TaskRecord(query="计算 3847 乘以 296", id="t1", status=TaskStatus.DONE, result="3847 * 296 等于 1138712")
    hist = [Message("user", "帮我算一下 3847 乘以 296"), Message("assistant", "大概一百万左右吧。"),
            Message("task", t.result, {"task_id": "t1", "query": t.query})]
    fast = _llm("[说] 更正一下我刚才说的，关于3847 乘以 296，结果是 1138712。", mode="hybrid")
    said = "".join(a.text for a in _run(fast, BrainContext("task_result", hist, tasks=[t], deliver=[t])) if isinstance(a, Say))
    assert said.startswith("更正一下") and "1138712" in said
