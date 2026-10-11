"""Rule-routed slow brain: parses the request and calls tools directly, no LLM.

Covers the frequent, well-structured requests (arithmetic incl. spoken Chinese operators, dates,
unit conversion) in milliseconds and exactly. Anything it does not understand is reported as
"unsupported" so `hybrid` can hand it to an LLM agent instead.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator

from duplexagent.core.interfaces import SlowBrain
from duplexagent.core.registry import create, register
from duplexagent.core.types import TaskRecord, TaskUpdate
from duplexagent.plugins.slow.toolbox import ToolBox, calc, fmt_num

_CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000, "万": 10_000, "亿": 100_000_000}


def cn2num(s: str) -> int:
    total, section, num = 0, 0, 0
    for ch in s:
        if ch in _CN_DIGIT:
            num = _CN_DIGIT[ch]
        elif ch in ("十", "百", "千"):
            section += (num or 1) * _CN_UNIT[ch]
            num = 0
        elif ch in ("万", "亿"):
            total = (total + section + num) * _CN_UNIT[ch]
            section = num = 0
    return total + section + num


_NOT_NUMBER_AFTER = set("下点些共起样般直定旦切致齐同")


def replace_cn_numbers(s: str) -> str:
    def sub(m: re.Match) -> str:
        w = m.group()
        nxt = s[m.end(): m.end() + 1]
        if re.fullmatch(r"[万亿千百]+", w) or (w in ("一", "两") and nxt in _NOT_NUMBER_AFTER):
            return w  # 一下 / 一共 / 两样 … are words, not numbers
        return str(cn2num(w))

    return re.sub(r"[零〇一二两三四五六七八九十百千万亿]{1,12}", sub, s)


def apply_amendments(query: str, amendments: list[str]) -> str:
    """Apply 'replace X with Y' style amendments to the query text; append the rest."""
    q = query
    for a in amendments:
        m = re.search(r"(?:把|将)?\s*(.+?)\s*(?:改成|改为|换成|变成|改到)\s*(.+?)[。.！!]?$", a.strip())
        old = m.group(1).strip() if m else ""
        pat = rf"(?<![\d.]){re.escape(old)}(?![\d.])" if re.fullmatch(r"\d+(?:\.\d+)?", old) else re.escape(old)
        if m and old and re.search(pat, q):
            q = re.sub(pat, m.group(2).strip(), q)
        else:
            q += "，" + a
    return q


_OPS = [(r"乘以|乘上|乘|×|x(?=\s*\d)|X(?=\s*\d)", "*"), (r"除以|÷", "/"), (r"加上|加|＋", "+"), (r"减去|减|－", "-"),
        (r"的平方根|开平方", "**0.5"), (r"的平方", "**2"), (r"的立方", "**3"), (r"的\s*(\d+)\s*次方", r"**\1"), (r"（", "("), (r"）", ")")]


_FILLER = re.compile(r"帮我|帮忙|请你|请|麻烦|计算一下|计算|算一下|算算|算|一下|等于|结果|是多少|多少|是|几|呢|吗|啊|的值|值|再|然后|[？?。.!！，,：:\s]")


def extract_math(q: str, strict: bool = True) -> str | None:
    """Pull an arithmetic expression out of a (spoken) request.

    strict: only succeed when the expression covers the whole request (apart from filler words),
    so word problems ("先打八折再减 50…") are left to an LLM instead of being half-parsed.
    """
    s = replace_cn_numbers(q)
    s = re.sub(r"保留\s*\d+\s*位小数", "", s)
    s = re.sub(r"百分之\s*(\d+(?:\.\d+)?)", r"\1%", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*的\s*(\d+(?:\.\d+)?)\s*%", r"(\1*\2/100)", s)
    for a, b in _OPS:
        s = re.sub(a, b, s)
    s = re.sub(r"([\+\-\*/])\s*(?:再|然后)\s*", r"\1", s)
    s = re.sub(r"(?:再|然后)\s*([\+\-\*/])", r"\1", s)
    s = re.sub(r"(\d)\s*%", r"(\1/100)", s)
    cands = re.findall(r"[\d\.\s\+\-\*/\(\)%]+", s)
    cands = [c.strip() for c in cands if re.search(r"\d", c) and re.search(r"[\+\-\*/]", c)]
    if not cands:
        return None
    expr = max(cands, key=len)
    if strict and _FILLER.sub("", s.replace(expr, "", 1)):
        return None
    expr = re.sub(r"^[\*/\+\s]+|[\+\-\*/\s]+$", "", expr)
    return expr or None


def decimals_wanted(q: str) -> int | None:
    m = re.search(r"保留\s*([\d一二两三四五])\s*位小数", q)
    return None if not m else int(m.group(1)) if m.group(1).isdigit() else cn2num(m.group(1))


def speak_expr(expr: str) -> str:
    s = re.sub(r"\*\*\s*0\.5", "的平方根", expr)
    s = re.sub(r"\*\*\s*(\d+)", r"的\1次方", s)
    s = s.replace("*", "乘").replace("/", "除以").replace("+", "加").replace("-", "减")
    return re.sub(r"\s+", "", s)


_DATE = r"(\d{4})\s*[年\-/.]\s*(\d{1,2})\s*[月\-/.]\s*(\d{1,2})\s*[日号]?"


class RuleSolver:
    def __init__(self, toolbox: ToolBox):
        self.tb = toolbox

    async def solve(self, q: str) -> tuple[list[tuple[str, dict]], str] | None:
        """Return (tool calls made, spoken answer) or None if not understood."""
        qn = replace_cn_numbers(q)
        dates = re.findall(_DATE, qn)
        calls: list[tuple[str, dict]] = []

        async def use(name: str, **args) -> str:
            calls.append((name, args))
            r = await self.tb.call(name, args)
            if not r.ok:
                raise ValueError(r.text)
            return r.text

        try:
            if len(dates) >= 2 and re.search(r"多少天|几天|相差|相隔|间隔|过多久", qn):
                a, b = ("-".join(d) for d in dates[:2])
                out = await use("date_calc", start=a, end=b)
                days = re.search(r"相差 (-?\d+) 天", out).group(1)
                return calls, f"相差 {days} 天。" + ("" if "相差" in q else "")
            if len(dates) >= 1:
                m = re.search(r"(?:后|之后|以后|过)\s*(\d+)\s*天", qn) or re.search(r"(\d+)\s*天(?:后|之后|以后)", qn)
                mb = re.search(r"(\d+)\s*天(?:前|之前|以前)", qn)
                d = "-".join(dates[0])
                if m or mb:
                    n = int((m or mb).group(1)) * (1 if m else -1)
                    out = await use("date_calc", start=d, add_days=n)
                    return calls, out.split("是 ", 1)[-1] + "。"
                if re.search(r"星期|周几|礼拜|第几天|闰年", qn):
                    out = await use("date_calc", start=d)
                    return calls, out + "。"
            if re.search(r"今天|现在|今日|几点|星期几|几号", qn) and not re.search(r"\d", qn.replace("几", "")):
                out = await use("now")
                return calls, out + "。"
            m = re.search(r"(-?\d+(?:\.\d+)?)\s*([a-zA-Z°℃℉/²³0-9]+|[一-鿿]{1,4}?)\s*(?:等于|是|合|换算成|换成|转成|折合|相当于|有|约)\s*(?:多少|几)\s*([a-zA-Z°℃℉/²³0-9]+|[一-鿿]{1,4})", qn)
            if m:
                frm, to = m.group(2), re.sub(r"[？?。]$", "", m.group(3))
                if frm.endswith("的"):
                    frm = frm[:-1]
                out = await use("unit_convert", value=float(m.group(1)), from_unit=frm, to_unit=to)
                nd = decimals_wanted(q)
                if nd is not None:
                    v = float(out.split("= ", 1)[1].split()[0])
                    return calls, f"{m.group(1)} {frm} 约等于 {v:.{nd}f} {to}。"
                return calls, out + "。"
            expr = extract_math(q)
            if expr:
                out = await use("calculator", expression=expr)
                v = out.rsplit("=", 1)[-1].strip()
                nd = decimals_wanted(q)
                if nd is not None:
                    v = f"{float(v):.{nd}f}"
                return calls, f"{speak_expr(expr)}等于 {v}。"
        except (ValueError, AttributeError, ZeroDivisionError, OverflowError):
            return None
        return None


@register("slow", "router")
class RouterSlowBrain(SlowBrain):
    """No-LLM baseline: deterministic parsing + tools. Unknown requests fail fast."""

    def __init__(self, min_delay_s: float = 0.0, timezone: str = "Asia/Shanghai", fixed_now: str | None = None):
        self.min_delay_s = min_delay_s
        self.toolbox = ToolBox(timezone=timezone, fixed_now=fixed_now)
        self.solver = RuleSolver(self.toolbox)
        self.traces: dict[str, list[dict]] = {}

    async def run(self, task: TaskRecord) -> AsyncIterator[TaskUpdate]:
        yield TaskUpdate("progress", "开始处理")
        if self.min_delay_s:
            await asyncio.sleep(self.min_delay_s)
        q = apply_amendments(task.query, task.amendments)
        got = await self.solver.solve(q)
        self.traces[task.id] = [{"tool": n, "args": a, "ok": True} for n, a in (got[0] if got else [])]
        if got is None:
            yield TaskUpdate("error", "这个问题超出了规则引擎的能力")
            return
        calls, answer = got
        for name, args in calls:
            yield TaskUpdate("progress", f"已{self.toolbox.label(name, args)}")
        if task.amendments:
            yield TaskUpdate("progress", f"已按补充「{'；'.join(task.amendments)}」调整")
        yield TaskUpdate("result", answer)


@register("slow", "hybrid")
class HybridSlowBrain(SlowBrain):
    """Rule router first (exact, ~ms); anything it cannot parse goes to an LLM agent (`inner`)."""

    def __init__(self, inner: dict | None = None, timezone: str = "Asia/Shanghai", fixed_now: str | None = None, **inner_kw):
        self.router = RouterSlowBrain(timezone=timezone, fixed_now=fixed_now)
        # Extra keys (e.g. the default config's `delay_s`) configure the inner brain.
        self.inner_cfg = dict(inner) if inner else {"impl": "mock", **inner_kw}
        self.inner = None
        self.traces: dict[str, list[dict]] = {}

    async def start(self) -> None:
        self.inner = create("slow", self.inner_cfg)
        await self.inner.start()

    async def close(self) -> None:
        if self.inner is not None:
            await self.inner.close()

    @property
    def details(self) -> dict[str, str]:
        return getattr(self.inner, "details", {})

    async def run(self, task: TaskRecord) -> AsyncIterator[TaskUpdate]:
        got = await self.router.solver.solve(apply_amendments(task.query, task.amendments))
        if got is not None and not task.amendments:
            calls, answer = got
            self.traces[task.id] = [{"tool": n, "args": a, "ok": True} for n, a in calls]
            yield TaskUpdate("progress", "开始处理")
            for name, args in calls:
                yield TaskUpdate("progress", f"已{self.router.toolbox.label(name, args)}")
            yield TaskUpdate("result", answer)
            return
        async for u in self.inner.run(task):
            yield u
        self.traces[task.id] = getattr(self.inner, "traces", {}).get(task.id, [])
