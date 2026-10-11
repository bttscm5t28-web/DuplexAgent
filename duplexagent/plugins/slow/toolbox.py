"""Tools for the slow brain, shared by every backend (OpenAI-compatible, Claude, MCP, rule router).

Each tool has a JSON-schema signature, an async implementation returning text, and a short
spoken label used for progress updates ("正在用计算器算 23*47") that the fast brain can relay.
Tools never raise: failures come back as ToolResult(ok=False, text=...) so the model can recover.
"""

from __future__ import annotations

import ast
import asyncio
import calendar
import datetime as dt
import ipaddress
import json
import math
import operator
import re
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlparse

from duplexagent.plugins.slow.sandbox import PythonSandbox

WEEKDAYS = "一二三四五六日"


@dataclass
class ToolResult:
    ok: bool
    text: str


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON schema of the arguments object
    fn: Callable[..., Awaitable[str]]
    label: Callable[[dict], str] = lambda a: "调用工具"

    def openai_spec(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}

    def anthropic_spec(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.parameters}


def _schema(props: dict[str, tuple[str, str]], required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": {k: {"type": t, "description": d} for k, (t, d) in props.items()},
        "required": required if required is not None else list(props),
    }


# ---- calculator ----------------------------------------------------------------------

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {k: getattr(math, k) for k in ("sqrt", "log", "log10", "log2", "exp", "sin", "cos", "tan", "asin", "acos", "atan",
                                        "floor", "ceil", "factorial", "comb", "perm", "gcd", "lcm", "radians", "degrees", "hypot")}
_FUNCS.update(abs=abs, round=round, min=min, max=max, int=int, float=float)
_CONST = {"pi": math.pi, "e": math.e}


def normalize_math(expr: str) -> str:
    s = expr.strip()
    for a, b in (("×", "*"), ("✕", "*"), ("÷", "/"), ("（", "("), ("）", ")"), ("，", ","), ("^", "**"), ("−", "-"), ("π", "pi")):
        s = s.replace(a, b)
    s = re.sub(r"(\d)\s*%", r"(\1/100)", s)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)  # thousands separators
    return s


def calc(expr: str) -> float | int:
    def ev(n: ast.AST):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            l, r = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and (abs(r) > 10000 or (abs(l) > 1 and abs(r) * math.log10(abs(l) + 1) > 4000)):
                raise ValueError("指数太大")
            return _BIN[type(n.op)](l, r)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UN:
            return _UN[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Name) and n.id in _CONST:
            return _CONST[n.id]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            args = [ev(a) for a in n.args]
            if n.func.id == "factorial" and args and args[0] > 3000:
                raise ValueError("阶乘太大")
            return _FUNCS[n.func.id](*args)
        raise ValueError(f"不支持的表达式：{ast.dump(n)[:60]}")

    return ev(ast.parse(normalize_math(expr), mode="eval").body)


def fmt_num(v: float | int) -> str:
    if isinstance(v, float):
        if v.is_integer() and abs(v) < 1e16:
            return str(int(v))
        return f"{v:.10g}"
    return str(v)


async def _calculator(expression: str) -> str:
    return f"{expression} = {fmt_num(calc(expression))}"


# ---- date / time ---------------------------------------------------------------------


def _parse_date(s: str) -> dt.date:
    s = s.strip()
    m = re.match(r"(\d{4})\s*[-/年.]\s*(\d{1,2})\s*[-/月.]\s*(\d{1,2})", s)
    if not m:
        raise ValueError(f"无法解析日期：{s}（请用 YYYY-MM-DD）")
    return dt.date(int(m[1]), int(m[2]), int(m[3]))


def _date_str(d: dt.date) -> str:
    return f"{d.year}年{d.month}月{d.day}日（星期{WEEKDAYS[d.weekday()]}）"


class Clock:
    def __init__(self, timezone: str = "Asia/Shanghai", fixed_now: str | None = None):
        from zoneinfo import ZoneInfo

        self.tz = ZoneInfo(timezone)
        self.fixed = dt.datetime.fromisoformat(fixed_now).replace(tzinfo=self.tz) if fixed_now else None

    def now(self) -> dt.datetime:
        return self.fixed or dt.datetime.now(self.tz)


# ---- units ---------------------------------------------------------------------------

_UNITS: dict[str, tuple[str, float]] = {}
for dim, table in {
    "length": {"m": 1, "米": 1, "km": 1e3, "公里": 1e3, "千米": 1e3, "cm": 0.01, "厘米": 0.01, "mm": 1e-3, "毫米": 1e-3,
               "mi": 1609.344, "mile": 1609.344, "英里": 1609.344, "ft": 0.3048, "foot": 0.3048, "英尺": 0.3048,
               "in": 0.0254, "inch": 0.0254, "英寸": 0.0254, "yd": 0.9144, "码": 0.9144, "里": 500, "海里": 1852, "nmi": 1852},
    "mass": {"kg": 1, "公斤": 1, "千克": 1, "g": 1e-3, "克": 1e-3, "mg": 1e-6, "毫克": 1e-6, "t": 1000, "吨": 1000,
             "lb": 0.45359237, "磅": 0.45359237, "oz": 0.028349523125, "盎司": 0.028349523125, "斤": 0.5, "两": 0.05},
    "volume": {"l": 1, "升": 1, "ml": 1e-3, "毫升": 1e-3, "m3": 1000, "立方米": 1000, "gal": 3.785411784, "加仑": 3.785411784},
    "area": {"m2": 1, "平方米": 1, "km2": 1e6, "平方公里": 1e6, "公顷": 1e4, "ha": 1e4, "亩": 10000 / 15, "acre": 4046.8564224, "英亩": 4046.8564224},
    "speed": {"m/s": 1, "米每秒": 1, "km/s": 1000, "公里每秒": 1000, "千米每秒": 1000, "km/h": 1 / 3.6, "千米每小时": 1 / 3.6, "公里每小时": 1 / 3.6, "mph": 0.44704, "节": 1852 / 3600, "knot": 1852 / 3600},
    "data": {"b": 1, "byte": 1, "字节": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3, "tb": 1024**4},
    "time": {"s": 1, "秒": 1, "min": 60, "分钟": 60, "h": 3600, "小时": 3600, "天": 86400, "day": 86400, "周": 604800, "week": 604800},
}.items():
    for k, f in table.items():
        _UNITS[k.lower()] = (dim, f)
_TEMPS = {"c": "c", "°c": "c", "℃": "c", "摄氏度": "c", "f": "f", "°f": "f", "℉": "f", "华氏度": "f", "k": "k", "开尔文": "k"}


def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    fu, tu = from_unit.strip().lower(), to_unit.strip().lower()
    if fu in _TEMPS and tu in _TEMPS:
        c = {"c": value, "f": (value - 32) * 5 / 9, "k": value - 273.15}[_TEMPS[fu]]
        return {"c": c, "f": c * 9 / 5 + 32, "k": c + 273.15}[_TEMPS[tu]]
    if fu not in _UNITS or tu not in _UNITS:
        raise ValueError(f"不认识的单位：{from_unit if fu not in _UNITS else to_unit}")
    (d1, f1), (d2, f2) = _UNITS[fu], _UNITS[tu]
    if d1 != d2:
        raise ValueError(f"{from_unit} 和 {to_unit} 不是同一类单位")
    return value * f1 / f2


# ---- web (optional) ------------------------------------------------------------------


def _public_host(url: str) -> bool:
    """SSRF guard: only http(s) to hosts that resolve to public addresses."""
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        return False
    try:
        infos = socket.getaddrinfo(u.hostname, None)
    except OSError:
        return True  # let the HTTP client (or proxy) fail; unresolvable locally is common behind proxies
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript|svg|head)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?s)<[^>]+>", " ", html)
    import html as _h

    return re.sub(r"\s+", " ", _h.unescape(html)).strip()


# ---- the toolbox ---------------------------------------------------------------------


@dataclass
class ToolStats:
    calls: int = 0
    ok: int = 0
    by_tool: dict[str, list[int]] = field(default_factory=dict)  # name -> [calls, ok]

    def add(self, name: str, ok: bool) -> None:
        self.calls += 1
        self.ok += ok
        c = self.by_tool.setdefault(name, [0, 0])
        c[0] += 1
        c[1] += ok


class ToolBox:
    """Named collection of tools. `enabled` picks a subset; `web=True` adds network tools."""

    def __init__(self, enabled: list[str] | None = None, web: bool = False, timezone: str = "Asia/Shanghai",
                 fixed_now: str | None = None, python_timeout_s: float = 10.0, user_agent: str = "DuplexAgent-slow/0.1"):
        self.clock = Clock(timezone, fixed_now)
        self.sandbox = PythonSandbox(timeout_s=python_timeout_s)
        self.user_agent = user_agent
        self.tools: dict[str, Tool] = {}
        self.stats = ToolStats()
        for t in self._builtin():
            self.tools[t.name] = t
        if web:
            for t in self._web():
                self.tools[t.name] = t
        if enabled is not None:
            self.tools = {k: v for k, v in self.tools.items() if k in enabled}
        self.extra_instructions: list[str] = []

    def add(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def specs(self, style: str = "openai") -> list[dict]:
        return [t.openai_spec() if style == "openai" else t.anthropic_spec() for t in self.tools.values()]

    def label(self, name: str, args: dict) -> str:
        t = self.tools.get(name)
        try:
            return t.label(args) if t else f"调用 {name}"
        except Exception:
            return f"调用 {name}"

    async def call(self, name: str, args: dict | str | None) -> ToolResult:
        t = self.tools.get(name)
        if t is None:
            res = ToolResult(False, f"没有名为 {name} 的工具，可用：{', '.join(self.tools)}")
        else:
            try:
                if isinstance(args, str):
                    args = json.loads(args) if args.strip() else {}
                args = args or {}
                allowed = set(t.parameters.get("properties", {}))
                res = ToolResult(True, await t.fn(**{k: v for k, v in args.items() if k in allowed}))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                res = ToolResult(False, f"{type(e).__name__}: {e}")
            if res.ok and res.text.startswith(("超时：", "超出 CPU")):
                res.ok = False
        self.stats.add(name, res.ok)
        return res

    # -- built-in tools ---------------------------------------------------------------

    def _builtin(self) -> list[Tool]:
        async def python(code: str) -> str:
            ok, out = await self.sandbox.run(code)
            if not ok:
                raise RuntimeError(out)
            return out

        async def now() -> str:
            n = self.clock.now()
            return f"现在是 {_date_str(n.date())} {n:%H:%M}（{self.clock.tz.key}）"

        async def date_calc(start: str, end: str = "", add_days: int = 0) -> str:
            a = _parse_date(start)
            if end:
                b = _parse_date(end)
                return f"从 {_date_str(a)} 到 {_date_str(b)} 相差 {(b - a).days} 天"
            if add_days:
                b = a + dt.timedelta(days=int(add_days))
                return f"{_date_str(a)} 加 {int(add_days)} 天是 {_date_str(b)}"
            leap = "闰年" if calendar.isleap(a.year) else "平年"
            return f"{_date_str(a)}，是当年第 {a.timetuple().tm_yday} 天，{a.year} 年是{leap}"

        async def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
            v = convert_units(float(value), from_unit, to_unit)
            return f"{fmt_num(float(value))} {from_unit} = {fmt_num(round(v, 6))} {to_unit}"

        return [
            Tool("calculator", "精确计算数学表达式。支持 + - * / // % **、括号、sqrt/log/sin/factorial/comb 等函数和 pi、e。",
                 _schema({"expression": ("string", "Python 风格的数学表达式，例如 (12+8)*3 或 sqrt(2)")}), _calculator,
                 lambda a: f"用计算器算 {a.get('expression', '')}"),
            Tool("python", "在隔离沙箱里运行 Python 代码（只有标准库，无网络、无文件写入，限时 10 秒）。用 print 输出结果；适合循环、枚举、统计、复杂计算。",
                 _schema({"code": ("string", "要运行的 Python 代码")}), python, lambda a: "写程序计算"),
            Tool("now", "获取当前日期、时间和星期。", _schema({}), now, lambda a: "查看当前日期时间"),
            Tool("date_calc", "日期计算：给 start 和 end 返回相差天数；给 start 和 add_days 返回加减天数后的日期；只给 start 返回星期几、当年第几天、是否闰年。日期格式 YYYY-MM-DD。",
                 _schema({"start": ("string", "起始日期 YYYY-MM-DD"), "end": ("string", "结束日期 YYYY-MM-DD，可选"), "add_days": ("integer", "要加的天数，可为负，可选")}, ["start"]),
                 date_calc, lambda a: f"推算日期 {a.get('start', '')}"),
            Tool("unit_convert", "单位换算：长度、重量（含斤、两）、体积、面积（含亩）、速度、数据大小、时间、温度（C/F/K）。",
                 _schema({"value": ("number", "数值"), "from_unit": ("string", "原单位，如 km、英里、斤、F"), "to_unit": ("string", "目标单位")}),
                 unit_convert, lambda a: f"把 {a.get('value', '')}{a.get('from_unit', '')} 换算成{a.get('to_unit', '')}"),
        ]

    def _web(self) -> list[Tool]:
        async def _get(url: str, params: dict | None = None) -> str:
            import httpx  # optional dependency

            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True, headers={"User-Agent": self.user_agent}) as c:
                r = await c.get(url, params=params)
                r.raise_for_status()
                return r.text

        async def web_search(query: str) -> str:
            # Wikipedia search API: keyless, stable, multilingual. Swap for a search API with a key in production.
            lang = "zh" if re.search(r"[一-鿿]", query) else "en"
            data = json.loads(await _get(f"https://{lang}.wikipedia.org/w/api.php", {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 5}))
            hits = data.get("query", {}).get("search", [])
            if not hits:
                return "没有搜索结果"
            return "\n".join(f"- {h['title']}: {html_to_text(h.get('snippet', ''))} (https://{lang}.wikipedia.org/wiki/{quote(h['title'])})" for h in hits)

        async def web_fetch(url: str) -> str:
            if not _public_host(url):
                raise PermissionError("只允许访问公网 http(s) 地址")
            text = html_to_text(await _get(url))
            return text[:4000] + ("…（截断）" if len(text) > 4000 else "")

        return [
            Tool("web_search", "联网搜索（百科），返回标题、摘要和链接。", _schema({"query": ("string", "搜索词")}), web_search, lambda a: f"搜索「{a.get('query', '')}」"),
            Tool("web_fetch", "抓取网页正文（前 4000 字）。", _schema({"url": ("string", "http(s) 地址")}), web_fetch, lambda a: "打开网页查看内容"),
        ]
