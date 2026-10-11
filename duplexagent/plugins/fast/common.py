"""Shared pieces for the experimental fast brains (router / llamacpp / hybrid).

- `route()`        rule router: cancel / amend / status / delegate / clarify / chat, ~0.1 ms, no model.
- `ack_for()` etc. spoken templates (receipts, status, cancel, fusion fallback) that never guess results.
- `PhraseChunker`  re-chunks a token stream into short phrases so TTS gets natural units early.
- `parse_protocol` the text protocol small LLMs use instead of tool calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from duplexagent.core.types import BrainContext, Message, TaskRecord, TaskStatus

# ---- text helpers ---------------------------------------------------------------------

_PUNCT = re.compile(r"[\s,.!?;:，。！？；：、…\"'“”‘’（）()\[\]【】~～-]+")


def norm(text: str) -> str:
    return _PUNCT.sub("", text.lower())


_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000}
_NUM_RE = re.compile(r"\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万亿点]+")


def cn_to_number(s: str) -> float | None:
    """'一百二十八' -> 128, '二点五' -> 2.5, '3847' -> 3847. None if not a number."""
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        return float(s)
    if "点" in s:
        a, _, b = s.partition("点")
        ia = cn_to_number(a) if a else 0
        if ia is None or not b or any(c not in _CN_DIGITS for c in b):
            return None
        return ia + float("0." + "".join(str(_CN_DIGITS[c]) for c in b))
    total, section, digit, last_unit = 0, 0, None, 0
    for c in s:
        if c in _CN_DIGITS:
            digit = _CN_DIGITS[c]
        elif c in _CN_UNITS:
            last_unit = _CN_UNITS[c]
            u = _CN_UNITS[c]
            if u >= 10000:
                total = (total + section + (digit or 0)) * u
                section = 0
            else:
                section += (digit if digit is not None else 1) * u
            digit = None
        else:
            return None
    if digit is not None and last_unit >= 100 and s[-2] in _CN_UNITS:
        digit *= last_unit // 10  # colloquial: 一百三 = 130, 一千二 = 1200, 一万五 = 15000
    return total + section + (digit or 0)


def numbers_in(text: str) -> list[float]:
    out = []
    for m in _NUM_RE.findall(text):
        v = cn_to_number(m)
        if v is not None:
            out.append(v)
    return out


def running_tasks(ctx: BrainContext) -> list[TaskRecord]:
    return [t for t in ctx.tasks if t.status == TaskStatus.RUNNING]


def _bigrams(s: str) -> set[str]:
    s = norm(s)
    return {s[i : i + 2] for i in range(len(s) - 1)}


_TASK_STOP = {"帮我", "一下", "查询", "查一", "计算", "一个", "的天", "搜索", "最近", "关于"}


def pick_task(text: str, tasks: list[TaskRecord]) -> TaskRecord | None:
    """The running task the user is most likely talking about (bigram overlap, else the latest)."""
    if not tasks:
        return None
    q = _bigrams(text) - _TASK_STOP
    scored = [(len(q & (_bigrams(t.query) - _TASK_STOP)), i, t) for i, t in enumerate(tasks)]
    best = max(scored, key=lambda x: (x[0], x[1]))
    return best[2]


# ---- rule router ----------------------------------------------------------------------


@dataclass
class Route:
    kind: str  # "cancel" | "amend" | "status" | "delegate" | "clarify" | "chat"
    confident: bool = True
    topic: str = ""  # delegate topic: realtime / personal / calc / plan / write
    task: TaskRecord | None = None
    why: list[str] = field(default_factory=list)


CANCEL_RE = re.compile(
    r"不用(了|查|算|写|订|弄|搞|管)|不需要了|不要了|算了|取消|别(查|弄|算|订|写|搞|管|找)了|先别|不想(知道|要|查)了|不订了|"
    r"停{2,}|我自己(写|来|弄|查|订)|never ?mind|cancel|forget it|不必了"
)
STATUS_RE = re.compile(
    r"好了(吗|没|嘛|么)|好了没有|(查|算|写|订|配|弄|规划|搞)(到|好|完|上)(了)?(吗|没|嘛|么)|怎么样了|进度|还要多久|要多久|多久(能|才)?好|"
    r"(结果|答案)(是|呢|出来|怎么样|多少)|还没好|还没(查|算|写|弄)完|(说|讲)了(些)?(啥|什么)|查得怎么样|有(结果|消息)了(吗|没)|完成了(吗|没)|"
    r"先(大概|简单)?(说|讲)(说|个|一下)?|大概(说|讲)|在(查|算|弄)了吗|how long|is it done|any update"
)
AMEND_RE = re.compile(
    r"^(哦)?(对了|还有|另外|补充|顺便看看|顺带)|我是说|不是.{1,8}是|^不对|是.{1,10}不是|改成|换成|要(求)?.{0,10}(的|一点|点)$|"
    r"只要|别安排|不要太|语气|就说|是请|最好|预算|^要|记得"
)
NEW_REQ_RE = re.compile(r"(再|另外|顺便|也)(帮我|给我)?(查|搜|算|计算|订|定|写|规划|看看)|再帮我|另外帮我|(帮我|给我)(定|订|设|发|叫|打)|提醒")

REALTIME_RE = re.compile(
    r"天气|下雨|下雪|气温|带伞|温度怎么样|新闻|股价|股票|指数|大盘|汇率|油价|金价|币.{0,3}(价|多少)|比分|比赛|谁赢|赛程|打谁|上映|堵不堵|路况|"
    r"附近|最近的.{0,6}(在哪|店|站)|最新|现在.{0,6}(价|多少钱|几点)|今天.{0,6}(价|多少|人多)|实时|营业|快递|weather|news|stock|price"
)
PERSONAL_RE = re.compile(
    r"我的(航班|日程|快递|邮件|订单|账单|会议|行程)|我(明天|今天|下周|后天).{0,4}(有什么|有啥)(安排|事|会)|提醒我|闹钟|定个|发(个)?(微信|消息|短信|邮件)|"
    r"打(个)?电话|(开|关)(了|掉)?.{0,4}(灯|空调|电视|窗帘)|把.{0,6}(灯|空调).{0,2}(开|关)|叫(个)?车|收藏|花了多少钱|新邮件|日历|改到|"
    r"订(一|个|张|间|份)|买(一|个|张|份)|下单|remind me|set an? alarm|call "
)
CALC_OP_RE = re.compile(r"乘|除|加|减|[×*/+÷^]|平方|次方|开方|根号|根是|利率|利息|复利|月供|换算|兑|多少磅|多少斤|多少公里|多少英里|摄氏|华氏|平摊|均摊|多少天|百分之|%|小费|个税|五险一金|质数|总和|之和")
PLAN_RE = re.compile(
    r"规划|(帮我|给我|做个|做一个|定个|制定).{0,8}(计划|方案|安排|清单)|行程|路线|攻略|清单|方案|配置|怎么配|配(一|个)台|推荐.{0,6}(款|个|家)|"
    r"(买|选)哪(款|个|家)|选一款|比较|对比|哪个(更|好)|调研|分析|走势|前景|怎么走|怎么去|预算.{0,8}(买|配|选)|学习计划|安排一下|compare|plan "
)
WRITE_RE = re.compile(
    r"写(一|个|篇|封|首|份|段|点|下)|帮我写|起草|作文|周报|日报|简历|合同|主持词|演讲稿|文案|代码|脚本|sql|python|java|"
    r"总结|概括|摘要|翻译(成|一下|这)|翻成|润色|改得|改写|write |translate|summar"
)
ACTION_PREFIX_RE = re.compile(r"(帮我|给我|替我|你帮我|麻烦你|请你?)(查|搜|找|看看|算|订|定|买|写|发|打|叫|规划|安排|总结|翻译|把|设|弄)")
SHORT_WORD_TRANSLATION = re.compile(r"(用|的)?(英文|英语|日语|日文|韩语|法语)(怎么说|是什么)|是什么意思")
CHITCHAT_GUARD = re.compile(r"我(刚|今天|已经)?(查|算|计划|规划|写)了|好累|心情|郁闷|难过|无聊|你觉得|你喜欢|你会|哈哈|我计划(明天|今天|今晚)")


def _calc_hard(text: str) -> bool:
    if not CALC_OP_RE.search(text):
        return False
    nums = numbers_in(text)
    if re.search(r"利率|利息|复利|月供|个税|五险一金|质数|多少天|华氏|摄氏|小费|汇率|兑|换(多少)?(人民币|美元)", text):
        return True
    if re.search(r"多少(磅|斤|公里|英里)|换算", text):
        return any(n not in (0, 1) for n in nums)
    if re.search(r"平方根|开方|根号|次方|平方是", text):
        return any(n > 10 for n in nums) or len(nums) >= 2
    if not nums:
        return False
    big = [n for n in nums if n > 12 or n != int(n)]
    if re.search(r"乘|除|[×*/÷]|平摊|均摊|百分之|%", text):
        return len(big) >= 1 and (max(nums) > 20 or len(big) >= 2)
    return max(nums) >= 100 or len(nums) >= 3  # add / subtract


def _incomplete(text: str) -> bool:
    n = norm(text)
    if len(n) <= 3:
        return False
    return bool(re.search(r"(帮我|你帮我|我想问一下|我想问)(订个|查一下那个|算算|那个)?$|(那个|这个|一下|个)$", n)) and len(n) <= 9


def route(ctx: BrainContext) -> Route:
    """Rule router. `confident=False` means "probably chat, but a model may know better"."""
    text = ctx.user_text.strip()
    n = norm(text)
    running = running_tasks(ctx)
    if running:
        if CANCEL_RE.search(text) and not NEW_REQ_RE.search(text):
            return Route("cancel", task=pick_task(text, running), why=["cancel_re"])
        if STATUS_RE.search(text) and not NEW_REQ_RE.search(text):
            return Route("status", task=pick_task(text, running), why=["status_re"])
        if AMEND_RE.search(text) and not NEW_REQ_RE.search(text) and len(n) <= 24:
            return Route("amend", task=pick_task(text, running), why=["amend_re"])
    if CHITCHAT_GUARD.search(text) and not ACTION_PREFIX_RE.search(text):
        return Route("chat", why=["chitchat_guard"])
    if _incomplete(text):
        return Route("clarify", why=["incomplete"])
    if SHORT_WORD_TRANSLATION.search(text) and len(n) <= 14:
        return Route("chat", why=["short_translation"])
    for topic, ok in (
        ("personal", PERSONAL_RE.search(text)),
        ("realtime", REALTIME_RE.search(text)),
        ("calc", _calc_hard(text)),
        ("write", WRITE_RE.search(text)),
        ("plan", PLAN_RE.search(text)),
    ):
        if ok:
            return Route("delegate", topic=topic, why=[topic])
    if ACTION_PREFIX_RE.search(text):
        return Route("delegate", topic="other", confident=False, why=["action_prefix"])
    return Route("chat", confident=False)


# ---- spoken templates -----------------------------------------------------------------

ACKS = {
    "realtime": "好，我让后台查一下最新的，稍等。",
    "personal": "好的，我这就让后台去办。",
    "calc": "好，我让后台精确算一下，马上告诉你。",
    "plan": "好，这个我让后台好好规划一下，你可以先聊点别的。",
    "write": "没问题，我让后台来写，写好了告诉你。",
    "other": "好的，我让后台去处理，你可以先聊点别的。",
}


def ack_for(topic: str) -> str:
    return ACKS.get(topic, ACKS["other"])


def status_text(task: TaskRecord | None, running: list[TaskRecord]) -> str:
    if task is None:
        return "现在后台没有在处理的事情。"
    p = task.progress[-1] if task.progress else "刚开始"
    extra = f"另外还有 {len(running) - 1} 件在处理。" if len(running) > 1 else ""
    return f"「{short_query(task.query)}」还在处理，目前{p}，结果出来我马上告诉你。{extra}"


def short_query(q: str, limit: int = 18) -> str:
    q = re.sub(r"^(帮我|请|麻烦)", "", q.strip())
    return q if len(q) <= limit else q[:limit] + "…"


CORRECTION_WORDS = ("更正", "不对", "说错", "不准", "纠正", "其实", "应该是")
_HEDGE = re.compile(r"大概|好像|应该|差不多|左右|估计|可能|大约|似乎")


def earlier_claim(history: list[Message], task: TaskRecord) -> str | None:
    """The assistant's own hedged guess about this task, if it made one (for corrections)."""
    keys = _bigrams(task.query) - _TASK_STOP
    for i, m in enumerate(history):
        if m.role != "user" or not (keys & _bigrams(m.content)):
            continue
        nxt = history[i + 1] if i + 1 < len(history) else None
        if nxt is not None and nxt.role == "assistant" and _HEDGE.search(nxt.content):
            return nxt.content
    return None


def conflicts(claim: str, result: str, query: str = "") -> bool | None:
    """Does the hedged claim contradict the result? None = can't tell from numbers (ask a model).

    Numbers that merely restate the question are ignored; a claim number within 10 % of a
    result number counts as agreeing.
    """
    asked = numbers_in(query)
    cn = [x for x in numbers_in(claim) if not any(abs(x - q) < 1e-9 for q in asked)]
    if not cn:
        return None
    rn = numbers_in(result)
    return not any(abs(c - r) <= 0.1 * max(abs(r), 1) for c in cn for r in rn)


def spoken_result(result: str, limit: int = 60) -> str:
    """Make a slow-brain result speakable: drop markdown/list markers, cut at a sentence boundary."""
    s = re.sub(r"[#*`>|]+", "", result)
    s = re.sub(r"\n+\s*(\d+[.、)]|[-•])?\s*", "，", s).strip("，")
    s = re.sub(r"(^|[；;，,。])\s*\d+[.、)]\s*", r"\1", s)
    if len(s) <= limit:
        return s
    cut = max(s.rfind(p, 0, limit) for p in "。；;！？")
    return s[: cut + 1] if cut > limit // 3 else s[:limit] + "……"


def fusion_text(ctx: BrainContext, task: TaskRecord) -> str:
    q = short_query(task.query)
    if task.status != TaskStatus.DONE:
        return f"抱歉，「{q}」没能完成：{spoken_result(task.result or '原因不明', 40)}。要不要再试一次？"
    prefix = ""
    claim = earlier_claim(ctx.history, task)
    if claim and conflicts(claim, task.result or "", task.query):  # None (unsure) -> don't correct
        prefix = "更正一下我刚才说的，"
    return f"{prefix}你刚才问的「{q}」有结果了：{spoken_result(task.result or '')}"


# ---- streaming helpers ----------------------------------------------------------------


class PhraseChunker:
    """Buffer streamed text and release it at phrase boundaries (or after `max_chars`).

    The first phrase is released early (`first_chars`) so TTS can start sooner.
    """

    BOUNDARY = "，。！？；：,.!?;:\n…"

    def __init__(self, first_chars: int = 4, max_chars: int = 16) -> None:
        self.buf = ""
        self.first = True
        self.first_chars = first_chars
        self.max_chars = max_chars

    def feed(self, piece: str) -> list[str]:
        self.buf += piece
        out = []
        while self.buf:
            idx = next((i for i, c in enumerate(self.buf) if c in self.BOUNDARY), -1)
            limit = self.first_chars if self.first else self.max_chars
            if idx >= 0 and (idx + 1 >= 2 or not self.first):
                out.append(self.buf[: idx + 1])
                self.buf = self.buf[idx + 1 :]
            elif len(self.buf) >= limit:
                out.append(self.buf[:limit])
                self.buf = self.buf[limit:]
            else:
                break
            self.first = False
        return out

    def flush(self) -> list[str]:
        out, self.buf = ([self.buf] if self.buf else []), ""
        return out


# ---- text protocol for small LLMs -------------------------------------------------------

TAG_RE = re.compile(r"^\s*\[(说|委派|取消|补充)\s*(t\d+)?\]\s*")
TASK_SEP = "[任务]"
INFO_SEP = "[信息]"


def parse_head(buf: str) -> tuple[str, str | None, str] | None:
    """Return (tag, task_id, rest) once the leading tag is complete; None while undecided."""
    m = TAG_RE.match(buf)
    if m:
        return m.group(1), m.group(2), buf[m.end() :]
    if len(buf) > 12 or (buf.strip() and not buf.lstrip().startswith("[")):
        return "说", None, buf.lstrip("[ ")  # model ignored the protocol: treat as speech
    return None


def strip_markdown(s: str) -> str:
    return re.sub(r"[*#`>]+", "", s)


_SYMBOLS = {"→": "到", "->": "到", "~": "到", "～": "到", "★": "评分", "☆": "评分", "&": "和", "|": "，", "=": "等于"}


def speakable(phrase: str) -> str:
    """Normalize one streamed phrase for TTS: no newlines, list markers, markdown or symbols."""
    s = re.sub(r"^\s*(\d{1,2}[.、)](?!\d)|[-•*]|#+)\s*", "", phrase)
    s = re.sub(r"[*#`>]+", "", s)
    for k, v in _SYMBOLS.items():
        s = s.replace(k, v)
    s = re.sub(r"\s*\n+\s*", "，", s)
    s = re.sub(r"([，。；：！？,;:])[，,]+", r"\1", s)
    return s
