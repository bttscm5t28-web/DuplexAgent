"""Chinese (+ a little English) word lists and text heuristics for turn decisions.

Stdlib only. Used by the `hybrid` family of turn detectors:

- `is_backchannel(text)`: "嗯/对对对/原来如此/哈哈…" — keep talking.
- `is_backchannel_prefix(text)`: text could still grow into a backchannel ("原来" → "原来如此").
- `is_stop_command(text)`: "停/等一下/别说了/不对…" — stop at once, even if very short.
- `text_eot_prob(text)`: rough P(the user finished the turn) from the words alone; used to
  stretch the silence timeout after "我想……" and shrink it after "…吗".
- `is_echo(text, agent_text)`: the "user" text is just the agent's own voice leaking back.
"""

from __future__ import annotations

import re

BACKCHANNELS = [
    # acknowledgement
    "嗯", "恩", "嗯哼", "嗯嗯", "唔", "啊", "哦", "噢", "喔", "哎", "诶", "欸", "哇",
    "对", "对的", "对呀", "对啊", "是", "是的", "是啊", "是呀", "好", "好的", "好啊", "行", "可以", "没错",
    "明白", "明白了", "了解", "懂了", "知道了", "我知道了", "我明白", "收到", "ok", "okay", "嗯好",
    # reactions
    "原来如此", "这样啊", "这样", "是吗", "真的吗", "真的啊", "真的", "哈哈", "呵呵", "嘿嘿", "厉害", "有道理", "确实",
    "然后呢", "然后", "嗯然后呢", "接着说",
    "uh-huh", "uhhuh", "yeah", "yes", "mm", "mhm", "right", "sure", "i see", "wow",
]

# Said while the agent talks, these mean "stop" no matter how short they are.
STOP_COMMANDS = [
    "停", "停一下", "停停", "暂停", "等一下", "等等", "等会", "打住", "别说了", "不要说了", "闭嘴", "安静",
    "不对", "不是", "错了", "不用了", "算了", "够了", "好了好了", "行了行了", "慢着",
    "wait", "stop", "hold on", "no no",
]

# A turn ending in these is almost surely unfinished.
_INCOMPLETE_TAIL = [
    # fillers
    "就是", "就是说", "然后", "然后呢就是", "呃", "额",
    # conjunctions / connectives
    "因为", "所以", "但是", "可是", "而且", "还有", "或者", "还是", "如果", "要是", "虽然", "不过", "以及", "并且", "然而",
    "和", "跟", "与", "及", "或", "比如", "比如说", "包括",
    # prepositions / coverbs
    "把", "被", "给", "从", "向", "往", "比", "关于", "对于", "为了", "除了",
    # modal / desire verbs that need a complement
    "想", "想要", "要", "打算", "准备", "需要", "希望", "能不能", "可不可以", "应该", "觉得", "认为",
    "帮我", "请", "请你", "麻烦", "麻烦你", "叫", "是", "问", "问一下",
    # quantifiers / determiners waiting for a noun
    "一个", "一张", "一份", "一些", "几个", "一只", "一杯", "一台", "每个", "所有", "这些", "那些", "哪个",
    "我的", "你的",
]
_INCOMPLETE_TAIL = sorted(set(_INCOMPLETE_TAIL), key=len, reverse=True)
_SHORT_FILLERS = ["那个", "这个", "嗯", "那"]  # unfinished only when that is (nearly) all that was said
_NEGATIONS = ("不", "没", "别")

_SUBORDINATE_HEADS = ["因为", "如果", "虽然", "要是", "假如", "只要", "除非", "既然", "等到", "当"]
_MAIN_CLAUSE_MARKS = ["所以", "那么", "那就", "就", "但", "可是", "也", "还是", "才", "，", ","]

_COMPLETE_TAIL = ["吗", "呢", "吧", "嘛", "么", "啦", "了", "呀", "哦", "喽", "咯", "谢谢", "再见", "你好", "好的", "可以吗", "行吗", "对吧"]
_QUESTION_WORDS = ["什么", "怎么样", "怎么", "为什么", "几点", "几号", "多少", "哪里", "哪儿", "哪个", "哪些", "谁", "多久", "多长", "几"]
_REQUEST_HEADS = ["帮我", "给我", "请", "麻烦", "我想", "我要", "我想要", "能不能", "可以帮"]
_DIGITS = set("零〇一二三四五六七八九十0123456789幺两")
_NUMBER_CONTEXT = ["号", "电话", "手机", "号码", "验证码", "卡号", "密码", "身份证", "地址"]

_PUNCT = re.compile(r"[\s,.!?;:，。！？；：、…~\-\"'“”‘’（）()「」『』《》]+")
_END_PUNCT = re.compile(r"[。！？!?]\s*$")


def norm(text: str) -> str:
    return _PUNCT.sub("", text.lower())


def _strip_all(text: str, words: list[str]) -> str:
    for w in sorted(words, key=len, reverse=True):
        text = text.replace(w, "")
    return text


def is_backchannel(text: str, extra: list[str] | None = None) -> bool:
    t = norm(text)
    if not t:
        return False
    return _strip_all(t, BACKCHANNELS + (extra or [])) == ""


def is_backchannel_prefix(text: str) -> bool:
    """`text` is a strict prefix of some backchannel (so it might still become one)."""
    t = norm(text)
    return bool(t) and any(b.startswith(t) and b != t for b in BACKCHANNELS)


def is_stop_command(text: str, extra: list[str] | None = None) -> bool:
    """Utterance starts with a stop command (after optional fillers like "嗯/哎")."""
    t = norm(text)
    t = re.sub(r"^(嗯|哎|诶|欸|啊|呃|额|喂)+", "", t)
    return any(t.startswith(norm(c)) for c in STOP_COMMANDS + (extra or []))


def is_echo(text: str, agent_text: str, min_chars: int = 4) -> bool:
    """User text that is a verbatim slice of what the agent is saying = speaker leak."""
    t, a = norm(text), norm(agent_text)
    return len(t) >= min_chars and t in a


def text_eot_prob(text: str) -> float:
    """Heuristic P(turn complete | words so far). 0.05 = clearly unfinished, 0.95 = clearly done."""
    raw = text.strip()
    t = norm(raw)
    if not t:
        return 0.5
    if _END_PUNCT.search(raw):  # real ASRs often punctuate finals
        base_punct = 0.85
    else:
        base_punct = None
    if t in {norm(c) for c in STOP_COMMANDS}:
        return 0.9  # a bare "等一下" / "停" is a complete turn
    if is_backchannel(t) or t in ("你好", "谢谢", "再见", "拜拜"):
        return 0.9
    for w in _INCOMPLETE_TAIL:
        if t.endswith(w):
            if t[: -len(w)].endswith(_NEGATIONS):  # "我不要" / "不是" are complete
                break
            if w in ("我的", "你的") and len(t) > 2 and t[:-2].endswith("是"):  # "这是我的"
                break
            return 0.08 if len(w) >= 2 or len(t) <= 3 else 0.15
    if t in _SHORT_FILLERS or (len(t) > 1 and t.endswith(("嗯", "呃", "额"))):
        return 0.1  # "那个……" / "我想嗯……"
    if t.endswith(("就是那个", "然后那个", "就是这个", "然后这个")):
        return 0.08
    # digits being dictated: unfinished until the number looks long enough
    tail_digits = len(t) - len(t.rstrip("".join(_DIGITS)))
    if tail_digits and any(c in t for c in _NUMBER_CONTEXT):
        return 0.1 if tail_digits < 11 else 0.7
    # dangling subordinate clause: "因为明天要下雨" without a main clause
    for h in _SUBORDINATE_HEADS:
        if t.startswith(h) and not any(m in t[len(h):] for m in _MAIN_CLAUSE_MARKS):
            return 0.15
    if t.endswith("的") and len(t) >= 4 and "是" not in t and any(q in t for q in ("一个", "一张", "一份", "一点", "一些", "一杯", "要")):
        return 0.35  # "我想要一个大一点的……" — the noun usually follows; "我是想问上海的" is done
    if any(t.endswith(w) for w in _COMPLETE_TAIL):
        return 0.92
    if any(q in t for q in _QUESTION_WORDS):
        return 0.88
    if base_punct is not None:
        return base_punct
    if any(t.startswith(h) for h in _REQUEST_HEADS) and len(t) <= 4:
        return 0.3  # "帮我看看" — the object usually follows
    return 0.7
