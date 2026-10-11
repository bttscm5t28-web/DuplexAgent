"""Chinese TTS text normalization that keeps the mapping back to the original text.

`units(text)` splits the input into alignment units. Each unit carries
  - `orig`: the exact slice of the input text (what goes into TTSChunk.text),
  - `spoken`: what the TTS model should read ("23" -> "二十三", "%" -> ...),
  - `weight`: expected duration in syllable-ish units (used to split audio by time),
  - `pause`: True for punctuation / spaces (no speech, possibly silence).

Concatenating `orig` always gives back the input, so alignment never loses text.
Stdlib only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DIGITS = "零一二三四五六七八九"
SMALL_UNITS = ["", "十", "百", "千"]
BIG_UNITS = ["", "万", "亿", "万亿"]

# Pause weights in syllables (a Chinese syllable is ~0.2 s at normal rate).
PAUSE_WEIGHT = {
    "，": 0.9, ",": 0.9, "、": 0.6, "；": 1.2, ";": 1.2, "：": 0.8, ":": 0.8,
    "。": 1.6, ".": 1.4, "！": 1.6, "!": 1.6, "？": 1.6, "?": 1.6, "…": 1.4,
    "—": 0.6, "\n": 1.4,
}
SENTENCE_END = set("。！？!?；;…\n")
CLAUSE_END = set("，,、：:—")
# Characters that are neither read nor paused on.
SILENT = set(" \t\r\"'“”‘’《》〈〉「」『』【】()（）[]{}<>*_`~^|")

SYMBOLS = {
    "+": "加", "=": "等于", "×": "乘", "÷": "除以", "&": "和", "@": "艾特", "#": "井号",
    "℃": "摄氏度", "°C": "摄氏度", "°": "度", "‰": "千分之", "→": "到", "～": "到", "~": "到",
}
UNITS = {
    "km/h": "公里每小时", "km": "公里", "kg": "公斤", "cm": "厘米", "mm": "毫米", "ml": "毫升", "mg": "毫克",
    "kb": "KB", "mb": "MB", "gb": "GB", "tb": "TB", "ms": "毫秒", "m²": "平方米", "㎡": "平方米",
    "m": "米", "g": "克", "l": "升", "s": "秒", "h": "小时", "w": "瓦", "kw": "千瓦",
}
CURRENCY = {"¥": "元", "￥": "元", "$": "美元", "€": "欧元", "£": "英镑"}

# Rough Chinese sound-alikes for letters, for models that cannot read Latin script.
ZH_LETTERS = {
    "a": "诶", "b": "比", "c": "西", "d": "迪", "e": "伊", "f": "艾弗", "g": "吉", "h": "艾尺", "i": "艾",
    "j": "杰", "k": "开", "l": "艾勒", "m": "艾姆", "n": "恩", "o": "欧", "p": "批", "q": "Q", "r": "阿",
    "s": "艾斯", "t": "提", "u": "优", "v": "维", "w": "达不溜", "x": "艾克斯", "y": "歪", "z": "贼德",
}
ZH_LETTERS["q"] = "丘"


@dataclass
class Unit:
    orig: str
    spoken: str
    weight: float
    pause: bool = False

    @property
    def sentence_end(self) -> bool:
        return self.pause and any(c in SENTENCE_END for c in self.orig)

    @property
    def clause_end(self) -> bool:
        return self.pause and any(c in CLAUSE_END or c in SENTENCE_END for c in self.orig)


# ---- numbers ---------------------------------------------------------------------------------


def _four(n: int, leading: bool) -> str:
    """0 < n < 10000 -> Chinese, e.g. 1081 -> 一千零八十一."""
    out, zero = "", False
    for i in range(3, -1, -1):
        d = n // 10**i % 10
        if d == 0:
            zero = bool(out)
            continue
        if zero or (not out and not leading and i < 3):
            out += "零"
        zero = False
        out += DIGITS[d] + SMALL_UNITS[i]
    return out


def int_to_zh(n: int) -> str:
    if n == 0:
        return "零"
    if n < 0:
        return "负" + int_to_zh(-n)
    groups, i = [], 0
    while n:
        groups.append((n % 10000, i))
        n //= 10000
        i += 1
    out = ""
    for k, (g, idx) in enumerate(reversed(groups)):
        if g == 0:
            continue
        leading = k == 0
        s = _four(g, leading)
        if not leading and g < 1000 and not out.endswith("零"):
            s = "零" + s.lstrip("零")
        out += s + BIG_UNITS[idx]
    if out.startswith("一十"):
        out = out[1:]
    return out


def digits_zh(s: str, yao: bool = False) -> str:
    return "".join(("幺" if yao and c == "1" else DIGITS[int(c)]) for c in s if c.isdigit())


def number_zh(s: str) -> str:
    """'3.5' -> 三点五, '1,000' -> 一千, '007' -> 零零七."""
    s = s.replace(",", "")
    neg = s.startswith("-")
    s = s.lstrip("-")
    if "." in s:
        a, b = s.split(".", 1)
        out = int_to_zh(int(a or "0")) + "点" + digits_zh(b)
    elif len(s) > 1 and s.startswith("0"):
        out = digits_zh(s)
    else:
        out = int_to_zh(int(s))
    return ("负" if neg else "") + out


def _liang(zh: str) -> str:
    """二 -> 两 before measure words (两个, 两点), but keep 十二/二十."""
    if zh == "二":
        return "两"
    if zh.startswith("二") and len(zh) > 1 and zh[1] in "千万亿":
        return "两" + zh[1:]
    return zh


# ---- tokenizer -------------------------------------------------------------------------------

_RULES: list[tuple[str, re.Pattern]] = [
    ("date", re.compile(r"(\d{2,4})[年/-](\d{1,2})[月/-](\d{1,2})[日号]?")),
    ("ym", re.compile(r"(\d{4})年(\d{1,2})月")),
    ("year", re.compile(r"(\d{4})年")),
    ("time", re.compile(r"(\d{1,2}):(\d{2})(?::(\d{2}))?")),
    ("phone", re.compile(r"(?<![\d.])(1[3-9]\d{9}|0\d{2,3}-\d{7,8}|400-?\d{3}-?\d{4})(?![\d.])")),
    ("range", re.compile(r"(\d+(?:\.\d+)?)(%?)\s*[-~～]\s*(\d+(?:\.\d+)?)(%?)(?=\s*[a-zA-Z°℃一-鿿，。,！？!?；;、]|\s*$)")),
    ("percent", re.compile(r"-?\d+(?:\.\d+)?%")),
    ("money", re.compile(r"([¥￥$€£])\s*(\d+(?:,\d{3})*(?:\.\d+)?)([万亿]?)")),
    ("unit", re.compile(r"(\d+(?:\.\d+)?)\s*(km/h|km|kg|cm|mm|ml|mg|kb|mb|gb|tb|ms|m²|㎡|kw|m|g|l|s|h|w)(?![a-zA-Z])", re.I)),
    ("fraction", re.compile(r"(\d+)/(\d+)")),
    ("number", re.compile(r"-?\d+(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")),
    ("word", re.compile(r"[A-Za-z][A-Za-z']*(?:[-.][A-Za-z0-9]+)*")),
]

_VOWELS = re.compile(r"[aeiouy]+", re.I)


def _english_weight(word: str) -> float:
    if word.isupper() or len(word) <= 2:
        return 0.8 * len(word)  # spelled out
    return max(1.0, 0.9 * len(_VOWELS.findall(word)))


class Normalizer:
    """Turns raw text into alignment units. `english`:
      "keep"    leave English words for the model (bilingual models),
      "spell"   spell acronyms letter by letter, keep other words,
      "zh"      spell every Latin letter with Chinese sound-alikes (Chinese-only models).
    """

    def __init__(self, english: str = "spell") -> None:
        self.english = english

    def units(self, text: str) -> list[Unit]:
        out: list[Unit] = []
        i, n = 0, len(text)
        while i < n:
            m, kind = None, None
            matched = False
            for kind, rx in _RULES:
                m = rx.match(text, i)
                if not m or m.end() == i:
                    continue
                spoken = self._render(kind, m, text)
                if spoken is not None:
                    self._emit_speech(out, m.group(0), spoken, kind)
                    i = m.end()
                    matched = True
                    break
            if matched:
                continue
            c = text[i]
            if c in PAUSE_WEIGHT or c in SILENT or c.isspace():
                j = i
                while j < n and (text[j] in PAUSE_WEIGHT or text[j] in SILENT or text[j].isspace()):
                    j += 1
                seg = text[i:j]
                w = max((PAUSE_WEIGHT.get(ch, 0.0) for ch in seg), default=0.0)
                spoken = "".join(ch for ch in seg if ch in PAUSE_WEIGHT and ch != "\n")
                # Normalize ASCII punctuation to Chinese so zh front-ends pause properly.
                spoken = spoken.translate(str.maketrans({",": "，", ".": "。", "!": "！", "?": "？", ";": "；", ":": "：", "…": "，", "—": "，"}))
                spoken = re.sub(r"([，。！？；：、])[，。！？；：、]+", r"\1", spoken)
                if " " in seg and not spoken and out and not out[-1].pause:
                    spoken = " "
                out.append(Unit(seg, spoken, w, pause=True))
                i = j
                continue
            if c in SYMBOLS:
                sym = SYMBOLS[c]
                if c == "#" and text.startswith("号", i + 1):
                    sym = "井"  # "#号键"
                self._emit_speech(out, c, sym, "symbol")
            elif c in CURRENCY:
                self._emit_speech(out, c, CURRENCY[c], "symbol")
            else:
                out.append(Unit(c, c, 1.0))
            i += 1
        return out

    def _emit_speech(self, out: list[Unit], orig: str, spoken: str, kind: str | None) -> None:
        if kind == "word":
            out.append(Unit(orig, spoken, _english_weight(orig)))
            return
        # one unit, weight by spoken syllables (Chinese chars) + letters
        w = sum(1.0 if "一" <= ch <= "鿿" else (0.8 if ch.isalnum() else 0.0) for ch in spoken)
        out.append(Unit(orig, spoken, max(w, 0.5)))

    def _render(self, kind: str | None, m: re.Match, text: str) -> str | None:
        g = m.groups()
        if kind == "date":
            y, mo, d = g
            if not (1 <= int(mo) <= 12 and 1 <= int(d) <= 31):
                return None
            return digits_zh(y) + "年" + int_to_zh(int(mo)) + "月" + int_to_zh(int(d)) + ("号" if m.group(0)[-1] == "号" else "日")
        if kind == "ym":
            return digits_zh(g[0]) + "年" + int_to_zh(int(g[1])) + "月"
        if kind == "year":
            return digits_zh(g[0]) + "年"
        if kind == "time":
            h, mi, s = g
            if int(h) > 24 or int(mi) > 59:
                return None
            out = _liang(int_to_zh(int(h))) + "点"
            out += ("" if mi == "00" and not s else ("零" + DIGITS[int(mi)] if mi.startswith("0") else int_to_zh(int(mi))) + "分")
            if s:
                out += int_to_zh(int(s)) + "秒"
            return out
        if kind == "phone":
            return digits_zh(m.group(0), yao=True)
        if kind == "range":
            a, pa, b, pb = g
            if (pa or pb) or text[m.end() : m.end() + 1] == "%":
                return "百分之" + number_zh(a) + "到百分之" + number_zh(b)
            if text[m.start() - 1 : m.start()].isdigit():
                return None
            if all(len(x) == 4 and x[:2] in ("19", "20") for x in (a, b)):
                return digits_zh(a) + "到" + digits_zh(b)  # 2023-2024 赛季
            return number_zh(a) + "到" + number_zh(b)
        if kind == "percent":
            return "百分之" + number_zh(m.group(0)[:-1])
        if kind == "money":
            sym, num, big = g
            word = CURRENCY[sym]
            if text.startswith(word[-1], m.end()) or text.startswith(word, m.end()):
                word = ""  # "¥3.5元"
            return number_zh(num) + big + word
        if kind == "unit":
            num, unit = g
            u = UNITS.get(unit.lower(), unit)
            if unit.lower() in ("m", "s", "h", "g", "l", "w") and _latin_follows(text, m.end()):
                return None
            return _liang(number_zh(num)) + u
        if kind == "fraction":
            a, b = g
            if int(b) == 0:
                return None
            return number_zh(b) + "分之" + number_zh(a)
        if kind == "number":
            s = m.group(0)
            raw = s.replace(",", "").lstrip("-")
            after_latin = m.start() > 0 and text[m.start() - 1].isascii() and text[m.start() - 1].isalpha()
            if (after_latin and len(raw) >= 3 and "." not in s) or len(raw) >= 5 and "," not in s and "." not in s and not _quantity_follows(text, m.end()):
                return digits_zh(s)  # IDs / codes / long digit strings: read digit by digit
            zh = number_zh(s)
            nxt = text[m.end() : m.end() + 1]
            if nxt and nxt in "个只条位张本次件点岁种名天年辆台家双块分" and text[m.start() - 1 : m.start()] != "第":
                zh = _liang(zh)
            return zh
        if kind == "word":
            return self._english(m.group(0))
        return None

    def _english(self, word: str) -> str:
        parts = re.split(r"[-.]", word)
        out = []
        for p in parts:
            if p.isdigit():
                out.append(number_zh(p))
            elif self.english == "zh":
                out.append("".join(ZH_LETTERS.get(c.lower(), c) for c in p if c.isalpha()))
            elif self.english == "spell" and p.isupper() and 1 < len(p) <= 5:
                out.append(" ".join(p))
            else:
                out.append(p)
        return " ".join(x for x in out if x)


def _latin_follows(text: str, i: int) -> bool:
    return i < len(text) and text[i].isascii() and text[i].isalpha()


def _quantity_follows(text: str, i: int) -> bool:
    return i < len(text) and text[i] in "个只条位张本次件岁种名天年辆台家双块元万亿人米公斤克"


def spoken_text(units: list[Unit]) -> str:
    out = ""
    for u in units:
        s = u.spoken
        if out and s and not u.pause and out[-1].isascii() and out[-1].isalnum() and s[0].isascii() and s[0].isalnum():
            out += " "
        out += s
    return out.strip()
