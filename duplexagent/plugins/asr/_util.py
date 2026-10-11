"""Shared helpers for the real ASR plugins: model download, worker thread, PCM conversion, ITN.

Everything heavy is imported lazily so the skeleton keeps working without optional deps.
"""

from __future__ import annotations

import asyncio
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

MODELS_DIR = os.environ.get("DUPLEXAGENT_MODELS", "models")


def hf_model(repo: str, allow: list[str] | None = None, subdir: str = "asr") -> str:
    """Download a Hugging Face model repo (once) into models/<subdir>/<name>/ and return the path."""
    local = os.path.join(MODELS_DIR, subdir, repo.split("/")[-1])
    if allow and all(_exists(local, p) for p in allow):
        return local
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, local_dir=local, allow_patterns=allow)


def _exists(root: str, pattern: str) -> bool:
    import glob

    return bool(glob.glob(os.path.join(root, pattern)))


def pcm_to_float(pcm: bytes):
    import numpy as np

    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


# One model instance per (kind, key) per process: `eval` runs several scenarios concurrently,
# each with its own plugin instance; loading the same weights five times wastes memory and time.
_CACHE: dict[tuple, Any] = {}
_CACHE_LOCK = threading.Lock()


def cached(key: tuple, load: Callable[[], Any]) -> Any:
    with _CACHE_LOCK:
        if key not in _CACHE:
            _CACHE[key] = load()
        return _CACHE[key]


class Worker:
    """A single background thread for model calls, so inference never blocks the event loop."""

    def __init__(self) -> None:
        self._ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")

    async def run(self, fn: Callable, *args):
        return await asyncio.get_running_loop().run_in_executor(self._ex, fn, *args)

    def submit(self, fn: Callable, *args) -> asyncio.Future:
        return asyncio.get_running_loop().run_in_executor(self._ex, fn, *args)

    def close(self) -> None:
        self._ex.shutdown(wait=False, cancel_futures=True)


# ---- text post-processing -------------------------------------------------------------

_TAG = re.compile(r"<\|[^|]*\|>")


def clean(text: str) -> str:
    """Strip model tags (<|zh|> ...), collapse whitespace between CJK chars, keep spaces between Latin words."""
    text = _TAG.sub("", text).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?<=[一-鿿])\s+|\s+(?=[一-鿿])", "", text)
    return text


_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_UNITS = {"十": 10, "百": 100, "千": 1000}
_BIG = {"万": 10**4, "亿": 10**8}
_RUN = re.compile(r"[零〇一二两三四五六七八九十百千万亿]+(?:点[零〇一二三四五六七八九]+)?")


def _cn_int(s: str) -> int | None:
    total, section, num = 0, 0, None
    for ch in s:
        if ch in _DIGITS:
            if num is not None and num != 0:  # "二三" inside a unit expression is not standard
                return None
            num = _DIGITS[ch]
        elif ch in _UNITS:
            section += (1 if num is None else num) * _UNITS[ch]
            num = None
        elif ch in _BIG:
            section += num or 0
            total += (section or 1) * _BIG[ch]
            section, num = 0, None
        else:
            return None
    return total + section + (num or 0)


def itn_zh(text: str) -> str:
    """Minimal inverse text normalization for Chinese numbers: 二十三乘以四十七 -> 23乘以47.

    Conservative on purpose: converts runs that contain a unit (十百千万亿) plus a digit, runs of
    three or more plain digits (phone numbers, years), and decimals. Leaves 一下 / 一二线 / 十分 alone.
    """

    def conv(m: re.Match) -> str:
        s = m.group()
        whole, _, frac = s.partition("点")
        has_digit = any(c in _DIGITS for c in whole)
        has_unit = any(c in _UNITS or c in _BIG for c in whole)
        if whole[:1] in ("百", "千", "万", "亿"):  # 万一 / 千万 / 百分之 are words, not numbers
            return s
        if has_unit and (has_digit or whole.startswith("十") and len(whole) > 1):
            n = _cn_int(whole)
        elif not has_unit and (len(whole) >= 3 or frac):
            n = None if "两" in whole else "".join(str(_DIGITS[c]) for c in whole)
        else:
            return s
        if n is None:
            return s
        out = str(n)
        if frac:
            out += "." + "".join(str(_DIGITS[c]) for c in frac)
        return out

    return _RUN.sub(conv, text)
