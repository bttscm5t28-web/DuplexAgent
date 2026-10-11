"""Tests for the TTS plugins' shared machinery (no model files needed).

    python -m pytest -q experiments/tts
"""

import asyncio
import math
import threading

import pytest

from duplexagent.core.registry import available
from duplexagent.plugins.tts.streaming import Segmenter
from duplexagent.plugins.tts.textnorm import Normalizer, int_to_zh, spoken_text


def say(text: str, english: str = "spell") -> str:
    return spoken_text(Normalizer(english).units(text))


@pytest.mark.parametrize("n,zh", [(0, "零"), (10, "十"), (15, "十五"), (101, "一百零一"), (1010, "一千零一十"),
                                  (10001, "一万零一"), (100010, "十万零一十"), (1081, "一千零八十一"), (120000000, "一亿二千万")])
def test_int_to_zh(n, zh):
    assert int_to_zh(n) == zh


@pytest.mark.parametrize("text,spoken", [
    ("23乘以47等于1081。", "二十三乘以四十七等于一千零八十一。"),
    ("2025年10月11日", "二零二五年十月十一日"),
    ("下午3:30开会", "下午三点三十分开会"),
    ("降水概率30%", "降水概率百分之三十"),
    ("增长10-20%", "增长百分之十到百分之二十"),
    ("花了¥3.5元", "花了三点五元"),
    ("$100", "一百美元"),
    ("电话13812345678", "电话幺三八幺二三四五六七八"),
    ("2个人，第2名", "两个人，第二名"),
    ("3/4", "四分之三"),
    ("120km/h", "一百二十公里每小时"),
    ("3,256.8元", "三千二百五十六点八元"),
    ("GPT-4", "G P T 四"),
    ("嗯……好吧", "嗯，好吧"),
])
def test_normalizer(text, spoken):
    assert say(text) == spoken


def test_units_cover_input_exactly():
    for text in ["OK, 我用iPhone 15打开了GPT-4的网页，花了¥3.5元。", "「三体」——好看！", "a  b\n c", "3+5=8"]:
        units = Normalizer().units(text)
        assert "".join(u.orig for u in units) == text


def test_english_modes():
    assert say("打开Wi-Fi", "keep") == "打开Wi Fi"
    assert all("一" <= c <= "鿿" or not c.isalpha() for c in say("用GPT", "zh"))


def feed_all(seg: Segmenter, text: str, step: int = 4, final_at_end: bool = True) -> list[str]:
    out = []
    for i in range(0, len(text), step):
        seg.feed(text[i : i + step])
        while (s := seg.pop()) is not None:
            out.append(s)
    while final_at_end and (s := seg.pop(final=True)) is not None:
        out.append(s)
    return out


def test_segmenter_short_first_then_clauses():
    segs = feed_all(Segmenter(), "好的，我让后台去处理，你可以先聊点别的。然后我们再看看结果怎么样。")
    assert "".join(segs) == "好的，我让后台去处理，你可以先聊点别的。然后我们再看看结果怎么样。"
    assert segs[0] == "好的，我让后台去处理，"  # first cut at the first punctuation with >= 4 chars
    assert all(s[-1] in "，。" for s in segs)


def test_segmenter_never_cuts_numbers_or_words():
    seg = Segmenter(first_min=1, first_max=3)
    seg.feed("价格是3.")
    assert seg.pop(urgent=True) in (None, "价格是")  # "3." may still become "3.5"
    seg.feed("5元，")
    rest = feed_all(seg, "")
    assert "3.5" in "".join(rest) or "3.5" in seg.buf
    segs = feed_all(Segmenter(first_max=4), "打开PythonScriptRunner程序")
    joined = lambda a, b: a[-1].isascii() and a[-1].isalnum() and b[0].isascii() and b[0].isalnum()  # noqa: E731
    assert len(segs) >= 2 and not any(joined(a, b) for a, b in zip(segs, segs[1:]))


def test_segmenter_long_clause_forced_cut():
    segs = feed_all(Segmenter(first_max=8), "这是一个非常非常非常非常非常长而且没有任何标点符号的句子")
    assert len(segs) >= 2 and "".join(segs) == "这是一个非常非常非常非常非常长而且没有任何标点符号的句子"


# ---- streaming base with a fake backend ---------------------------------------------------------

np = pytest.importorskip("numpy")

from duplexagent.core.registry import register  # noqa: E402
from duplexagent.plugins.tts.streaming import StreamingTTS, align_units  # noqa: E402


@register("tts", "_fake_test")
class FakeTTS(StreamingTTS):
    """0.2 s tone per CJK char, 0.15 s silence per punctuation (so alignment ground truth is known)."""

    def __init__(self, delay: float = 0.0, **kw):
        super().__init__(**kw)
        self.delay = delay
        self.calls = []

    def _cache_key(self):
        return (id(self),)

    def _load(self):
        return object()

    def _synth(self, model, text, stop):
        import time

        self.calls.append(text)
        time.sleep(self.delay)
        sr = 16000
        parts = []
        for c in text:
            if "一" <= c <= "鿿" or c.isalnum():
                t = np.arange(int(0.2 * sr)) / sr
                parts.append(0.3 * np.sin(2 * math.pi * 220 * t))
            elif c in "，。！？":
                parts.append(np.zeros(int(0.15 * sr)))
        return (np.concatenate(parts) if parts else np.zeros(0)).astype(np.float32), sr


async def _collect(tts, text, step=3):
    async def pieces():
        for i in range(0, len(text), step):
            yield text[i : i + step]

    return [c async for c in tts.synthesize(pieces())]


def test_align_units_tracks_pauses():
    units = Normalizer().units("你好，世界。")
    x = FakeTTS()._synth(None, "你好，世界。", threading.Event())[0]
    spans = align_units(units, x, 16000)
    starts = [a / 16000 for a, _ in spans]
    assert abs(starts[3] - 0.55) < 0.03  # "世" starts after 2 chars + the comma pause


def test_streaming_chunks_are_char_level_and_cover_text():
    tts = FakeTTS()
    text = "好的，我让后台去处理，你可以先聊点别的。"
    chunks = asyncio.run(_collect(tts, text))
    assert "".join(c.text for c in chunks) == text
    assert all(c.sample_rate == 16000 for c in chunks)
    assert len(chunks) >= len(text) - 3  # roughly one chunk per character
    assert len(tts.calls) >= 2  # segmented, not one big call


def test_ignores_mock_kwargs():
    t = FakeTTS(chars_per_sec=6.0, first_chunk_ms=80)
    assert t.ignored == ["chars_per_sec", "first_chunk_ms"]


def test_cancel_stops_synthesis():
    tts = FakeTTS(delay=0.05)

    async def go():
        async def pieces():
            yield "第一句话说完了。第二句话也说完了。第三句话还没说。第四句。"
            await asyncio.sleep(10)

        agen = tts.synthesize(pieces())
        await agen.__anext__()
        await agen.aclose()
        n = len(tts.calls)
        await asyncio.sleep(0.3)
        return n, len(tts.calls)

    n, later = asyncio.run(go())
    assert later <= n + 1  # at most the in-flight segment finishes


def test_sherpa_registered():
    assert "sherpa" in available("tts")["tts"]


def test_openai_speech_against_stub_server():
    import http.server
    import json as _json

    got = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            got["body"] = _json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            n = int(0.2 * 24000) * sum(1 for c in got["body"]["input"] if c.isalnum())
            pcm = (0.3 * np.sin(np.arange(n) * 0.06) * 32767).astype("<i2").tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "audio/pcm")
            self.end_headers()
            self.wfile.write(pcm)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        from duplexagent.core.registry import create

        tts = create("tts", {"impl": "openai_speech", "base_url": f"http://127.0.0.1:{srv.server_port}/v1", "voice": "x"})
        chunks = asyncio.run(_collect(tts, "你好，世界。"))
    finally:
        srv.shutdown()
    assert "".join(c.text for c in chunks) == "你好，世界。"
    assert got["body"]["voice"] == "x" and got["body"]["response_format"] == "pcm"
    assert sum(c.duration for c in chunks) > 0.6


def test_marks_alignment():
    from duplexagent.plugins.tts.streaming import marks_to_spans

    units = Normalizer().units("关于23，好")
    marks = [("关于", 0.0, 0.4), ("二十三", 0.4, 1.0), ("好", 1.3, 1.5)]
    spans = marks_to_spans(units, marks, 16000 * 2, 16000)
    starts = [round(a / 16000, 2) for a, _ in spans]
    assert [u.orig for u in units] == ["关", "于", "23", "，", "好"]
    assert starts == [0.0, 0.2, 0.4, 1.0, 1.3]
