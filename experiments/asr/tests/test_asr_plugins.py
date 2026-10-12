"""ASR 插件的轻量测试：python -m pytest -q experiments/asr/tests

不需要模型的部分总会跑；需要 sherpa-onnx 且模型已下载（models/asr/）的冒烟测试会自动跳过。
"""

import asyncio
import json
import os

import pytest

from duplexagent.core.registry import available, create
from duplexagent.plugins.asr._util import clean, itn_zh

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_plugins_register_without_optional_deps():
    names = available("asr")["asr"]
    for n in ("sherpa_online", "sherpa_offline", "faster_whisper", "funasr_streaming"):
        assert n in names


def test_unknown_preset_is_rejected():
    with pytest.raises(ValueError, match="presets"):
        create("asr", {"impl": "sherpa_online", "model": "nope"})
    with pytest.raises(ValueError, match="presets"):
        create("asr", {"impl": "sherpa_offline", "model": "nope"})


@pytest.mark.parametrize("src,want", [
    ("帮我计算一下二十三乘以四十七", "帮我计算一下23乘以47"),
    ("一二线城市", "一二线城市"),
    ("万一他不来", "万一他不来"),
    ("十分感谢", "十分感谢"),
    ("一百零五个", "105个"),
    ("三万五千块", "35000块"),
    ("二零二六年", "2026年"),
    ("一点五倍", "1.5倍"),
    ("我有一点点累", "我有一点点累"),
    ("两百块", "200块"),
])
def test_itn_zh(src, want):
    assert itn_zh(src) == want


def test_clean():
    assert clean("<|zh|><|NEUTRAL|>你好 啊 hello  world 我们") == "你好啊hello world我们"


def _have(path):
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return False
    return os.path.isdir(os.path.join(ROOT, path))


@pytest.mark.parametrize("cfg,model_dir", [
    ({"impl": "sherpa_online", "model": "zipformer-zh-2025"}, "models/asr/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30"),
    ({"impl": "sherpa_offline", "model": "sensevoice"}, "models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"),
])
def test_smoke_real_model(cfg, model_dir):
    data = os.path.join(ROOT, "bench_data/asr/aishell.jsonl")
    if not _have(model_dir) or not os.path.exists(data):
        pytest.skip("model or data not downloaded")
    from duplexagent.bench.offline import bench_asr

    os.chdir(ROOT)
    item = json.loads(open(data, encoding="utf-8").readline())
    if not os.path.exists(item["wav"]):
        pytest.skip("wav not generated (run experiments/asr/prepare_data.py)")
    res = asyncio.run(bench_asr({"asr": cfg}, [item]))
    assert res["cer"] < 0.3
