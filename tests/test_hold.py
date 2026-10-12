"""User speech while the agent is still thinking: a backchannel must not cancel the answer,
a continuation must be merged into the same turn."""

import asyncio

from duplexagent.config import build_engine, load_config
from duplexagent.core.timeline import Timeline


def _run(turns, think_ms=1500, duration=9.0):
    cfg = load_config("configs/default.toml")
    cfg["fast"]["think_ms"] = think_ms
    cfg["input"] = {"impl": "sim", "scenario": {"name": "t", "duration": duration, "turns": turns}}
    tl = Timeline()
    eng = build_engine(cfg, tl)
    asyncio.run(eng.run())
    return eng, tl


def test_backchannel_while_thinking_keeps_answer():
    # turn ends ~1.4 s; agent thinks 1.5 s; "嗯" at 1.8 s lands while it is still thinking
    eng, tl = _run([{"at": 0.5, "text": "你好"}, {"at": 1.8, "text": "嗯"}])
    assert len(tl.of("turn_end")) == 1
    assert tl.of("backchannel") and tl.of("hold")
    assert any("有什么可以帮你" in e.data["played"] for e in tl.of("agent_end"))


def test_continuation_while_thinking_is_merged():
    eng, tl = _run([{"at": 0.5, "text": "我想问"}, {"at": 1.9, "text": "你是谁"}], duration=10.0)
    users = [m.content for m in eng.history if m.role == "user"]
    assert users == ["我想问你是谁"]
    assert any("语音助手" in e.data["played"] for e in tl.of("agent_end"))


def test_fused_result_cut_off_is_redelivered():
    # Faster TTS: the result starts just before the user's "你好" at 7.8 s and gets cut off.
    cfg = load_config("configs/default.toml")
    cfg["tts"]["chars_per_sec"] = 10.0
    cfg["input"]["scenario"] = "scenarios/delegate.toml"
    tl = Timeline()
    asyncio.run(build_engine(cfg, tl).run())
    played = "".join(e.data["played"] for e in tl.of("agent_end"))
    assert any(e.data["interrupted"] and e.data["played"].startswith("关于") for e in tl.of("agent_end"))
    assert "1081" in played
