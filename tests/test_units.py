import asyncio

import pytest

from duplexagent.config import apply_override, build_engine, load_config
from duplexagent.core.audio import noise, silence
from duplexagent.core.interfaces import TurnContext
from duplexagent.core.registry import available, create
from duplexagent.core.types import AudioFrame, Delegate, Transcript, TurnEventType as T
from duplexagent.plugins.turn.rule import RuleTurnDetector


def test_every_kind_has_a_baseline():
    av = available()
    for kind, name in [("input", "sim"), ("output", "sim"), ("asr", "mock"), ("turn", "rule"),
                       ("fast", "rule"), ("slow", "mock"), ("tts", "mock"), ("fusion", "idle")]:
        assert name in av[kind]


def test_unknown_plugin_lists_alternatives():
    with pytest.raises(ValueError, match="available"):
        create("asr", {"impl": "nope"})


def test_override_parses_values():
    cfg = {}
    apply_override(cfg, "turn.eot_silence_ms=300")
    apply_override(cfg, 'fast.model=gpt-x')
    apply_override(cfg, 'turn.backchannels=["嗯","对"]')
    assert cfg == {"turn": {"eot_silence_ms": 300, "backchannels": ["嗯", "对"]}, "fast": {"model": "gpt-x"}}


def _feed(det, pattern, agent_speaking=False, text=None):
    """pattern: string of 's' (speech) / '.' (silence), one char per 20 ms frame."""
    events, ts = [], 0.0
    for i, c in enumerate(pattern):
        ts = i * 0.02
        ctx = TurnContext(now=ts, agent_speaking=agent_speaking)
        f = AudioFrame(noise(seed=i) if c == "s" else silence(), ts)
        events += det.on_audio(f, ctx)
        if text and i == 5:
            events += det.on_transcript(Transcript(text, True, ts), ctx)
    return [e.type for e in events]


def test_turn_end_after_silence():
    ev = _feed(RuleTurnDetector(), "s" * 20 + "." * 60, text="你好")
    assert ev == [T.SPEECH_START, T.SPEECH_END, T.TURN_END]


def test_short_pause_does_not_end_turn():
    ev = _feed(RuleTurnDetector(), "s" * 20 + "." * 10 + "s" * 20 + "." * 60, text="我想")
    assert ev.count(T.TURN_END) == 1


def test_backchannel_while_agent_speaks():
    ev = _feed(RuleTurnDetector(), "s" * 10 + "." * 60, agent_speaking=True, text="嗯")
    assert T.BACKCHANNEL in ev and T.BARGE_IN not in ev and T.TURN_END not in ev


def test_barge_in_while_agent_speaks():
    ev = _feed(RuleTurnDetector(), "s" * 30 + "." * 60, agent_speaking=True, text="停一下")
    assert ev.index(T.BARGE_IN) < ev.index(T.TURN_END)


def test_backchannel_matcher():
    d = RuleTurnDetector()
    assert d.is_backchannel("嗯嗯，好的")
    assert not d.is_backchannel("好的我想问")


def test_orchestrator_cancel():
    async def go():
        cfg = load_config("configs/default.toml")
        eng = build_engine(cfg)
        from duplexagent.core.orchestrator import TaskOrchestrator
        from duplexagent.core.timeline import Timeline

        orch = TaskOrchestrator(eng.c.slow, lambda: 0.0, Timeline())
        rec = orch.spawn(Delegate("计算 1+1"), 1)
        await asyncio.sleep(0.05)
        orch.cancel(rec.id)
        await orch.drain(1)
        assert rec.status.value == "cancelled" and not rec.awaiting_delivery

    asyncio.run(go())
