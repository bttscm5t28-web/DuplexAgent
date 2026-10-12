"""Unit tests for the hybrid turn detectors (stdlib parts). Run: python -m pytest -q experiments/turn/tests"""

from duplexagent.core.audio import noise, silence
from duplexagent.core.interfaces import TurnContext
from duplexagent.core.registry import create
from duplexagent.core.types import AudioFrame, Transcript, TurnEventType as T
from duplexagent.plugins.turn import lexicon


def feed(det, pattern, agent_speaking=False, texts=None, agent_text=""):
    """pattern: 's' speech / '.' silence per 20 ms frame; texts: {frame_index: (text, is_final)}."""
    out = []
    for i, c in enumerate(pattern):
        ts = i * 0.02
        ctx = TurnContext(now=ts, agent_speaking=agent_speaking, agent_text=agent_text)
        out += [(i, e.type) for e in det.on_audio(AudioFrame(noise(seed=i) if c == "s" else silence(), ts), ctx)]
        if texts and i in texts:
            txt, final = texts[i]
            out += [(i, e.type) for e in det.on_transcript(Transcript(txt, final, ts), ctx)]
    return out


def first(events, typ):
    return next((i for i, t in events if t == typ), None)


def adaptive():
    return create("turn", {"impl": "adaptive"})


def test_lexicon():
    assert lexicon.is_backchannel("对对对") and lexicon.is_backchannel("原来如此")
    assert not lexicon.is_backchannel("嗯那我换个问题")
    assert lexicon.is_stop_command("停") and lexicon.is_stop_command("嗯不是这个意思")
    assert lexicon.text_eot_prob("我想") < 0.2 < 0.8 < lexicon.text_eot_prob("明天会下雨吗")
    assert lexicon.text_eot_prob("我的手机号是一三八") < 0.2
    assert lexicon.text_eot_prob("我不要") > 0.5
    assert lexicon.is_echo("你说的是给我讲讲", "嗯，你说的是「给我讲讲你今天」")


def test_complete_question_ends_fast():
    ev = feed(adaptive(), "s" * 20 + "." * 100, texts={10: ("明天会下雨", False), 28: ("明天会下雨吗", True)})
    assert ev.count((first(ev, T.TURN_END), T.TURN_END)) == 1
    assert (first(ev, T.TURN_END) - 20) * 20 <= 300  # ≤ 300 ms of silence


def test_unfinished_text_waits():
    # "我想" + 900 ms pause: the rule baseline (500 ms) would end the turn; adaptive must not
    ev = feed(adaptive(), "s" * 10 + "." * 45 + "s" * 30 + "." * 120,
              texts={5: ("我想", False), 18: ("我想", True), 60: ("订机票", False), 93: ("订一张机票", True)})
    assert [t for _, t in ev].count(T.TURN_END) == 1
    assert first(ev, T.TURN_END) > 85


def test_stop_command_barges_immediately():
    ev = feed(adaptive(), "s" * 10 + "." * 60, agent_speaking=True, texts={1: ("停", False)})
    assert first(ev, T.BARGE_IN) is not None and first(ev, T.BARGE_IN) <= 4


def test_backchannel_phrase_does_not_barge():
    ev = feed(adaptive(), "s" * 40 + "." * 80, agent_speaking=True,
              texts={5: ("原", False), 15: ("原来", False), 25: ("原来如", False), 35: ("原来如此", False), 50: ("原来如此", True)})
    types = [t for _, t in ev]
    assert T.BARGE_IN not in types and T.TURN_END not in types and T.BACKCHANNEL in types


def test_echo_is_ignored():
    ev = feed(adaptive(), "s" * 40 + "." * 80, agent_speaking=True, agent_text="嗯，你说的是「给我讲讲你今天」",
              texts={20: ("你说的是给我", False), 50: ("你说的是给我讲讲", True)})
    types = [t for _, t in ev]
    assert T.BARGE_IN not in types and T.TURN_END not in types


def test_late_final_does_not_open_a_turn():
    det = adaptive()
    ev = feed(det, "s" * 20 + "." * 100, texts={10: ("你好", False), 22: ("你好", True)})
    assert [t for _, t in ev].count(T.TURN_END) == 1
    ctx = TurnContext(now=5.0, agent_speaking=False)
    assert det.on_transcript(Transcript("你好", True, 5.0), ctx) == []
    ev2 = [e.type for i in range(100) for e in det.on_audio(AudioFrame(silence(), 5 + i * 0.02), ctx)]
    assert T.TURN_END not in ev2
