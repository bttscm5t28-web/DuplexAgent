"""ElevenLabs TTS / ASR plugins against local mock servers that follow the API reference."""

import asyncio
import base64
import json

import pytest

pytest.importorskip("websockets")

from websockets.asyncio.server import serve  # noqa: E402

from duplexagent.core.audio import noise  # noqa: E402
from duplexagent.core.types import AudioFrame  # noqa: E402
from duplexagent.plugins.asr.elevenlabs import ElevenLabsASR  # noqa: E402
from duplexagent.plugins.tts.elevenlabs import ElevenLabsTTS, split_by_alignment  # noqa: E402

MS_PER_CHAR = 100


async def fake_tts(ws):
    init = json.loads(await ws.recv())
    assert init["text"] == " " and "voice_settings" in init
    assert "/v1/text-to-speech/v1/stream-input" in ws.request.path and "sync_alignment=true" in ws.request.path
    async for raw in ws:
        msg = json.loads(raw)
        if msg["text"] == "":
            await ws.send(json.dumps({"audio": None, "isFinal": True}))
            return
        chars = list(msg["text"])
        n = len(chars) * MS_PER_CHAR * 16  # 16 samples per ms
        await ws.send(json.dumps({
            "audio": base64.b64encode(b"\x01\x00" * n).decode(),
            "alignment": {"chars": chars, "charStartTimesMs": [i * MS_PER_CHAR for i in range(len(chars))],
                          "charDurationsMs": [MS_PER_CHAR] * len(chars)},
        }))


def test_split_by_alignment():
    pcm = b"\x00\x00" * 1600 * 2  # 200 ms
    chunks = split_by_alignment(pcm, {"chars": ["你", "好", " "], "charStartTimesMs": [0, 100, 200], "charDurationsMs": [100, 100, 0]})
    assert [c.text for c in chunks] == ["你", "好", " "]
    assert sum(len(c.pcm) for c in chunks) == len(pcm)


def test_tts_streams_text_and_aligns():
    async def go():
        async with serve(fake_tts, "127.0.0.1", 18766):
            tts = ElevenLabsTTS(voice_id="v1", base_url="ws://127.0.0.1:18766")
            await tts.start()

            async def text():
                for p in ["你好，", "我是", "语音助手。"]:
                    yield p

            chunks = [c async for c in tts.synthesize(text())]
            await tts.close()
            return chunks

    chunks = asyncio.run(go())
    assert "".join(c.text for c in chunks).replace(" ", "") == "你好，我是语音助手。"
    assert all(abs(c.duration - MS_PER_CHAR / 1000) < 1e-6 for c in chunks if c.text)


async def fake_stt(ws):
    assert "model_id=scribe_v2_realtime" in ws.request.path and "audio_format=pcm_16000" in ws.request.path
    n = 0
    async for raw in ws:
        msg = json.loads(raw)
        assert msg["message_type"] == "input_audio_chunk" and msg["sample_rate"] == 16000
        n += 1
        if n == 2:
            await ws.send(json.dumps({"message_type": "partial_transcript", "text": "你"}))
        if n == 4:
            await ws.send(json.dumps({"message_type": "partial_transcript", "text": "你好"}))
        if msg["commit"]:
            await ws.send(json.dumps({"message_type": "committed_transcript", "text": "你好。"}))
            return


def test_asr_partials_and_commit():
    async def go():
        async with serve(fake_stt, "127.0.0.1", 18767):
            asr = ElevenLabsASR(base_url="ws://127.0.0.1:18767")

            async def frames():
                for i in range(12):
                    yield AudioFrame(noise(seed=i), i * 0.02)
                    await asyncio.sleep(0.005)

            return [t async for t in asr.transcribe(frames())]

    out = asyncio.run(go())
    assert [(t.text, t.is_final) for t in out] == [("你", False), ("你好", False), ("你好。", True)]
