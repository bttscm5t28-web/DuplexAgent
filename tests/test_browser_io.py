"""Browser transport: page is served, mic frames come in, agent audio / text / stop go out."""

import asyncio
import json
import struct
import urllib.request

import pytest

pytest.importorskip("websockets")

from duplexagent.core.audio import tone  # noqa: E402
from duplexagent.core.types import FRAME_SAMPLES, TTSChunk  # noqa: E402
from duplexagent.plugins.io.browser import BrowserInput, BrowserOutput  # noqa: E402


def test_browser_roundtrip(tmp_path):
    async def go():
        from websockets.asyncio.client import connect

        port = 18765
        rec = tmp_path / "mic.wav"
        inp, out = BrowserInput(port=port, record_path=str(rec)), BrowserOutput(port=port)
        inp.bind_clock(lambda: 0.0)
        out.bind_clock(lambda: 0.0)
        await inp.start()
        await out.start()
        html = await asyncio.to_thread(lambda: urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode())
        assert "DuplexAgent" in html

        async with connect(f"ws://127.0.0.1:{port}/ws") as ws:
            await ws.send(b"\x01\x00" * (FRAME_SAMPLES * 3 // 2))  # 1.5 frames ...
            await ws.send(b"\x01\x00" * (FRAME_SAMPLES // 2))  # ... + 0.5 = exactly 2 frames
            frames = inp.frames()
            got = [await anext(frames), await anext(frames)]
            assert all(len(f.pcm) == FRAME_SAMPLES * 2 for f in got)

            await out.play(TTSChunk(tone(0.02, sr=24000), "你", sample_rate=24000))
            audio = await ws.recv()
            assert struct.unpack("<I", audio[:4])[0] == 24000 and len(audio) == 4 + 480 * 2
            assert json.loads(await ws.recv()) == {"type": "text", "text": "你"}
            await out.stop()
            assert json.loads(await ws.recv()) == {"type": "stop"}
        with pytest.raises(StopAsyncIteration):  # client left -> input ends
            await anext(frames)
        await inp.close()
        await out.close()
        assert rec.stat().st_size > 44

    asyncio.run(go())
