"""Talk to the engine from a browser tab: mic in, speaker out, over one WebSocket.

The browser's WebRTC audio stack gives echo cancellation, noise suppression and AGC, so
this is the recommended way to test full duplex with loudspeakers. It also works with a
remote GPU host: `ssh -L 8765:localhost:8765 host`, then open http://localhost:8765
(getUserMedia needs localhost or https).

    [input]
    impl = "browser"
    [output]
    impl = "browser"     # same host/port as the input

Needs `pip install websockets`.
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
from collections.abc import AsyncIterator

from duplexagent.core.interfaces import AudioInput, AudioOutput
from duplexagent.core.registry import register
from duplexagent.core.types import FRAME_SAMPLES, AudioFrame, TTSChunk
from duplexagent.plugins.io.live import _write_wav

_PAGE = os.path.join(os.path.dirname(__file__), "static", "duplex.html")
_HUBS: dict[tuple[str, int], "_Hub"] = {}


class _Hub:
    """One HTTP+WebSocket server shared by the browser input and output plugins."""

    def __init__(self, host: str, port: int) -> None:
        self.host, self.port = host, port
        self.frames: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.ws = None
        self.connected = asyncio.Event()
        self._server = None
        self._users = 0

    async def acquire(self) -> None:
        self._users += 1
        if self._server is not None:
            return
        from http import HTTPStatus

        from websockets.asyncio.server import serve

        def process_request(connection, request):
            if request.path.startswith("/ws"):
                return None
            with open(_PAGE, encoding="utf-8") as f:
                resp = connection.respond(HTTPStatus.OK, f.read())
            resp.headers["Content-Type"] = "text/html; charset=utf-8"
            return resp

        self._server = await serve(self._handler, self.host, self.port, process_request=process_request, max_size=2**22)
        print(f"[browser] open http://localhost:{self.port} in Chrome and click 开始", flush=True)

    async def release(self) -> None:
        self._users -= 1
        if self._users <= 0 and self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            _HUBS.pop((self.host, self.port), None)

    async def _handler(self, ws) -> None:
        if self.ws is not None:
            await ws.close(1013, "another client is already connected")
            return
        self.ws = ws
        self.connected.set()
        try:
            async for msg in ws:
                if isinstance(msg, bytes):
                    self.frames.put_nowait(msg)
        finally:
            self.ws = None
            self.frames.put_nowait(None)  # client left: end the session

    async def send(self, msg) -> None:
        ws = self.ws
        if ws is not None:
            try:
                await ws.send(msg)
            except Exception:
                pass


def _hub(host: str, port: int) -> _Hub:
    return _HUBS.setdefault((host, port), _Hub(host, port))


@register("input", "browser")
class BrowserInput(AudioInput):
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, record_path: str | None = None) -> None:
        self.hub = _hub(host, port)
        self.record_path = record_path  # keep the processed mic audio (aligned with the timeline) for labeling
        self._rec = bytearray()

    async def start(self) -> None:
        await self.hub.acquire()

    async def close(self) -> None:
        await self.hub.release()
        if self.record_path:
            _write_wav(self.record_path, bytes(self._rec))

    async def frames(self) -> AsyncIterator[AudioFrame]:
        await self.hub.connected.wait()
        step = FRAME_SAMPLES * 2
        buf = b""
        while (msg := await self.hub.frames.get()) is not None:
            buf += msg
            while len(buf) >= step:
                if self.record_path:
                    self._rec += buf[:step]
                yield AudioFrame(pcm=buf[:step], ts=self.clock())
                buf = buf[step:]


@register("output", "browser")
class BrowserOutput(AudioOutput):
    """Sends 16-bit PCM (prefixed with its sample rate) to the page, which schedules it gap-free."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.hub = _hub(host, port)

    async def start(self) -> None:
        await self.hub.acquire()

    async def close(self) -> None:
        await self.hub.release()

    async def play(self, chunk: TTSChunk) -> None:
        await self.hub.send(struct.pack("<I", chunk.sample_rate) + chunk.pcm)
        if chunk.text:
            await self.hub.send(json.dumps({"type": "text", "text": chunk.text}, ensure_ascii=False))
        await asyncio.sleep(chunk.duration)

    async def stop(self) -> None:
        await self.hub.send(json.dumps({"type": "stop"}))
