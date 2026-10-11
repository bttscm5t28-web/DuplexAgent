from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TypeVar

T = TypeVar("T")

_END = object()


class Channel(AsyncIterator[T]):
    """A closable asyncio queue usable as an async iterator."""

    def __init__(self) -> None:
        self._q: asyncio.Queue = asyncio.Queue()

    def put(self, item: T) -> None:
        self._q.put_nowait(item)

    def close(self) -> None:
        self._q.put_nowait(_END)

    def __aiter__(self) -> Channel[T]:
        return self

    async def __anext__(self) -> T:
        item = await self._q.get()
        if item is _END:
            self._q.put_nowait(_END)
            raise StopAsyncIteration
        return item
