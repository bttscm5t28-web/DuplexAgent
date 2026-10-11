"""Minimal client for OpenAI-compatible chat APIs (OpenAI, vLLM, Ollama, DeepSeek, Qwen/DashScope, ...).

httpx is imported lazily so the core has no hard dependency on it.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any


class ChatClient:
    def __init__(self, base_url: str, model: str, api_key_env: str = "OPENAI_API_KEY", timeout: float = 120.0, **params: Any):
        import httpx  # optional dependency: pip install duplexagent[llm]

        key = os.environ.get(api_key_env, "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.model = model
        self.params = params
        self._http = httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)

    async def complete(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        body = {"model": self.model, "messages": messages, **self.params}
        if tools:
            body["tools"] = tools
        r = await self._http.post("/chat/completions", json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]

    async def stream(self, messages: list[dict], tools: list[dict] | None = None) -> AsyncIterator[dict]:
        """Yield `delta` objects of a streamed chat completion."""
        body = {"model": self.model, "messages": messages, "stream": True, **self.params}
        if tools:
            body["tools"] = tools
        async with self._http.stream("POST", "/chat/completions", json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                choices = json.loads(data).get("choices") or []
                if choices:
                    yield choices[0].get("delta", {})

    async def close(self) -> None:
        await self._http.aclose()
