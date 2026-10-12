"""MCP bridge for the slow brain (both directions; `mcp` is an optional dependency, imported lazily).

Client: `mcp_servers = [{name = "x", command = "...", args = [...]}]` in an agent config spawns
those stdio MCP servers and exposes their tools to the model as `<name>__<tool>`.

Server: `python -m duplexagent.plugins.slow.mcp_tools` serves the built-in sandboxed toolbox over
stdio, so other agents (Claude Desktop, Claude Code, any MCP client) can reuse the same tools.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import re
from typing import Any

from duplexagent.plugins.slow.toolbox import Tool


class MCPToolSource:
    def __init__(self, servers: list[dict], call_timeout_s: float = 30.0):
        self.servers = servers
        self.call_timeout_s = call_timeout_s
        self._stack: contextlib.AsyncExitStack | None = None

    async def start(self) -> list[Tool]:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._stack = contextlib.AsyncExitStack()
        tools: list[Tool] = []
        for spec in self.servers:
            name = re.sub(r"\W", "_", spec.get("name", "mcp"))
            params = StdioServerParameters(command=spec["command"], args=list(spec.get("args", [])), env=spec.get("env"), cwd=spec.get("cwd"))
            read, write = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            listed = await session.list_tools()
            for t in listed.tools:
                tools.append(self._wrap(name, session, t))
        return tools

    def _wrap(self, server: str, session: Any, t: Any) -> Tool:
        remote = t.name

        async def fn(**args) -> str:
            res = await asyncio.wait_for(session.call_tool(remote, args), self.call_timeout_s)
            text = "\n".join(getattr(c, "text", "") for c in (res.content or []) if getattr(c, "text", None))
            if getattr(res, "is_error", False) or getattr(res, "isError", False):
                raise RuntimeError(text or "MCP 工具出错")
            return text or "(空结果)"

        schema = dict(getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {"type": "object", "properties": {}})
        return Tool(f"{server}__{remote}", t.description or remote, schema, fn, lambda a, r=remote: f"调用 {r}")

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None


def serve(web: bool = False) -> None:
    """Run the built-in toolbox as a stdio MCP server."""
    from mcp.server.mcpserver import MCPServer

    from duplexagent.plugins.slow.toolbox import ToolBox

    tb = ToolBox(web=web)
    server = MCPServer("duplexagent-slow-tools")
    for t in tb.tools.values():
        def make(t=t):
            @functools.wraps(t.fn)  # keep the typed signature for schema generation
            async def run(**args) -> str:
                r = await tb.call(t.name, args)
                return r.text if r.ok else f"错误：{r.text}"
            return run
        server.add_tool(make(), name=t.name, description=t.description)
    server.run("stdio")


if __name__ == "__main__":
    import sys

    serve(web="--web" in sys.argv)
