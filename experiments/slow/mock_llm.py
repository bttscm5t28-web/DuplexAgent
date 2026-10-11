"""Scripted LLM server speaking both the OpenAI Chat Completions and Anthropic Messages wire formats.

Used to validate the slow-brain adapters (tool calls, tool results, amendments, cancellation)
without API keys or a real model. The "model" is deterministic:

- no tool result yet for the current question → call `calculator` if the (amended) task contains
  arithmetic, otherwise `python` if it says 用程序/编程, otherwise answer directly;
- tool result present → answer `结论：<result>` + `详细：…`.

Latency per request is `delay` seconds (CLI flag or `X-Mock-Delay` header), so cancellation and
amendment pre-emption can be timed.

    python experiments/slow/mock_llm.py --port 8090 --delay 0.5
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
import sys
import uuid
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for duplexagent imports


def _texts(content) -> str:
    if isinstance(content, str):
        return content
    out = []
    for b in content or []:
        if isinstance(b, dict):
            if b.get("type") == "text":
                out.append(b.get("text", ""))
            elif b.get("type") == "tool_result":
                c = b.get("content")
                out.append("<tool_response>" + (c if isinstance(c, str) else _texts(c)) + "</tool_response>")
    return "\n".join(out)


def decide(messages: list[dict]) -> tuple[str, dict | None]:
    """Return (text, tool_call) for a normalized [{role, content:str}] conversation."""
    from duplexagent.plugins.slow.router import apply_amendments, extract_math

    task = next((m["content"] for m in messages if m["role"] == "user"), "")
    task = re.sub(r"^任务：", "", task.split("\n")[0])
    amends = []
    for m in messages:
        if m["role"] == "user":
            amends += re.findall(r"用户补充：([^\n]+)", m["content"])
    amends = [a for s in amends for a in s.split("；")]
    q = apply_amendments(task, amends)
    # Has the latest user turn (question or amendment) been answered by a tool already?
    last_user = max((i for i, m in enumerate(messages) if m["role"] == "user" and "<tool_response>" not in m["content"] and m["content"].strip()), default=-1)
    tool_out = None
    for m in messages[last_user + 1:]:
        if m["role"] in ("tool",) or "<tool_response>" in m["content"]:
            tool_out = re.sub(r"</?tool_response>", "", m["content"]).strip()
    if tool_out is not None:
        value = tool_out.rsplit("=", 1)[-1].strip()
        return f"结论：结果是 {value}。\n详细：根据工具输出「{tool_out}」。", None
    if "用程序" in q or "编程" in q:
        return "", {"name": "python", "arguments": {"code": "import time\nfor _ in range(10**8): pass\nprint('done')"}}
    expr = extract_math(q)
    if expr:
        return "先算一下。", {"name": "calculator", "arguments": {"expression": expr}}
    return f"结论：（模拟）关于「{q}」我没有工具可用。", None


class Handler(BaseHTTPRequestHandler):
    delay = 0.3
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"object": "list", "data": [{"id": "scripted", "object": "model"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        time.sleep(float(self.headers.get("X-Mock-Delay", self.delay)))
        path = self.path.split("?")[0]
        if path.endswith("/chat/completions"):
            msgs = [{"role": m["role"], "content": _texts(m.get("content"))} for m in req["messages"] if m["role"] != "system"]
            prompt_tools = "<tools>" in _texts(next((m["content"] for m in req["messages"] if m["role"] == "system"), ""))
            text, call = decide(msgs) if req.get("tools") or prompt_tools else (decide(msgs)[0] or "结论：好的。", None)
            if prompt_tools and call:  # text-form tool call (Hermes/Qwen style)
                text, call = text + f"\n<tool_call>{json.dumps(call, ensure_ascii=False)}</tool_call>", None
            msg = {"role": "assistant", "content": text}
            if call:
                msg["tool_calls"] = [{"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                                      "function": {"name": call["name"], "arguments": json.dumps(call["arguments"], ensure_ascii=False)}}]
            self._send({"id": "x", "object": "chat.completion", "model": req.get("model"),
                        "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if call else "stop"}]})
        elif path.endswith("/messages"):
            msgs = [{"role": m["role"], "content": _texts(m.get("content"))} for m in req["messages"]]
            allow = bool(req.get("tools")) and req.get("tool_choice", {}).get("type") != "none"
            text, call = decide(msgs)
            if not allow and call:
                text, call = "结论：好的。", None
            content = [{"type": "thinking", "thinking": "先确定需要哪个工具。", "signature": "sig"}] if req.get("thinking", {}).get("display") == "updates" else []
            if text:
                content.append({"type": "text", "text": text})
            if call:
                content.append({"type": "tool_use", "id": f"toolu_{uuid.uuid4().hex[:8]}", "name": call["name"], "input": call["arguments"]})
            self._send({"id": f"msg_{uuid.uuid4().hex[:8]}", "type": "message", "role": "assistant", "model": req.get("model"),
                        "content": content, "stop_reason": "tool_use" if call else "end_turn", "stop_sequence": None,
                        "usage": {"input_tokens": 10, "output_tokens": 5}})
        else:
            self._send({"error": "not found"}, 404)


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):  # client went away (pre-emption / cancel): expected
        pass


def serve(port: int = 0, delay: float = 0.3) -> tuple[ThreadingHTTPServer, int]:
    """Start in a background thread; returns (server, port)."""
    handler = type("H", (Handler,), {"delay": delay})
    srv = _Server(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--delay", type=float, default=0.3)
    a = ap.parse_args()
    srv, port = serve(a.port, a.delay)
    print(f"mock LLM on http://127.0.0.1:{port} (OpenAI: /v1/chat/completions, Anthropic: /v1/messages)")
    threading.Event().wait()
