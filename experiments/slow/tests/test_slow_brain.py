"""Slow-brain experiment tests: tools, sandbox, router, agent loops (via a scripted mock LLM), MCP.

    python -m pytest -q experiments/slow/tests
"""

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "slow"))
os.chdir(ROOT)

from duplexagent.core.registry import create  # noqa: E402
from duplexagent.core.types import TaskRecord  # noqa: E402
from duplexagent.plugins.slow.agent import parse_prompt_tool_calls, split_answer  # noqa: E402
from duplexagent.plugins.slow.router import RuleSolver, apply_amendments, cn2num  # noqa: E402
from duplexagent.plugins.slow.sandbox import PythonSandbox  # noqa: E402
from duplexagent.plugins.slow.toolbox import ToolBox, calc, convert_units  # noqa: E402


def run(coro):
    return asyncio.run(coro)


def test_calculator_and_units():
    assert calc("(12+8)*3") == 60
    assert calc("sqrt(16) + 2^3") == 12
    assert abs(convert_units(100, "F", "C") - 37.7777) < 1e-3
    assert convert_units(3, "斤", "克") == 1500
    with pytest.raises(ValueError):
        calc("__import__('os')")
    with pytest.raises(ValueError):
        calc("9**9**9")


def test_toolbox_date_and_errors():
    tb = ToolBox(fixed_now="2026-10-11T10:00")
    r = run(tb.call("date_calc", {"start": "2024-02-01", "end": "2024-03-01"}))
    assert r.ok and "29 天" in r.text
    r = run(tb.call("now", {}))
    assert "星期日" in r.text
    assert not run(tb.call("nope", {})).ok
    assert not run(tb.call("calculator", {"expression": "1/0"})).ok
    assert tb.stats.calls == 4 and tb.stats.ok == 2


@pytest.mark.parametrize("code,ok,needle", [
    ("print(sum(range(101)))", True, "5050"),
    ("2**10", True, "1024"),
    ("import socket", False, "not allowed"),
    ("import os; os.system('id')", False, "not allowed"),
    ("open('/etc/passwd').read()", False, "not allowed"),
    ("open('x','w')", False, "not allowed"),
    ("while True: pass", False, "CPU"),
])
def test_sandbox(code, ok, needle):
    got_ok, out = run(PythonSandbox(timeout_s=6, cpu_s=2).run(code))
    assert got_ok == ok and needle in out


def test_sandbox_runs_unprivileged_without_network():
    ok, out = run(PythonSandbox().run("import os; print(os.getuid())"))
    if os.geteuid() == 0:
        assert out.strip() == "65534"


def test_router_and_helpers():
    assert cn2num("二十三") == 23 and cn2num("一万零五") == 10005
    assert apply_amendments("计算 360/3", ["把 3 改成 4"]) == "计算 360/4"
    s = RuleSolver(ToolBox(fixed_now="2026-10-11T10:00"))
    assert "1081" in run(s.solve("帮我计算一下 23 乘以 47"))[1]
    assert "星期四" in run(s.solve("2026年10月1日是星期几"))[1]
    assert run(s.solve("水的化学式是什么")) is None


def test_answer_parsing():
    sp, de = split_answer("<think>x</think>结论：**答案是 1081**。\n详细：23×47=1081")
    assert sp == "答案是 1081。" and de == "23×47=1081"
    text, calls = parse_prompt_tool_calls('<tool_call>\n{{"name": "calculator", "arguments": {"expression": "1+1"}}\n</tool_call>')
    assert calls and calls[0].name == "calculator" and calls[0].args == {"expression": "1+1"}


@pytest.fixture(scope="module")
def mock_url():
    import mock_llm

    srv, port = mock_llm.serve(0, 0.3)
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


def _cfgs(url):
    return [
        {"impl": "agent", "base_url": url + "/v1", "model": "scripted"},
        {"impl": "agent", "base_url": url + "/v1", "model": "scripted", "tool_mode": "prompt"},
        {"impl": "claude", "base_url": url, "model": "claude-opus-5-5"},
        {"impl": "agent", "base_url": url + "/v1", "model": "scripted", "stream": True},
    ]


async def _drive(cfg, query, amend=None, cancel_at=None):
    os.environ.setdefault("ANTHROPIC_API_KEY", "dummy")
    brain = create("slow", cfg)
    await brain.start()
    rec, ups = TaskRecord(query=query), []

    async def consume():
        async for u in brain.run(rec):
            ups.append(u)

    t = asyncio.create_task(consume())
    cancel_latency = None
    try:
        if amend:
            await asyncio.sleep(amend[0])
            rec.amendments.append(amend[1])
        if cancel_at is not None:
            await asyncio.sleep(cancel_at)
            t0 = time.perf_counter()
            t.cancel()
            await asyncio.gather(t, return_exceptions=True)
            cancel_latency = time.perf_counter() - t0
        else:
            await asyncio.wait_for(t, 20)
    finally:
        await brain.close()
    return ups, cancel_latency


@pytest.mark.parametrize("i", [0, 1, 2, 3])
def test_agent_loop_tools_progress(mock_url, i):
    ups, _ = run(_drive(_cfgs(mock_url)[i], "计算 360/3"))
    kinds = [u.kind for u in ups]
    assert kinds[0] == "progress" and kinds[-1] == "result"
    assert "120" in ups[-1].text
    assert any("计算器" in u.text for u in ups if u.kind == "progress")


@pytest.mark.parametrize("i", [0, 1, 2, 3])
def test_agent_amendment_preempts(mock_url, i):
    ups, _ = run(_drive(_cfgs(mock_url)[i], "计算 360/3", amend=(0.1, "把 3 改成 4")))
    assert "90" in ups[-1].text
    assert any("收到补充" in u.text for u in ups)


def _sandbox_children() -> int:
    n = 0
    for p in Path("/proc").iterdir():
        try:
            cmd = (p / "cmdline").read_bytes() if p.name.isdigit() else b""
            if b"\x00-I\x00-S\x00-c\x00" in cmd and b"_BLOCK_MODS" in cmd:
                n += 1
        except OSError:
            pass
    return n


@pytest.mark.parametrize("i", [0, 2])
def test_agent_cancel_kills_sandbox(mock_url, i):
    ups, lat = run(_drive(_cfgs(mock_url)[i], "用程序算个大数", cancel_at=0.8))
    assert lat is not None and lat < 0.5
    assert not any(u.kind == "result" for u in ups)
    time.sleep(0.2)
    assert _sandbox_children() == 0


def test_mcp_roundtrip():
    pytest.importorskip("mcp")
    from duplexagent.plugins.slow.mcp_tools import MCPToolSource

    async def go():
        src = MCPToolSource([{"name": "local", "command": sys.executable, "args": ["-m", "duplexagent.plugins.slow.mcp_tools"], "cwd": str(ROOT)}])
        tools = {t.name: t for t in await src.start()}
        try:
            return await tools["local__calculator"].fn(expression="23*47")
        finally:
            await src.close()

    assert "1081" in run(go())


def test_hybrid_routes_and_falls_back():
    async def go():
        b = create("slow", {"impl": "hybrid", "inner": {"impl": "mock", "delay_s": 0.05}})
        await b.start()
        out = []
        for q in ("计算 23*47", "水的化学式是什么"):
            ups = [u async for u in b.run(TaskRecord(query=q))]
            out.append(ups[-1].text)
        await b.close()
        return out

    a, b = run(go())
    assert "1081" in a and "模拟结果" in b


def test_claude_degrades_without_betas():
    import mock_llm

    srv, port = mock_llm.serve(0, 0.05, reject_betas=True)
    try:
        cfg = {"impl": "claude", "base_url": f"http://127.0.0.1:{port}", "model": "claude-opus-5-5"}

        async def go():
            os.environ.setdefault("ANTHROPIC_API_KEY", "dummy")
            b = create("slow", cfg)
            await b.start()
            ups = [u async for u in b.run(TaskRecord(query="计算 360/3"))]
            await b.close()
            return ups, b

        ups, b = run(go())
        assert "120" in ups[-1].text and not b.progress_notes and not b.fallbacks
    finally:
        srv.shutdown()
