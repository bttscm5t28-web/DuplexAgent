"""End-to-end: every bundled scenario passes with the all-mock baseline (runs ~25 s, concurrently)."""

import asyncio
import glob

from duplexagent.cli import run_one
from duplexagent.config import load_config


def test_all_scenarios_pass():
    async def go():
        files = sorted(glob.glob("scenarios/*.toml"))
        return await asyncio.gather(*(run_one(load_config("configs/default.toml"), f, False) for f in files))

    results = asyncio.run(go())
    failed = {m["scenario"]: {k: v for k, v in m["checks"].items() if not v} for m in results if not m["passed"]}
    assert not failed, failed
