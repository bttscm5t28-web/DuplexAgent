"""Load a TOML config and build the engine's components from the plugin registry."""

from __future__ import annotations

import copy
import os
import tomllib
from typing import Any

from duplexagent.core.engine import Components, DuplexEngine, EngineConfig
from duplexagent.core.registry import create, load_modules
from duplexagent.core.timeline import Timeline


def load_config(path: str, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load TOML. A top-level `extends = "other.toml"` (relative to this file) is loaded first
    and this file is deep-merged over it; a table that sets `impl` replaces the base table."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    base = cfg.pop("extends", None)
    if base:
        cfg = merge(load_config(os.path.join(os.path.dirname(path), base)), cfg)
    for ov in overrides or []:
        apply_override(cfg, ov)
    return cfg


def merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and "impl" not in v:
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_override(cfg: dict, expr: str) -> None:
    """`asr.impl=funasr`, `fast.think_ms=50`, `turn.backchannels=["嗯","对"]`."""
    key, _, raw = expr.partition("=")
    try:
        value = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw
    *parents, leaf = key.strip().split(".")
    node = cfg
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = value


def build_engine(cfg: dict[str, Any], timeline: Timeline | None = None) -> DuplexEngine:
    cfg = copy.deepcopy(cfg)
    load_modules(cfg.get("plugins", {}).get("modules", []))
    parts = {k: create(k, cfg.get(k, {})) for k in Components.__annotations__}
    return DuplexEngine(Components(**parts), EngineConfig(**cfg.get("engine", {})), timeline)
