"""Plugin registry. Every swappable module registers under (kind, name).

    @register("asr", "funasr")
    class FunASR(ASR): ...

Config then selects it with `[asr] impl = "funasr"`; remaining keys become kwargs.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable

KINDS = ("input", "output", "asr", "turn", "fast", "slow", "tts", "fusion")

_REGISTRY: dict[str, dict[str, type]] = {k: {} for k in KINDS}


def register(kind: str, name: str) -> Callable[[type], type]:
    if kind not in _REGISTRY:
        raise ValueError(f"unknown plugin kind {kind!r}; expected one of {KINDS}")

    def deco(cls: type) -> type:
        if name in _REGISTRY[kind] and _REGISTRY[kind][name] is not cls:
            raise ValueError(f"duplicate plugin {kind}/{name}")
        _REGISTRY[kind][name] = cls
        cls.plugin_name = name
        return cls

    return deco


def available(kind: str | None = None) -> dict[str, list[str]]:
    load_builtin_plugins()
    kinds = [kind] if kind else list(KINDS)
    return {k: sorted(_REGISTRY[k]) for k in kinds}


def create(kind: str, cfg: dict[str, Any]) -> Any:
    load_builtin_plugins()
    cfg = dict(cfg)
    name = cfg.pop("impl", None)
    if name is None:
        raise ValueError(f"[{kind}] needs an `impl` key")
    try:
        cls = _REGISTRY[kind][name]
    except KeyError:
        raise ValueError(f"no {kind} plugin named {name!r}; available: {sorted(_REGISTRY[kind])}") from None
    return cls(**cfg)


_loaded = False


def load_builtin_plugins() -> None:
    """Import every module under duplexagent.plugins so their @register calls run.

    A plugin whose optional dependency is missing must import it lazily (inside
    __init__ or start), so one missing package never breaks the rest.
    """
    global _loaded
    if _loaded:
        return
    _loaded = True
    import duplexagent.plugins as pkg

    for mod in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        importlib.import_module(mod.name)


def load_modules(modules: list[str]) -> None:
    """Import extra plugin modules named in config (`[plugins] modules = [...]`)."""
    for m in modules:
        importlib.import_module(m)
