"""Event log of a session, used for metrics and debugging."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    t: float
    name: str
    data: dict[str, Any] = field(default_factory=dict)


class Timeline:
    def __init__(self, echo: bool = False) -> None:
        self.events: list[Event] = []
        self.echo = echo

    def log(self, t: float, name: str, **data: Any) -> None:
        self.events.append(Event(t, name, data))
        if self.echo:
            extra = " ".join(f"{k}={v!r}" for k, v in data.items())
            print(f"[{t:7.3f}] {name:<16} {extra}", flush=True)

    def of(self, name: str) -> list[Event]:
        return [e for e in self.events if e.name == name]

    def dump(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps({"t": round(e.t, 4), "name": e.name, **e.data}, ensure_ascii=False) + "\n")
