"""Short-timescale event discovery package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "dimming_window",
    "events",
    "filter",
    "periodicity_gate",
    "pipeline",
    "plot",
    "score",
    "tag",
    "triggering",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return import_module(f"malca.stv.{name}")
    raise AttributeError(name)
