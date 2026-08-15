"""Source factory — the model asks for an adapter, this decides which."""

from __future__ import annotations

from .base import NflDataSource
from .nflverse import NflverseSource

_singleton: NflDataSource | None = None


def nfl_source() -> NflDataSource:
    global _singleton
    if _singleton is None:
        _singleton = NflverseSource()
    return _singleton


def set_source(source: NflDataSource | None) -> None:
    """Swap the adapter. Exists so the offline test suite can drive the whole projection
    pipeline (board/analytics included) off synthetic rows without any HTTP; passing None
    restores the live nflverse adapter."""
    global _singleton
    _singleton = source
