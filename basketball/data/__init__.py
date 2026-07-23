"""Source factory — the model asks for adapters, config decides which."""

from __future__ import annotations

from .espn import EspnBasketball

_gamelog_singleton = None


def gamelog_source() -> "EspnBasketball":
    global _gamelog_singleton
    if _gamelog_singleton is None:
        _gamelog_singleton = EspnBasketball()
    return _gamelog_singleton
