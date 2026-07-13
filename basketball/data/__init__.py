"""Source factory — the model asks for adapters, config decides which."""

from __future__ import annotations

from .espn import EspnBasketball

_gamelog_singleton = None
_background_singleton = None


def gamelog_source() -> "EspnBasketball":
    global _gamelog_singleton
    if _gamelog_singleton is None:
        _gamelog_singleton = EspnBasketball()
    return _gamelog_singleton


def background_source():
    """Summer League background (draft slot + translated pre-NBA rates).

    Singleton so the (multi-MB) Torvik college index is fetched once and reused
    across every Summer-League player in a build, not re-pulled per player.
    """
    global _background_singleton
    if _background_singleton is None:
        from .background import TorvikRealGMBackground
        _background_singleton = TorvikRealGMBackground()
    return _background_singleton
