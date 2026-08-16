"""Source factory — the model asks for adapters, config decides which."""

from __future__ import annotations

from .wnba_stats import WnbaStats

_gamelog_singleton = None


def gamelog_source() -> "WnbaStats":
    """stats.wnba.com/cdn.wnba.com — see wnba_stats.py's module docstring for why this
    replaced the ESPN adapter (basketball/data/espn.py, kept in the tree with its own
    tests but no longer wired in: ESPN returns HTTP 403 on every basketball/wnba endpoint,
    2026-08)."""
    global _gamelog_singleton
    if _gamelog_singleton is None:
        _gamelog_singleton = WnbaStats()
    return _gamelog_singleton
