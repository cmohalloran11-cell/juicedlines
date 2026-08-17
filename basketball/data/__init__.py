"""Source factory — the model asks for adapters, config decides which."""

from __future__ import annotations

from .balldontlie import BallDontLie

_gamelog_singleton = None


def gamelog_source() -> "BallDontLie":
    """balldontlie.io — WNBA's third data-source attempt; see balldontlie.py's module
    docstring for the full history. Both prior adapters are kept in the tree with their
    own tests passing, just no longer wired in here:
      - basketball/data/espn.py       — ESPN returns HTTP 403 on every basketball/wnba
                                         endpoint (2026-08).
      - basketball/data/wnba_stats.py — every stats.wnba.com call from the production
                                         runner timed out (2026-08); confirmed dead, not
                                         just unreachable from a dev sandbox.
    Requires the BALLDONTLIE_API_KEY env var; absent, every method degrades to an honest
    empty result (see balldontlie.py's `_key()`) rather than a crash."""
    global _gamelog_singleton
    if _gamelog_singleton is None:
        _gamelog_singleton = BallDontLie()
    return _gamelog_singleton
