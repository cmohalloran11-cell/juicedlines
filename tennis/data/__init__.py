"""Source factory — returns the adapters selected in config `sources`.
(Phase 1 wires history; upcoming/rankings/odds land in later phases.)"""

from __future__ import annotations

from ..config import cfg
from .base import MatchHistorySource


def get_history() -> MatchHistorySource:
    src = cfg("sources", "history")
    if src in ("sackmann", None):
        from .sackmann import SackmannHistory
        return SackmannHistory()
    raise NotImplementedError(f"history source '{src}' not implemented — use 'sackmann' or wire a licensed adapter")
