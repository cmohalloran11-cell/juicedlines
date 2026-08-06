"""
fantasy.projections.base — the ProjectionsProvider interface every projection source
implements. Nothing outside fantasy/projections/ may import a concrete adapter directly --
callers get a provider from fantasy.projections.get_provider(). This is the seam the
commercial-provider licensing decision (still TBD) plugs into without touching any
scoring/VOR/draft code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

# Canonical stat keys every adapter must normalize its output to. These match Sleeper's own
# scoring_settings vocabulary (see fantasy.scoring) so a league's real scoring rules can be
# applied without any provider-specific translation living outside the adapter.
CANONICAL_STAT_KEYS: tuple[str, ...] = (
    "pass_yd", "pass_td", "pass_int", "pass_cmp", "pass_att", "pass_2pt",
    "rush_yd", "rush_td", "rush_att", "rush_2pt",
    "rec", "rec_yd", "rec_td", "rec_2pt", "rec_tgt",
    "fum_lost", "fum",
)


class ProjectionsProvider(ABC):
    """One method: raw per-player stat projections for a season (optionally a single week).
    Implementations own everything about their data source -- HTTP, parsing, shrinkage -- and
    must return only the canonical stat keys above, with player identity already resolved to
    our canonical player_id (see fantasy.player_matching) before returning."""

    #: short id stored on every fantasy_projections row (provenance, not a display label)
    name: str

    @abstractmethod
    def get_projections(self, season: int, week: Optional[int] = None) -> list[dict]:
        """Return one dict per projected player:
        {"player_id": str, "position": str, "stats": {canonical_key: float, ...},
         "provider": self.name, "methodology": str}
        `methodology` is a short, honest, user-facing description of how the number was
        derived (e.g. "shrunk prior-season per-game rate x 17 games") -- never omitted, so
        nothing downstream can present this as more authoritative than it is."""
