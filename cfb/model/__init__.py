"""
cfb.model -- the CFB engine's fitting layer: league priors, pace, opponent strength,
garbage time, and the three-tier player-rate fit that blends them.

Every module here is PURE with respect to I/O: it takes already-fetched CFBD dataclasses
(cfb/data/base.py) plus the `GameContext` index below and returns a fitted model object. The
fetching, caching and DB lookups live in cfb/projections.py, the same split
nfl/model/* vs nfl/projections.py already uses -- which is what makes every fit here testable
against fixture rows with no network and no database.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GameContext:
    """One team's view of one game, keyed (game_id, team) -- the join between a box-score row
    and everything the schedule knows about the game it happened in. Built once in
    cfb/projections.py from CFBDClient.schedule so no fitting module has to re-derive home/away
    sign conventions or re-look-up an opponent.

    `spread` is flipped into THIS team's perspective (positive = this team favoured), unlike
    CFBD's own /lines value which is always the home team's -- the same normalisation
    nfl/model/environment.py performs on the schedules release's spread_line, and for the same
    reason: every downstream fit wants "how big a favourite is the team I'm projecting".
    """
    game_id: str
    season: int
    week: int
    team: str
    opponent: str
    is_home: bool
    opponent_classification: Optional[str] = None    # "fbs" | "fcs" | None (unpublished)
    spread: Optional[float] = None                    # positive = THIS team favoured
    over_under: Optional[float] = None
    margin: Optional[float] = None                    # completed games only, this team's view
