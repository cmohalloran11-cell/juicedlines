"""
Basketball data types + adapter interfaces.

The model only ever sees these dataclasses, so any source (ESPN live, stats.wnba.com,
nba_api, or a paid feed) just maps into them. Minimum per-game fields the core needs:
player id, team, opponent, minutes, and the raw box counts for every projected stat.
Summer League additionally needs a PlayerBackground (draft slot + pre-NBA league +
translated rates) keyed to each player.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class PlayerRef:
    id: str
    name: str
    team_id: str
    team: str
    position: str = ""          # G | F | C (ESPN abbrev)


@dataclass
class PlayerGame:
    """One player's box line in a completed game."""
    date: str                   # YYYY-MM-DD
    league: str
    player_id: str
    player: str
    team_id: str
    team: str
    opp_id: str
    opp: str
    minutes: float
    pts: float = 0.0
    reb: float = 0.0
    # Rebound split — only some feeds carry it (box scores yes, athlete gamelogs no). Used to
    # estimate the player's offensive SHARE, not as rates of their own (see DERIVED_STATS);
    # 0/0 means "this feed didn't say", which drops the game from the split sample.
    orb: float = 0.0
    drb: float = 0.0
    ast: float = 0.0
    stl: float = 0.0
    blk: float = 0.0
    to: float = 0.0             # turnovers
    fgm: float = 0.0
    fga: float = 0.0
    tpm: float = 0.0            # 3-pointers made
    tpa: float = 0.0
    ftm: float = 0.0
    fta: float = 0.0
    pf: float = 0.0
    started: bool | None = None

    def stat(self, key: str) -> float:
        """Base stat or combo by canonical key (pts/reb/ast/stl/blk/3pm/to + combos)."""
        if key == "3pm":
            return self.tpm
        if key in ("pts", "reb", "orb", "drb", "ast", "stl", "blk", "to"):
            return getattr(self, key)
        if key == "stocks":
            return self.stl + self.blk
        if key == "pra":
            return self.pts + self.reb + self.ast
        if key == "pr":
            return self.pts + self.reb
        if key == "pa":
            return self.pts + self.ast
        if key == "ra":
            return self.reb + self.ast
        return 0.0


@dataclass
class PlayerBackground:
    """Pre-NBA translation prior for a Summer League player (thin/no pro history)."""
    player: str
    draft_pick: int | None = None       # overall pick; None = undrafted
    pre_league: str = ""                # NCAA | G-League | International | ""
    archetype: str = ""                 # e.g. rim-runner, ball-handler, 3-and-D wing
    # translated per-40 base rates (already run through the source→SL translation)
    rates40: dict = field(default_factory=dict)   # {"pts": .., "reb": .., ...}
    minutes_prior: float | None = None  # expected minutes from draft slot / role


@dataclass
class TeamPace:
    team_id: str
    team: str
    pace: float                 # possessions / game
    off_rtg: float = 0.0        # points scored / 100 poss
    def_rtg: float = 0.0        # points allowed / 100 poss (opponent adjustment)
    games: int = 0


# ── adapter interfaces ────────────────────────────────────────────────────────

class GameLogSource(ABC):
    """Live rosters + per-game logs + team pace for a league."""

    @abstractmethod
    def teams(self, league: str) -> dict:
        """Team displayName → team id."""

    @abstractmethod
    def players(self, league: str) -> dict:
        """Normalized player name → PlayerRef, across every roster in the league."""

    @abstractmethod
    def gamelog(self, league: str, player_id: str) -> list[PlayerGame]:
        """A player's recent completed games, most-recent-first."""

    @abstractmethod
    def team_pace(self, league: str, team_id: str) -> TeamPace | None:
        """Team pace / ratings from recent box scores (None if unknown)."""

    @abstractmethod
    def league_pace(self, league: str) -> float:
        """League-average possessions per game."""


class BackgroundSource(ABC):
    """Summer League only — draft slot + pre-NBA league + translated rates."""

    @abstractmethod
    def background(self, player: str) -> PlayerBackground | None: ...
