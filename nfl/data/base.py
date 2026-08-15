"""
NFL data types + the adapter interface.

The model only ever sees these dataclasses, so any source (nflverse releases today, a paid
feed later) just maps into them. Every field that a source may genuinely not carry is
Optional and defaults to None — a missing snap row means "role unknown, widen the band",
never a zero that reads like a measured value of zero.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PlayerWeek:
    """One player's stat line in one completed game."""
    season: int
    week: int
    season_type: str            # "REG" | "POST" (nflverse publishes no PRE stat lines)
    game_id: str
    player_id: str
    player: str
    position: str
    team: str
    opponent: str
    completions: float = 0.0
    pass_attempts: float = 0.0
    pass_yards: float = 0.0
    pass_tds: float = 0.0
    interceptions: float = 0.0
    sacks: float = 0.0
    carries: float = 0.0
    rush_yards: float = 0.0
    rush_tds: float = 0.0
    receptions: float = 0.0
    targets: float = 0.0
    rec_yards: float = 0.0
    rec_tds: float = 0.0

    def stat(self, key: str) -> float:
        """Value of a canonical market key for this game (see nfl.BASE_MARKETS/COMBOS)."""
        if key == "rush_rec_yards":
            return self.rush_yards + self.rec_yards
        return {
            "pass_yards": self.pass_yards, "pass_attempts": self.pass_attempts,
            "completions": self.completions, "pass_tds": self.pass_tds,
            "rush_yards": self.rush_yards, "rush_attempts": self.carries,
            "receptions": self.receptions, "rec_yards": self.rec_yards,
            "targets": self.targets,
        }.get(key, 0.0)


@dataclass
class SnapWeek:
    """One player's snap counts in one completed game. `offense_pct` is the share of the
    team's offensive snaps they were on the field for — the primary playing-time signal."""
    season: int
    week: int
    season_type: str
    game_id: str
    player: str
    position: str
    team: str
    opponent: str
    offense_snaps: Optional[float] = None
    offense_pct: Optional[float] = None
    defense_snaps: Optional[float] = None
    defense_pct: Optional[float] = None


@dataclass
class ScheduleGame:
    """One scheduled game. `spread_line` is from the HOME team's perspective (positive = home
    favoured), matching the nflverse convention. `total_line`/`spread_line` are the market's
    own numbers — the game-environment model reads them directly rather than re-deriving an
    implied total from a model of its own."""
    game_id: str
    season: int
    game_type: str              # REG | WC | DIV | CON | SB (nflverse publishes no PRE games)
    week: int
    gameday: str                # YYYY-MM-DD
    home_team: str
    away_team: str
    spread_line: Optional[float] = None
    total_line: Optional[float] = None
    roof: Optional[str] = None
    surface: Optional[str] = None
    temp: Optional[float] = None
    wind: Optional[float] = None
    home_coach: Optional[str] = None
    away_coach: Optional[str] = None

    def opponent_of(self, team: str) -> Optional[str]:
        if team == self.home_team:
            return self.away_team
        if team == self.away_team:
            return self.home_team
        return None


@dataclass
class DepthChartEntry:
    """A player's slot on a team's depth chart. `rank` is 1-based (1 = first team). `week` is
    None on the post-2024 nflverse schema, which publishes a rolling snapshot keyed by
    timestamp instead of by week — see nflverse.py."""
    season: int
    team: str
    position: str
    rank: Optional[int]
    player: str
    player_id: Optional[str] = None
    week: Optional[int] = None
    slot: Optional[str] = None      # e.g. "RG", "LWR" — the specific alignment, if published


@dataclass
class RosterWeek:
    """Weekly roster status. `status` is nflverse's raw code (ACT/RES/DEV/CUT/...); the model
    only uses whether it is ACT, and treats anything it doesn't recognise as unknown rather
    than as inactive."""
    season: int
    week: int
    team: str
    player: str
    position: str
    status: Optional[str] = None
    depth_chart_position: Optional[str] = None
    player_id: Optional[str] = None


class NflDataSource(ABC):
    """Weekly stats + snap counts + schedule + depth charts + rosters for one season.

    Every method returns [] rather than raising when a season genuinely isn't published yet
    (a preseason board asks for the current season before any of it exists). A caller must
    treat [] as "unknown", not as "zero" — see model/playing_time.py, which drops to a
    depth-chart prior and then to a positional prior, lowering confidence at each step.
    """

    @abstractmethod
    def player_weeks(self, season: int) -> list[PlayerWeek]:
        """Every skill-position player's weekly stat lines for `season`."""

    @abstractmethod
    def snap_counts(self, season: int) -> list[SnapWeek]:
        """Every player's weekly snap counts for `season`."""

    @abstractmethod
    def schedule(self, season: int) -> list[ScheduleGame]:
        """Every scheduled game for `season`, with the market's spread/total where priced."""

    @abstractmethod
    def depth_charts(self, season: int) -> list[DepthChartEntry]:
        """The most recent depth chart per (team, position, player) for `season`."""

    @abstractmethod
    def rosters(self, season: int) -> list[RosterWeek]:
        """Weekly roster/status rows for `season`."""
