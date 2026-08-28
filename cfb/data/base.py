"""
cfb.data.base — CFB data types + the swappable adapter interface.

Same contract as nfl/data/base.py and basketball/data/base.py: the model only ever sees these
dataclasses, so any source (CFBD today, a different provider later) just maps into them.
Every field a source may genuinely not carry is Optional and defaults to None -- a missing
efficiency number means "unknown, widen the band", never a silently-substituted zero (an
offense that hasn't played this week can't have a real 0.0 PPA).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class TeamRef:
    id: str                                # CFBD team id
    school: str
    conference: Optional[str] = None
    classification: Optional[str] = None   # "fbs" | "fcs" | ...
    abbreviation: Optional[str] = None


@dataclass
class PlayerRef:
    id: str                                # CFBD athlete id
    name: str
    team: str
    position: str = ""
    jersey: Optional[int] = None


@dataclass
class PlayerGameStats:
    """One player's box line in one completed game."""
    season: int
    week: int
    season_type: str                       # "regular" | "postseason"
    game_id: str
    player_id: str
    player: str
    team: str
    opponent: str
    pass_completions: float = 0.0
    pass_attempts: float = 0.0
    pass_yards: float = 0.0
    pass_tds: float = 0.0
    interceptions: float = 0.0
    rush_attempts: float = 0.0
    rush_yards: float = 0.0
    rush_tds: float = 0.0
    receptions: float = 0.0
    targets: float = 0.0
    rec_yards: float = 0.0
    rec_tds: float = 0.0

    def stat(self, key: str) -> float:
        """Value of a canonical market key (see cfb.config.ODDS_MARKET_TO_STAT's values) for
        this game. "Anytime TD" is derived (rush or receiving TD > 0), not a raw box field."""
        if key == "Anytime TD":
            return 1.0 if (self.rush_tds + self.rec_tds) > 0 else 0.0
        return {
            "Passing Yards": self.pass_yards,
            "Rushing Yards": self.rush_yards,
            "Receiving Yards": self.rec_yards,
            "Receptions": self.receptions,
        }.get(key, 0.0)


@dataclass
class TeamGameEfficiency:
    """Advanced per-game team efficiency (CFBD's `/stats/game/advanced`). The modeling
    agent's opponent-adjustment (vs FCS / bottom-quartile opponents) and pace inputs. None
    fields mean the source hasn't published that number for this game yet."""
    season: int
    week: int
    team: str
    opponent: str
    offense_ppa: Optional[float] = None            # predicted points added / play, offense
    defense_ppa: Optional[float] = None             # PPA/play ALLOWED (higher = worse defense)
    offense_success_rate: Optional[float] = None
    defense_success_rate: Optional[float] = None
    plays: Optional[int] = None


@dataclass
class ScheduleGame:
    """One scheduled game. `spread`/`over_under` are the market's own numbers where CFBD's
    `/lines` endpoint has them (it aggregates real sportsbook lines) -- read these directly
    for game environment, same as nfl/data/base.py's ScheduleGame, rather than re-deriving an
    implied total from a model of its own."""
    id: str
    season: int
    week: int
    season_type: str
    start_date: str                         # ISO-8601
    home_team: str
    away_team: str
    home_classification: Optional[str] = None
    away_classification: Optional[str] = None
    spread: Optional[float] = None          # home-team perspective, negative = home favoured
    over_under: Optional[float] = None
    completed: bool = False

    def opponent_of(self, team: str) -> Optional[str]:
        if team == self.home_team:
            return self.away_team
        if team == self.away_team:
            return self.home_team
        return None


class CfbDataSource(ABC):
    """Teams + rosters + schedule + per-game player/team stats for one season.

    Every method returns [] rather than raising when a season/week genuinely isn't published
    yet (e.g. a preseason board asks for the current season before any games exist) -- a
    caller must treat [] as "unknown", not "zero", the same contract nfl/data/base.py's
    NflDataSource documents.
    """

    @abstractmethod
    def teams(self, season: int, classification: str = "fbs") -> list[TeamRef]:
        """Every team in `classification` for `season` (all 134 FBS teams by default)."""

    @abstractmethod
    def roster(self, team: str, season: int) -> list[PlayerRef]:
        """One team's roster for `season`."""

    @abstractmethod
    def schedule(self, season: int, week: Optional[int] = None) -> list[ScheduleGame]:
        """Scheduled/completed games for `season` (optionally one `week`), with the market's
        own spread/total where CFBD's `/lines` has them."""

    @abstractmethod
    def player_game_stats(self, season: int, week: Optional[int] = None,
                          team: Optional[str] = None) -> list[PlayerGameStats]:
        """Per-player box scores for `season` (optionally scoped to one week/team)."""

    @abstractmethod
    def team_efficiency(self, season: int, week: Optional[int] = None) -> list[TeamGameEfficiency]:
        """Per-team-game advanced efficiency (PPA, success rate) -- the modeling agent's
        opponent-adjustment and pace inputs."""
