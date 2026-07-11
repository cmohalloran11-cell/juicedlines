"""
Tennis data types + adapter interfaces. 1v1 — players and matches, no teams.

The model only ever sees these dataclasses, so any source (Sackmann-format history,
ESPN live, or a licensed feed) just maps into them. Minimum fields the model needs
per historical match are all here: date, tournament, surface, best-of format, both
player ids, serve points won/played, return points won/played, aces, DFs, break
points, and the score.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class PlayerMatch:
    """One PLAYER's line in a completed match (each match yields two of these —
    one per player — so serve and return are both first-class)."""
    date: str                     # YYYY-MM-DD (or YYYYMMDD as given)
    tournament: str
    surface: str                  # Hard | Clay | Grass | Carpet
    best_of: int                  # 3 or 5
    round: str
    player_id: str
    player: str
    opp_id: str
    opp: str
    won: bool
    serve_won: int = 0            # serve points won
    serve_played: int = 0         # serve points played
    return_won: int = 0           # return points won (= opp serve points they lost)
    return_played: int = 0        # return points played (= opp serve points)
    aces: int = 0
    dfs: int = 0
    bp_faced: int = 0
    bp_saved: int = 0
    sv_games: int = 0
    score: str = ""

    @property
    def spw(self) -> float | None:
        return self.serve_won / self.serve_played if self.serve_played else None

    @property
    def rpw(self) -> float | None:
        return self.return_won / self.return_played if self.return_played else None


@dataclass
class UpcomingMatch:
    """A match to project — strictly 1v1."""
    id: str
    tour: str                     # ATP | WTA
    player_a_id: str
    player_a: str
    player_b_id: str
    player_b: str
    surface: str
    best_of: int = 3
    tournament: str = ""
    round: str = ""
    date: str = ""
    # final-set rule varies by event; None → tour default (see config)
    final_set_tb_at: int | None = None   # tiebreak played at N-N in the final set


@dataclass
class RankPoint:
    date: str
    player_id: str
    rank: int
    points: int = 0


@dataclass
class OddsLine:
    match_id: str
    player: str
    market: str                   # match | games | aces | dfs | sets | ...
    line: float | None
    side: str = "over"            # over | under | yes
    price: float = 0.0            # American
    book: str = "sample"


# ── adapter interfaces ────────────────────────────────────────────────────────

class MatchHistorySource(ABC):
    """Historical completed matches with serve stats — for fitting + backtest."""
    @abstractmethod
    def player_matches(self, tour: str, years: list[int]) -> list[PlayerMatch]: ...


class UpcomingSource(ABC):
    """Matches to project (today's/near-future slate)."""
    @abstractmethod
    def upcoming(self, tour: str) -> list[UpcomingMatch]: ...


class RankingSource(ABC):
    @abstractmethod
    def rankings(self, tour: str) -> list[RankPoint]: ...


class OddsSource(ABC):
    @abstractmethod
    def odds(self, match_id: str) -> list[OddsLine]: ...
