"""
Data adapter interfaces + shared types.

Each real source (API-Football, Understat/FBref, The Odds API, or a CSV you drop
in wc/data/files/) implements these interfaces, so the model never touches a raw
source. Swap sources in config.yaml. See sample.py for a full working stub.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Fixture:
    id: str
    home: str
    away: str
    date: str
    stage: str = "group"          # group | r32 | r16 | qf | sf | final
    neutral: bool = True          # WC 2026: most games neutral; hosts get a home edge
    knockout: bool = False        # elimination game → card-intensity bump
    rivalry: bool = False         # derby / historic rivalry → card-intensity bump
    host_home: str | None = None  # host nation playing "at home" (gets home_advantage)


@dataclass
class TeamStrength:
    team: str
    gf_pg: float                  # goals for / game (recency-weighted by the source)
    ga_pg: float                  # goals against / game
    xg_pg: float = 0.0
    xga_pg: float = 0.0
    shots_pg: float = 12.0
    possession: float = 0.50


@dataclass
class Player:
    name: str
    team: str
    position: str                 # GK | DF | MF | FW
    minutes: float = 0.0          # minutes in the lookback window
    shots90: float = 0.0
    sot90: float = 0.0
    xg90: float = 0.0
    xg_share: float = 0.0         # share of the team's xG (0..1) — drives goal allocation
    fouls90: float = 0.0
    yellow90: float = 0.0
    red90: float = 0.0
    save_pct: float = 0.0         # GK only
    start_prob: float = 0.80      # 1.0 confirmed … <0.55 rotation risk


@dataclass
class OddsLine:
    match_id: str
    player: str
    market: str                   # goal | sot | card | saves
    line: float | None            # over/under line; None = anytime/yes-no
    side: str = "over"            # over | under | yes
    price: float = 0.0            # AMERICAN odds (+150, -120)
    book: str = "sample"


# ── adapter interfaces ────────────────────────────────────────────────────────

class FixtureSource(ABC):
    @abstractmethod
    def fixtures(self) -> list[Fixture]: ...


class TeamStrengthSource(ABC):
    @abstractmethod
    def strength(self, team: str) -> TeamStrength | None: ...


class PlayerSource(ABC):
    @abstractmethod
    def players(self, team: str) -> list[Player]: ...


class OddsSource(ABC):
    @abstractmethod
    def odds(self, match_id: str) -> list[OddsLine]: ...
