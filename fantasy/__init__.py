"""
fantasy — data layer for the fantasy football draft assistant (Sleeper leagues + a swappable
projections provider) and everything built on it: draft board (VOR + tiering, draft_state),
lineup optimizer (lineup), waiver wire (composed from vor + draft_state in routes_fantasy.py),
and trade analyzer (trade). All of it shares the same canonical player table, league config,
and scoring engine -- see routes_fantasy.py for the HTTP surface.

Public surface:
    init_schema(db)          -> create the fantasy_* tables (idempotent)
    PlayerRepository, PlayerMappingRepository, UnmatchedPlayerRepository,
    SyncLogRepository, LeagueRepository, ProjectionRepository
"""
from .schema import init_schema
from .repositories import (
    PlayerRepository,
    PlayerMappingRepository,
    UnmatchedPlayerRepository,
    SyncLogRepository,
    LeagueRepository,
    ProjectionRepository,
)

__all__ = [
    "init_schema",
    "PlayerRepository",
    "PlayerMappingRepository",
    "UnmatchedPlayerRepository",
    "SyncLogRepository",
    "LeagueRepository",
    "ProjectionRepository",
]
