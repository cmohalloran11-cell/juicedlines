"""
fantasy — data layer for the fantasy football draft assistant (Sleeper leagues + a swappable
projections provider). Phase 1: canonical player/mapping tables, league import, a pure scoring
engine, VOR + tiering, and live-draft state -- see routes_fantasy.py for the HTTP surface.

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
