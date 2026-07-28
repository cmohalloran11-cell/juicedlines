"""
store — backend-agnostic persistence for Juiced 2.0's relational data.

Public surface:
    get_database()          → the process-wide Database (SQLite or Postgres via DATABASE_URL)
    init_schema(db)         → create the users/watchlists/portfolio tables (idempotent)
    UserRepository, WatchlistRepository, PortfolioRepository
"""
from .database import Database, get_database, reset_singleton
from .schema import init_schema
from .repositories import (
    UserRepository,
    WatchlistRepository,
    PortfolioRepository,
    AlertRepository,
)

__all__ = [
    "Database",
    "get_database",
    "reset_singleton",
    "init_schema",
    "UserRepository",
    "WatchlistRepository",
    "PortfolioRepository",
    "AlertRepository",
]
