"""
store.schema — DDL for the Juiced 2.0 relational tables (users, watchlists, portfolio).

A single set of CREATE TABLE IF NOT EXISTS statements valid on BOTH SQLite and Postgres:
all keys are UUID text, all timestamps are ISO-8601 text, so no dialect-specific types are
needed. init_schema() is idempotent and safe to call on every startup.

The legacy line_history / prop_clv tables are NOT defined here — they remain owned by the
top-level db.py until that ledger is migrated onto this layer in a later phase.
"""
from __future__ import annotations

from .database import Database

# NOTE: keep column types to TEXT / REAL / INTEGER only (portable across both engines).
_TABLES: tuple[str, ...] = (
    # Mirrors the authenticated identity (id == Supabase auth uid). Upserted on first
    # authenticated request so the rest of the schema can foreign-key to a local row.
    """
    CREATE TABLE IF NOT EXISTS users (
        id            TEXT PRIMARY KEY,
        email         TEXT UNIQUE,
        display_name  TEXT,
        role          TEXT NOT NULL DEFAULT 'USER',   -- USER | ADMIN | SUPER_ADMIN
        tier          TEXT NOT NULL DEFAULT 'FREE',   -- FREE | ELITE
        created_at    TEXT NOT NULL,
        updated_at    TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS watchlists (
        id          TEXT PRIMARY KEY,
        user_id     TEXT NOT NULL,
        name        TEXT NOT NULL,
        created_at  TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS watchlist_items (
        id            TEXT PRIMARY KEY,
        watchlist_id  TEXT NOT NULL,
        line_id       TEXT NOT NULL,
        sport         TEXT,
        player        TEXT,
        stat_type     TEXT,
        note          TEXT,
        created_at    TEXT NOT NULL
    )
    """,
    # A tracked prop position (research portfolio, not a real-money bet). status transitions
    # open → won | lost | void when the underlying prop settles.
    """
    CREATE TABLE IF NOT EXISTS portfolio_entries (
        id          TEXT PRIMARY KEY,
        user_id     TEXT NOT NULL,
        line_id     TEXT NOT NULL,
        sport       TEXT,
        player      TEXT,
        stat_type   TEXT,
        side        TEXT NOT NULL,                 -- over | under
        line_value  REAL,
        stake       REAL NOT NULL DEFAULT 1.0,
        odds        REAL,
        status      TEXT NOT NULL DEFAULT 'open',  -- open | won | lost | void
        result      REAL,
        placed_at   TEXT NOT NULL,
        settled_at  TEXT
    )
    """,
    # Model registry (spec Ch 178): one row per recorded backtest run, so a model version's
    # measured accuracy is tracked over time and versions are comparable.
    """
    CREATE TABLE IF NOT EXISTS model_runs (
        id            TEXT PRIMARY KEY,
        model_version TEXT,
        sport         TEXT,
        kind          TEXT,          -- baseline | replay
        n             INTEGER,
        mae           REAL,
        rmse          REAL,
        bias          REAL,
        hit_rate      REAL,
        ece           REAL,
        coverage      REAL,
        recorded_at   TEXT NOT NULL
    )
    """,
    # A user alert: fire when a prop's metric crosses a threshold (spec Ch 35).
    """
    CREATE TABLE IF NOT EXISTS alerts (
        id            TEXT PRIMARY KEY,
        user_id       TEXT NOT NULL,
        line_id       TEXT,
        player        TEXT,
        stat_type     TEXT,
        metric        TEXT NOT NULL,               -- edge | juice | probability | line_move
        comparator    TEXT NOT NULL DEFAULT 'gte', -- gte | lte
        threshold     REAL NOT NULL,
        active        INTEGER NOT NULL DEFAULT 1,
        created_at    TEXT NOT NULL,
        triggered_at  TEXT
    )
    """,
)

_INDEXES: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_watchlists_user ON watchlists(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_watchlist_items_wl ON watchlist_items(watchlist_id)",
    "CREATE INDEX IF NOT EXISTS idx_portfolio_user ON portfolio_entries(user_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_alerts_user ON alerts(user_id, active)",
    "CREATE INDEX IF NOT EXISTS idx_model_runs ON model_runs(sport, recorded_at)",
)


def init_schema(db: Database) -> None:
    """Create the Juiced 2.0 relational tables + indexes if absent. Idempotent."""
    for ddl in _TABLES:
        db.execute(ddl)
    for ddl in _INDEXES:
        db.execute(ddl)
