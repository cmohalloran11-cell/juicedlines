"""
fantasy.schema — DDL for the fantasy draft assistant's relational tables.

Same portability rule as store/schema.py: UUID text keys, ISO-8601 text timestamps, no
dialect-specific types, so a single set of statements is valid on SQLite and Postgres.
init_schema(db) is idempotent -- call it once at startup (main.py's lifespan, alongside
store.init_schema).
"""
from __future__ import annotations

from store.database import Database

_TABLES: tuple[str, ...] = (
    # Canonical player, source-agnostic. Every other fantasy table keys off this id, never a
    # Sleeper id or GSIS id directly -- see fantasy_player_ids for the mapping layer.
    """
    CREATE TABLE IF NOT EXISTS fantasy_players (
        id            TEXT PRIMARY KEY,
        full_name     TEXT NOT NULL,
        first_name    TEXT,
        last_name     TEXT,
        position      TEXT,
        team          TEXT,
        status        TEXT,
        birth_date    TEXT,
        created_at    TEXT NOT NULL,
        updated_at    TEXT NOT NULL
    )
    """,
    # source -> source_id -> canonical player_id. PK on (source, source_id) since a source's
    # own id is only unique within that source; a player can (and usually does) exist in more
    # than one source under a different id.
    """
    CREATE TABLE IF NOT EXISTS fantasy_player_ids (
        player_id   TEXT NOT NULL,
        source      TEXT NOT NULL,
        source_id   TEXT NOT NULL,
        confidence  REAL NOT NULL DEFAULT 1.0,
        matched_at  TEXT NOT NULL,
        PRIMARY KEY (source, source_id)
    )
    """,
    # Fuzzy-match fallback review log: every source row that couldn't be mapped with
    # confidence, so a human resolves it instead of it silently vanishing or silently
    # mis-mapping to the wrong player.
    """
    CREATE TABLE IF NOT EXISTS fantasy_unmatched_players (
        id                    TEXT PRIMARY KEY,
        source                TEXT NOT NULL,
        source_id             TEXT,
        raw_name              TEXT,
        raw_team              TEXT,
        raw_position          TEXT,
        best_guess_player_id  TEXT,
        best_guess_score      REAL,
        status                TEXT NOT NULL DEFAULT 'pending',  -- pending | resolved | ignored
        created_at            TEXT NOT NULL,
        resolved_at           TEXT
    )
    """,
    # One row per external bulk-fetch source, so the "call at most once a day" constraint
    # (Sleeper's /v1/players/nfl) survives process restarts, not just in-memory state.
    """
    CREATE TABLE IF NOT EXISTS fantasy_players_sync_log (
        source       TEXT PRIMARY KEY,
        synced_at    TEXT NOT NULL,
        player_count INTEGER
    )
    """,
    # A user's imported Sleeper league: scoring rules + roster shape, the two inputs every
    # downstream computation (scoring, VOR, draft board) needs and must never hardcode.
    """
    CREATE TABLE IF NOT EXISTS fantasy_leagues (
        id                 TEXT PRIMARY KEY,
        user_id            TEXT NOT NULL,
        sleeper_league_id  TEXT NOT NULL,
        name               TEXT,
        season             TEXT,
        league_size        INTEGER,
        scoring_settings   TEXT NOT NULL,   -- JSON: Sleeper's native stat_key -> points
        roster_positions   TEXT NOT NULL,   -- JSON array of slot codes
        imported_at        TEXT NOT NULL,
        updated_at         TEXT NOT NULL
    )
    """,
    # Raw provider stat projections in canonical stat keys (see
    # fantasy.projections.CANONICAL_STAT_KEYS) -- scoring.py turns these into league-specific
    # fantasy_points at read time, never stored pre-scored, so one row serves every league
    # regardless of its rules.
    """
    CREATE TABLE IF NOT EXISTS fantasy_projections (
        id                 TEXT PRIMARY KEY,
        player_id          TEXT NOT NULL,
        provider           TEXT NOT NULL,
        season             INTEGER NOT NULL,
        week               INTEGER,          -- NULL = season-long (redraft) projection
        stats              TEXT NOT NULL,    -- JSON: canonical stat key -> value
        methodology        TEXT,
        fetched_at         TEXT NOT NULL
    )
    """,
)

_INDEXES: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_fantasy_player_ids_player ON fantasy_player_ids(player_id)",
    "CREATE INDEX IF NOT EXISTS idx_fantasy_unmatched_status ON fantasy_unmatched_players(status)",
    "CREATE INDEX IF NOT EXISTS idx_fantasy_leagues_user ON fantasy_leagues(user_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_fantasy_leagues_user_sleeper "
    "ON fantasy_leagues(user_id, sleeper_league_id)",
    # Not UNIQUE: `week IS NULL` doesn't collide under SQL NULL semantics on either backend,
    # so the season-long (week=NULL) case can't be de-duplicated by an index alone --
    # ProjectionRepository.upsert does an explicit find-then-update/insert instead.
    "CREATE INDEX IF NOT EXISTS idx_fantasy_projections_lookup "
    "ON fantasy_projections(provider, season, week)",
)


def init_schema(db: Database) -> None:
    """Create the fantasy draft assistant's tables + indexes if absent. Idempotent."""
    for ddl in _TABLES:
        db.execute(ddl)
    for ddl in _INDEXES:
        db.execute(ddl)
