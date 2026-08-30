"""
cfb.schema — DDL for the CFB data layer's relational tables.

Same portability rule as store/schema.py and fantasy/schema.py: UUID text keys, ISO-8601 text
timestamps, no dialect-specific types, so one set of statements is valid on SQLite and
Postgres. init_schema(db) is idempotent -- call it once at startup (main.py's lifespan,
alongside store.init_schema / fantasy.init_schema).
"""
from __future__ import annotations

from store.database import Database

_TABLES: tuple[str, ...] = (
    # All 134(+) FBS teams (+ any FCS/other classification CFBD returns, kept for schedule
    # opponent lookups -- an FBS team's opponent is sometimes FCS). Source-agnostic id.
    """
    CREATE TABLE IF NOT EXISTS cfb_teams (
        id             TEXT PRIMARY KEY,
        school         TEXT NOT NULL,
        conference     TEXT,
        classification TEXT,
        abbreviation   TEXT,
        cfbd_id        TEXT,
        created_at     TEXT NOT NULL,
        updated_at     TEXT NOT NULL
    )
    """,
    # Canonical player, source-agnostic -- every other CFB table keys off this id, never a
    # CFBD athlete id or an Odds API player name directly. Mirrors fantasy_players exactly
    # (see fantasy/schema.py) -- same problem, already solved once in this codebase.
    """
    CREATE TABLE IF NOT EXISTS cfb_players (
        id            TEXT PRIMARY KEY,
        full_name     TEXT NOT NULL,
        first_name    TEXT,
        last_name     TEXT,
        position      TEXT,
        team          TEXT,
        jersey        INTEGER,
        created_at    TEXT NOT NULL,
        updated_at    TEXT NOT NULL
    )
    """,
    # source -> source_id -> canonical player_id. PK on (source, source_id): a source's own id
    # (CFBD athlete id, or an Odds API player display-name string -- Odds API carries no
    # numeric player id) is only unique within that source.
    """
    CREATE TABLE IF NOT EXISTS cfb_player_ids (
        player_id   TEXT NOT NULL,
        source      TEXT NOT NULL,
        source_id   TEXT NOT NULL,
        confidence  REAL NOT NULL DEFAULT 1.0,
        matched_at  TEXT NOT NULL,
        PRIMARY KEY (source, source_id)
    )
    """,
    # Fuzzy-match fallback review log -- every source row that couldn't be mapped with
    # confidence, so a human resolves it instead of it silently vanishing or silently
    # mis-mapping to the wrong player. Same shape as fantasy_unmatched_players.
    """
    CREATE TABLE IF NOT EXISTS cfb_unmatched_players (
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
    # Manual, admin-editable player-status override -- no mandated CFB injury report exists
    # (unlike MLB/NFL's official feeds), so this is the only source of truth for "is this
    # player playing". `status` is free text (out | doubtful | questionable | probable |
    # available); `as_of` is when a human last confirmed it, which is what lets a caller near
    # kickoff decide a projection is stale (see cfb/player_status.py::is_stale).
    """
    CREATE TABLE IF NOT EXISTS cfb_player_status (
        player_id   TEXT PRIMARY KEY,
        status      TEXT NOT NULL,
        note        TEXT,
        set_by      TEXT,
        as_of       TEXT NOT NULL
    )
    """,
    # One row per external bulk-fetch source (CFBD teams/rosters today) -- same shape and
    # purpose as fantasy_players_sync_log, kept as its own table rather than sharing the
    # fantasy_ table so CFB stays cleanly addable/removable on its own.
    """
    CREATE TABLE IF NOT EXISTS cfb_sync_log (
        source       TEXT PRIMARY KEY,
        synced_at    TEXT NOT NULL,
        row_count    INTEGER
    )
    """,
)

_INDEXES: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_cfb_players_team ON cfb_players(team)",
    "CREATE INDEX IF NOT EXISTS idx_cfb_player_ids_player ON cfb_player_ids(player_id)",
    "CREATE INDEX IF NOT EXISTS idx_cfb_unmatched_status ON cfb_unmatched_players(status)",
    "CREATE INDEX IF NOT EXISTS idx_cfb_teams_school ON cfb_teams(school)",
)


def init_schema(db: Database) -> None:
    """Create the CFB data layer's tables + indexes if absent. Idempotent."""
    for ddl in _TABLES:
        db.execute(ddl)
    for ddl in _INDEXES:
        db.execute(ddl)
