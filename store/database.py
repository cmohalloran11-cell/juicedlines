"""
store.database — backend-agnostic database layer (SQLite now, Postgres-ready).

Why this exists
---------------
The legacy line-movement + CLV ledger lives in the top-level db.py on SQLite and works
well, so it is intentionally left alone. This package is the home for the NEW relational
data of Juiced 2.0 — users, watchlists, portfolios — written against a small interface so
the same repository code runs on SQLite today and Postgres in production with only an env
change (DATABASE_URL). That de-risks the migration the audit called for: no call site has
to change when the backend does.

Portability strategy
--------------------
Repositories write SQL with `?` placeholders; the adapter translates to the backend's
paramstyle. Primary keys are Python-generated UUID strings and timestamps are ISO-8601
text, so a single DDL (see schema.py) is valid on both engines — the usual SQLite/Postgres
type mismatches (SERIAL vs UUID, AUTOINCREMENT, TIMESTAMPTZ) simply don't arise.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Sequence


class Database(ABC):
    """Minimal interface every backend adapter implements."""

    placeholder: str = "?"

    def q(self, sql: str) -> str:
        """Translate `?` placeholders to the backend's paramstyle."""
        return sql if self.placeholder == "?" else sql.replace("?", self.placeholder)

    @abstractmethod
    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict]:
        """Run a SELECT and return rows as dicts."""

    @abstractmethod
    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        """Run a single write statement and commit."""

    @abstractmethod
    def executemany(self, sql: str, seq: Iterable[Sequence[Any]]) -> None:
        """Run a write statement over many parameter tuples and commit."""

    @property
    @abstractmethod
    def dialect(self) -> str:
        ...


class SQLiteDatabase(Database):
    """Default local backend. Connect-per-call + a process lock, mirroring db.py's style
    (SQLite serializes writers itself, so this is safe under the background loop)."""

    placeholder = "?"

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._lock = threading.Lock()

    @property
    def dialect(self) -> str:
        return "sqlite"

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict]:
        with self._lock, self._conn() as c:
            return [dict(r) for r in c.execute(self.q(sql), tuple(params)).fetchall()]

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        with self._lock, self._conn() as c:
            c.execute(self.q(sql), tuple(params))
            c.commit()

    def executemany(self, sql: str, seq: Iterable[Sequence[Any]]) -> None:
        with self._lock, self._conn() as c:
            c.executemany(self.q(sql), [tuple(x) for x in seq])
            c.commit()


class PostgresDatabase(Database):
    """Production backend (psycopg 3). Kept behind the same interface so switching is a
    DATABASE_URL change. Lazily imported so SQLite-only environments need no psycopg."""

    placeholder = "%s"

    def __init__(self, url: str):
        try:
            import psycopg  # noqa: F401  (import-time availability check)
        except Exception as exc:  # pragma: no cover - only hit without psycopg installed
            raise RuntimeError(
                "DATABASE_URL points at Postgres but psycopg is not installed. "
                "`pip install 'psycopg[binary]'` or unset DATABASE_URL to use SQLite."
            ) from exc
        self.url = url
        self._lock = threading.Lock()
        self._conn_obj = None

    @property
    def dialect(self) -> str:
        return "postgres"

    def _conn(self):
        import psycopg
        if self._conn_obj is None or self._conn_obj.closed:
            self._conn_obj = psycopg.connect(self.url, autocommit=True)
        return self._conn_obj

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict]:
        from psycopg.rows import dict_row
        with self._lock, self._conn().cursor(row_factory=dict_row) as cur:
            cur.execute(self.q(sql), tuple(params))
            return [dict(r) for r in cur.fetchall()]

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        with self._lock, self._conn().cursor() as cur:
            cur.execute(self.q(sql), tuple(params))

    def executemany(self, sql: str, seq: Iterable[Sequence[Any]]) -> None:
        with self._lock, self._conn().cursor() as cur:
            cur.executemany(self.q(sql), [tuple(x) for x in seq])


# ── factory ───────────────────────────────────────────────────────────────────

_DEFAULT_SQLITE = Path(__file__).resolve().parent.parent / "juiced.db"
_singleton: Database | None = None


def get_database(url: str | None = None) -> Database:
    """
    Return the process-wide Database, chosen from DATABASE_URL:
      • postgres:// or postgresql://  → PostgresDatabase
      • anything else / unset          → SQLiteDatabase (store/../juiced.db)
    Cached after first call.
    """
    global _singleton
    if _singleton is not None and url is None:
        return _singleton
    resolved = url if url is not None else os.getenv("DATABASE_URL", "")
    if resolved.startswith(("postgres://", "postgresql://")):
        db: Database = PostgresDatabase(resolved)
    else:
        db = SQLiteDatabase(resolved or _DEFAULT_SQLITE)
    if url is None:
        _singleton = db
    return db


def reset_singleton() -> None:
    """Test hook — drop the cached database so the next get_database() re-resolves."""
    global _singleton
    _singleton = None
