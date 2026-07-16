"""
db.py — SQLite store for raw pulls, engineered features, projections & actuals.

Tables
------
raw_cache       : keyed blob cache of pulled source data (with TTL)
features        : engineered feature vector per player/game/sport (JSON)
projections     : a projection we produced (mean/floor/ceiling/... per stat)
actuals         : observed result for a player/game/stat (for backtesting)
backtest_runs   : summary metrics from a backtest run
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Iterable

from . import config

_LOCK = threading.Lock()


def connect() -> sqlite3.Connection:
    c = sqlite3.connect(config.db_path())
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db() -> None:
    with _LOCK, connect() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS raw_cache (
                key        TEXT PRIMARY KEY,
                source     TEXT,
                payload    TEXT,
                fetched_at REAL
            );
            CREATE TABLE IF NOT EXISTS features (
                sport      TEXT, player TEXT, game_id TEXT,
                vector     TEXT,            -- JSON dict of features
                built_at   REAL,
                PRIMARY KEY (sport, player, game_id)
            );
            CREATE TABLE IF NOT EXISTS projections (
                sport TEXT, player TEXT, game_id TEXT, stat TEXT,
                mean REAL, median REAL, p25 REAL, p75 REAL,
                floor REAL, ceiling REAL, std REAL,
                model TEXT, created_at REAL,
                PRIMARY KEY (sport, player, game_id, stat, model)
            );
            CREATE TABLE IF NOT EXISTS actuals (
                sport TEXT, player TEXT, game_id TEXT, stat TEXT,
                value REAL,
                PRIMARY KEY (sport, player, game_id, stat)
            );
            CREATE TABLE IF NOT EXISTS backtest_runs (
                run_id TEXT, sport TEXT, stat TEXT,
                n INTEGER, mae REAL, rmse REAL, bias REAL,
                calibration TEXT, created_at REAL,
                PRIMARY KEY (run_id, sport, stat)
            );
            """
        )
        c.commit()


# ── raw cache (TTL) ──────────────────────────────────────────────────────────

def cache_get(key: str, ttl_hours: float | None = None) -> Any | None:
    ttl = (ttl_hours if ttl_hours is not None
           else config.load()["paths"]["cache_ttl_hours"]) * 3600
    with _LOCK, connect() as c:
        row = c.execute("SELECT payload, fetched_at FROM raw_cache WHERE key=?",
                        (key,)).fetchone()
    if not row:
        return None
    if ttl >= 0 and time.time() - row["fetched_at"] > ttl:
        return None
    try:
        return json.loads(row["payload"])
    except Exception:
        return None


def cache_set(key: str, payload: Any, source: str = "") -> None:
    with _LOCK, connect() as c:
        c.execute("INSERT OR REPLACE INTO raw_cache VALUES (?,?,?,?)",
                  (key, source, json.dumps(payload, default=str), time.time()))
        c.commit()


# ── features / projections / actuals ─────────────────────────────────────────

def store_features(sport: str, player: str, game_id: str, vector: dict) -> None:
    with _LOCK, connect() as c:
        c.execute("INSERT OR REPLACE INTO features VALUES (?,?,?,?,?)",
                  (sport, player, game_id, json.dumps(vector, default=str), time.time()))
        c.commit()


def store_projection(sport: str, player: str, game_id: str, stat: str,
                     dist: dict, model: str = "ensemble") -> None:
    with _LOCK, connect() as c:
        c.execute(
            "INSERT OR REPLACE INTO projections VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (sport, player, game_id, stat, dist.get("mean"), dist.get("median"),
             dist.get("p25"), dist.get("p75"), dist.get("floor"),
             dist.get("ceiling"), dist.get("std"), model, time.time()))
        c.commit()


def store_actual(sport: str, player: str, game_id: str, stat: str, value: float) -> None:
    with _LOCK, connect() as c:
        c.execute("INSERT OR REPLACE INTO actuals VALUES (?,?,?,?,?)",
                  (sport, player, game_id, stat, value))
        c.commit()


def store_backtest(run_id: str, sport: str, stat: str, metrics: dict) -> None:
    with _LOCK, connect() as c:
        c.execute("INSERT OR REPLACE INTO backtest_runs VALUES (?,?,?,?,?,?,?,?,?)",
                  (run_id, sport, stat, metrics.get("n"), metrics.get("mae"),
                   metrics.get("rmse"), metrics.get("bias"),
                   json.dumps(metrics.get("calibration", [])), time.time()))
        c.commit()


def joined_projection_actuals(sport: str, model: str = "ensemble") -> list[dict]:
    """Projection-vs-actual rows for backtesting/calibration."""
    with _LOCK, connect() as c:
        rows = c.execute(
            """SELECT p.stat, p.mean, p.median, p.p25, p.p75, p.floor, p.ceiling,
                      p.std, a.value AS actual
               FROM projections p JOIN actuals a
                 ON p.sport=a.sport AND p.player=a.player
                AND p.game_id=a.game_id AND p.stat=a.stat
               WHERE p.sport=? AND p.model=?""",
            (sport, model)).fetchall()
    return [dict(r) for r in rows]
