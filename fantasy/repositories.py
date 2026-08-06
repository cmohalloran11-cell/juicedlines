"""
fantasy.repositories — data access for the fantasy draft assistant's relational tables.

Same conventions as store/repositories.py: every method takes a Database, PKs are uuid4 hex
strings, timestamps are ISO-8601 UTC text, SQL uses `?` placeholders translated per-backend.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from store.database import Database


def _uid() -> str:
    return uuid.uuid4().hex


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PlayerRepository:
    """The canonical player table. Callers resolve identity via PlayerMappingRepository
    BEFORE deciding whether to create a new canonical player or update an existing one (see
    fantasy.players_sync / fantasy.player_matching) -- this repository never guesses identity
    itself."""

    def __init__(self, db: Database):
        self.db = db

    def create(self, full_name: str, first_name: Optional[str] = None,
              last_name: Optional[str] = None, position: Optional[str] = None,
              team: Optional[str] = None, status: Optional[str] = None,
              birth_date: Optional[str] = None) -> dict:
        pid = _uid()
        now = _now()
        self.db.execute(
            "INSERT INTO fantasy_players (id, full_name, first_name, last_name, position, "
            "team, status, birth_date, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (pid, full_name, first_name, last_name, position, team, status, birth_date, now, now))
        return self.get(pid)  # type: ignore[return-value]

    def get(self, player_id: str) -> Optional[dict]:
        rows = self.db.query("SELECT * FROM fantasy_players WHERE id=?", (player_id,))
        return rows[0] if rows else None

    def update(self, player_id: str, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE fantasy_players SET {cols}, updated_at=? WHERE id=?",
                        (*fields.values(), _now(), player_id))

    def find_by_position(self, position: str) -> list[dict]:
        return self.db.query("SELECT * FROM fantasy_players WHERE position=?", (position,))

    def all(self) -> list[dict]:
        return self.db.query("SELECT * FROM fantasy_players", ())


class PlayerMappingRepository:
    def __init__(self, db: Database):
        self.db = db

    def resolve(self, source: str, source_id: str) -> Optional[str]:
        """Canonical player_id for a source's id, or None if unmapped."""
        rows = self.db.query(
            "SELECT player_id FROM fantasy_player_ids WHERE source=? AND source_id=?",
            (source, source_id))
        return rows[0]["player_id"] if rows else None

    def map(self, player_id: str, source: str, source_id: str, confidence: float = 1.0) -> None:
        """Link a source id to a canonical player. Idempotent -- remapping the same
        source/source_id just refreshes confidence + matched_at."""
        existing = self.db.query(
            "SELECT 1 FROM fantasy_player_ids WHERE source=? AND source_id=?", (source, source_id))
        if existing:
            self.db.execute(
                "UPDATE fantasy_player_ids SET player_id=?, confidence=?, matched_at=? "
                "WHERE source=? AND source_id=?",
                (player_id, confidence, _now(), source, source_id))
            return
        self.db.execute(
            "INSERT INTO fantasy_player_ids (player_id, source, source_id, confidence, matched_at) "
            "VALUES (?,?,?,?,?)", (player_id, source, source_id, confidence, _now()))

    def ids_for_player(self, player_id: str) -> list[dict]:
        return self.db.query("SELECT * FROM fantasy_player_ids WHERE player_id=?", (player_id,))


class UnmatchedPlayerRepository:
    def __init__(self, db: Database):
        self.db = db

    def log(self, source: str, source_id: Optional[str], raw_name: str,
            raw_team: Optional[str] = None, raw_position: Optional[str] = None,
            best_guess_player_id: Optional[str] = None,
            best_guess_score: Optional[float] = None) -> dict:
        uid = _uid()
        row = {
            "id": uid, "source": source, "source_id": source_id, "raw_name": raw_name,
            "raw_team": raw_team, "raw_position": raw_position,
            "best_guess_player_id": best_guess_player_id, "best_guess_score": best_guess_score,
            "status": "pending", "created_at": _now(), "resolved_at": None,
        }
        self.db.execute(
            "INSERT INTO fantasy_unmatched_players (id, source, source_id, raw_name, raw_team, "
            "raw_position, best_guess_player_id, best_guess_score, status, created_at, resolved_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            tuple(row[k] for k in ("id", "source", "source_id", "raw_name", "raw_team",
                                   "raw_position", "best_guess_player_id", "best_guess_score",
                                   "status", "created_at", "resolved_at")))
        return row

    def list_pending(self) -> list[dict]:
        return self.db.query(
            "SELECT * FROM fantasy_unmatched_players WHERE status='pending' "
            "ORDER BY created_at DESC", ())

    def resolve(self, unmatched_id: str, player_id: str) -> bool:
        """A human confirms the match: link it in fantasy_player_ids and close the log entry."""
        rows = self.db.query("SELECT * FROM fantasy_unmatched_players WHERE id=?", (unmatched_id,))
        if not rows:
            return False
        row = rows[0]
        PlayerMappingRepository(self.db).map(
            player_id, row["source"], row["source_id"] or "", confidence=1.0)
        self.db.execute(
            "UPDATE fantasy_unmatched_players SET status='resolved', resolved_at=? WHERE id=?",
            (_now(), unmatched_id))
        return True

    def ignore(self, unmatched_id: str) -> bool:
        rows = self.db.query("SELECT 1 FROM fantasy_unmatched_players WHERE id=?", (unmatched_id,))
        if not rows:
            return False
        self.db.execute(
            "UPDATE fantasy_unmatched_players SET status='ignored', resolved_at=? WHERE id=?",
            (_now(), unmatched_id))
        return True


class SyncLogRepository:
    """Generic "last successful external sync" log -- table name says players_sync_log for
    historical reasons (the players dump was the first caller) but the schema (source TEXT PK,
    synced_at, player_count) is generic and is also used for the projections provider sync."""

    def __init__(self, db: Database):
        self.db = db

    def last_synced(self, source: str) -> Optional[str]:
        rows = self.db.query(
            "SELECT synced_at FROM fantasy_players_sync_log WHERE source=?", (source,))
        return rows[0]["synced_at"] if rows else None

    def is_stale(self, source: str, min_hours: float = 20.0) -> bool:
        """True if `source` has never synced or its last sync is older than min_hours. Shared
        by every "call this external source at most once a day" gate (Sleeper's players dump,
        the projections provider)."""
        last = self.last_synced(source)
        if not last:
            return True
        try:
            last_dt = datetime.fromisoformat(last)
        except ValueError:
            return True
        age_hours = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600.0
        return age_hours >= min_hours

    def record(self, source: str, player_count: int) -> None:
        existing = self.db.query("SELECT 1 FROM fantasy_players_sync_log WHERE source=?", (source,))
        if existing:
            self.db.execute(
                "UPDATE fantasy_players_sync_log SET synced_at=?, player_count=? WHERE source=?",
                (_now(), player_count, source))
        else:
            self.db.execute(
                "INSERT INTO fantasy_players_sync_log (source, synced_at, player_count) "
                "VALUES (?,?,?)", (source, _now(), player_count))


class LeagueRepository:
    def __init__(self, db: Database):
        self.db = db

    def upsert(self, user_id: str, sleeper_league_id: str, name: Optional[str],
              season: Optional[str], league_size: Optional[int], scoring_settings: dict,
              roster_positions: list[str]) -> dict:
        existing = self.db.query(
            "SELECT id FROM fantasy_leagues WHERE user_id=? AND sleeper_league_id=?",
            (user_id, sleeper_league_id))
        now = _now()
        if existing:
            lid = existing[0]["id"]
            self.db.execute(
                "UPDATE fantasy_leagues SET name=?, season=?, league_size=?, scoring_settings=?, "
                "roster_positions=?, updated_at=? WHERE id=?",
                (name, season, league_size, json.dumps(scoring_settings),
                 json.dumps(roster_positions), now, lid))
            return self.get(user_id, lid)  # type: ignore[return-value]
        lid = _uid()
        self.db.execute(
            "INSERT INTO fantasy_leagues (id, user_id, sleeper_league_id, name, season, "
            "league_size, scoring_settings, roster_positions, imported_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (lid, user_id, sleeper_league_id, name, season, league_size,
             json.dumps(scoring_settings), json.dumps(roster_positions), now, now))
        return self.get(user_id, lid)  # type: ignore[return-value]

    def get(self, user_id: str, league_id: str) -> Optional[dict]:
        rows = self.db.query(
            "SELECT * FROM fantasy_leagues WHERE id=? AND user_id=?", (league_id, user_id))
        if not rows:
            return None
        return self._deserialize(rows[0])

    def list_for_user(self, user_id: str) -> list[dict]:
        rows = self.db.query(
            "SELECT * FROM fantasy_leagues WHERE user_id=? ORDER BY imported_at DESC", (user_id,))
        return [self._deserialize(r) for r in rows]

    @staticmethod
    def _deserialize(row: dict) -> dict:
        row = dict(row)
        row["scoring_settings"] = json.loads(row["scoring_settings"]) if row.get("scoring_settings") else {}
        row["roster_positions"] = json.loads(row["roster_positions"]) if row.get("roster_positions") else []
        return row


class ProjectionRepository:
    def __init__(self, db: Database):
        self.db = db

    def upsert(self, player_id: str, provider: str, season: int, week: Optional[int],
              stats: dict, methodology: Optional[str]) -> dict:
        """Explicit find-then-update/insert (not a DB unique constraint -- see schema.py's
        note on why `week IS NULL` can't be de-duplicated by an index)."""
        existing = self._find(player_id, provider, season, week)
        now = _now()
        if existing:
            self.db.execute(
                "UPDATE fantasy_projections SET stats=?, methodology=?, fetched_at=? WHERE id=?",
                (json.dumps(stats), methodology, now, existing["id"]))
            existing.update(stats=stats, methodology=methodology, fetched_at=now)
            return existing
        pid = _uid()
        self.db.execute(
            "INSERT INTO fantasy_projections (id, player_id, provider, season, week, stats, "
            "methodology, fetched_at) VALUES (?,?,?,?,?,?,?,?)",
            (pid, player_id, provider, season, week, json.dumps(stats), methodology, now))
        return {"id": pid, "player_id": player_id, "provider": provider, "season": season,
                "week": week, "stats": stats, "methodology": methodology, "fetched_at": now}

    def _find(self, player_id: str, provider: str, season: int,
             week: Optional[int]) -> Optional[dict]:
        if week is None:
            rows = self.db.query(
                "SELECT * FROM fantasy_projections WHERE player_id=? AND provider=? AND "
                "season=? AND week IS NULL", (player_id, provider, season))
        else:
            rows = self.db.query(
                "SELECT * FROM fantasy_projections WHERE player_id=? AND provider=? AND "
                "season=? AND week=?", (player_id, provider, season, week))
        if not rows:
            return None
        row = dict(rows[0])
        row["stats"] = json.loads(row["stats"]) if row.get("stats") else {}
        return row

    def for_season(self, provider: str, season: int, week: Optional[int] = None) -> list[dict]:
        if week is None:
            rows = self.db.query(
                "SELECT * FROM fantasy_projections WHERE provider=? AND season=? AND week IS NULL",
                (provider, season))
        else:
            rows = self.db.query(
                "SELECT * FROM fantasy_projections WHERE provider=? AND season=? AND week=?",
                (provider, season, week))
        out = []
        for r in rows:
            r = dict(r)
            r["stats"] = json.loads(r["stats"]) if r.get("stats") else {}
            out.append(r)
        return out
