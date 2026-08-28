"""
cfb.repositories — data access for the CFB data layer's relational tables.

Same conventions as fantasy/repositories.py: every method takes a Database, PKs are uuid4 hex
strings, timestamps are ISO-8601 UTC text, SQL uses `?` placeholders translated per-backend.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from store.database import Database


def _uid() -> str:
    return uuid.uuid4().hex


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TeamRepository:
    def __init__(self, db: Database):
        self.db = db

    def upsert(self, school: str, conference: Optional[str], classification: Optional[str],
              abbreviation: Optional[str], cfbd_id: Optional[str]) -> dict:
        existing = self.db.query("SELECT id FROM cfb_teams WHERE cfbd_id=?", (cfbd_id,)) \
            if cfbd_id else self.db.query("SELECT id FROM cfb_teams WHERE school=?", (school,))
        now = _now()
        if existing:
            tid = existing[0]["id"]
            self.db.execute(
                "UPDATE cfb_teams SET school=?, conference=?, classification=?, "
                "abbreviation=?, cfbd_id=?, updated_at=? WHERE id=?",
                (school, conference, classification, abbreviation, cfbd_id, now, tid))
            return self.get(tid)  # type: ignore[return-value]
        tid = _uid()
        self.db.execute(
            "INSERT INTO cfb_teams (id, school, conference, classification, abbreviation, "
            "cfbd_id, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (tid, school, conference, classification, abbreviation, cfbd_id, now, now))
        return self.get(tid)  # type: ignore[return-value]

    def get(self, team_id: str) -> Optional[dict]:
        rows = self.db.query("SELECT * FROM cfb_teams WHERE id=?", (team_id,))
        return rows[0] if rows else None

    def all(self, classification: Optional[str] = None) -> list[dict]:
        if classification:
            return self.db.query(
                "SELECT * FROM cfb_teams WHERE classification=? ORDER BY school", (classification,))
        return self.db.query("SELECT * FROM cfb_teams ORDER BY school", ())


class PlayerRepository:
    """The canonical player table. Callers resolve identity via PlayerMappingRepository BEFORE
    deciding whether to create a new canonical player or update an existing one (see
    cfb.player_matching) -- this repository never guesses identity itself."""

    def __init__(self, db: Database):
        self.db = db

    def create(self, full_name: str, first_name: Optional[str] = None,
              last_name: Optional[str] = None, position: Optional[str] = None,
              team: Optional[str] = None, jersey: Optional[int] = None) -> dict:
        pid = _uid()
        now = _now()
        self.db.execute(
            "INSERT INTO cfb_players (id, full_name, first_name, last_name, position, team, "
            "jersey, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (pid, full_name, first_name, last_name, position, team, jersey, now, now))
        return self.get(pid)  # type: ignore[return-value]

    def get(self, player_id: str) -> Optional[dict]:
        rows = self.db.query("SELECT * FROM cfb_players WHERE id=?", (player_id,))
        return rows[0] if rows else None

    def update(self, player_id: str, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE cfb_players SET {cols}, updated_at=? WHERE id=?",
                        (*fields.values(), _now(), player_id))

    def find_by_position(self, position: str) -> list[dict]:
        return self.db.query("SELECT * FROM cfb_players WHERE position=?", (position,))

    def find_by_team(self, team: str) -> list[dict]:
        return self.db.query("SELECT * FROM cfb_players WHERE team=?", (team,))

    def all(self) -> list[dict]:
        return self.db.query("SELECT * FROM cfb_players", ())


class PlayerMappingRepository:
    def __init__(self, db: Database):
        self.db = db

    def resolve(self, source: str, source_id: str) -> Optional[str]:
        rows = self.db.query(
            "SELECT player_id FROM cfb_player_ids WHERE source=? AND source_id=?",
            (source, source_id))
        return rows[0]["player_id"] if rows else None

    def map(self, player_id: str, source: str, source_id: str, confidence: float = 1.0) -> None:
        existing = self.db.query(
            "SELECT 1 FROM cfb_player_ids WHERE source=? AND source_id=?", (source, source_id))
        if existing:
            self.db.execute(
                "UPDATE cfb_player_ids SET player_id=?, confidence=?, matched_at=? "
                "WHERE source=? AND source_id=?",
                (player_id, confidence, _now(), source, source_id))
            return
        self.db.execute(
            "INSERT INTO cfb_player_ids (player_id, source, source_id, confidence, matched_at) "
            "VALUES (?,?,?,?,?)", (player_id, source, source_id, confidence, _now()))

    def ids_for_player(self, player_id: str) -> list[dict]:
        return self.db.query("SELECT * FROM cfb_player_ids WHERE player_id=?", (player_id,))


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
            "INSERT INTO cfb_unmatched_players (id, source, source_id, raw_name, raw_team, "
            "raw_position, best_guess_player_id, best_guess_score, status, created_at, resolved_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            tuple(row[k] for k in ("id", "source", "source_id", "raw_name", "raw_team",
                                   "raw_position", "best_guess_player_id", "best_guess_score",
                                   "status", "created_at", "resolved_at")))
        return row

    def list_pending(self) -> list[dict]:
        return self.db.query(
            "SELECT * FROM cfb_unmatched_players WHERE status='pending' ORDER BY created_at DESC", ())

    def resolve(self, unmatched_id: str, player_id: str) -> bool:
        rows = self.db.query("SELECT * FROM cfb_unmatched_players WHERE id=?", (unmatched_id,))
        if not rows:
            return False
        row = rows[0]
        PlayerMappingRepository(self.db).map(
            player_id, row["source"], row["source_id"] or "", confidence=1.0)
        self.db.execute(
            "UPDATE cfb_unmatched_players SET status='resolved', resolved_at=? WHERE id=?",
            (_now(), unmatched_id))
        return True

    def ignore(self, unmatched_id: str) -> bool:
        rows = self.db.query("SELECT 1 FROM cfb_unmatched_players WHERE id=?", (unmatched_id,))
        if not rows:
            return False
        self.db.execute(
            "UPDATE cfb_unmatched_players SET status='ignored', resolved_at=? WHERE id=?",
            (_now(), unmatched_id))
        return True


class PlayerStatusRepository:
    """Manual admin override, one row per player (upsert-only -- there is no history of past
    statuses, only the current one and when it was last confirmed). See cfb.player_status for
    the staleness rule this backs."""

    def __init__(self, db: Database):
        self.db = db

    def set(self, player_id: str, status: str, note: Optional[str] = None,
           set_by: Optional[str] = None) -> dict:
        now = _now()
        existing = self.db.query("SELECT 1 FROM cfb_player_status WHERE player_id=?", (player_id,))
        if existing:
            self.db.execute(
                "UPDATE cfb_player_status SET status=?, note=?, set_by=?, as_of=? "
                "WHERE player_id=?", (status, note, set_by, now, player_id))
        else:
            self.db.execute(
                "INSERT INTO cfb_player_status (player_id, status, note, set_by, as_of) "
                "VALUES (?,?,?,?,?)", (player_id, status, note, set_by, now))
        return {"player_id": player_id, "status": status, "note": note, "set_by": set_by, "as_of": now}

    def get(self, player_id: str) -> Optional[dict]:
        rows = self.db.query("SELECT * FROM cfb_player_status WHERE player_id=?", (player_id,))
        return rows[0] if rows else None

    def all(self) -> list[dict]:
        return self.db.query("SELECT * FROM cfb_player_status ORDER BY as_of DESC", ())


class SyncLogRepository:
    def __init__(self, db: Database):
        self.db = db

    def last_synced(self, source: str) -> Optional[str]:
        rows = self.db.query("SELECT synced_at FROM cfb_sync_log WHERE source=?", (source,))
        return rows[0]["synced_at"] if rows else None

    def is_stale(self, source: str, min_hours: float = 20.0) -> bool:
        last = self.last_synced(source)
        if not last:
            return True
        try:
            last_dt = datetime.fromisoformat(last)
        except ValueError:
            return True
        age_hours = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600.0
        return age_hours >= min_hours

    def record(self, source: str, row_count: int) -> None:
        existing = self.db.query("SELECT 1 FROM cfb_sync_log WHERE source=?", (source,))
        if existing:
            self.db.execute(
                "UPDATE cfb_sync_log SET synced_at=?, row_count=? WHERE source=?",
                (_now(), row_count, source))
        else:
            self.db.execute(
                "INSERT INTO cfb_sync_log (source, synced_at, row_count) VALUES (?,?,?)",
                (source, _now(), row_count))
