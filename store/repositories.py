"""
store.repositories — data access for users, watchlists, and portfolio entries.

Backend-agnostic: every method uses the Database interface, so these run unchanged on
SQLite or Postgres. PKs are uuid4 hex strings; timestamps are ISO-8601 UTC text.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from .database import Database


def _uid() -> str:
    return uuid.uuid4().hex


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class UserRepository:
    def __init__(self, db: Database):
        self.db = db

    def upsert(self, user_id: str, email: str | None,
               display_name: str | None = None) -> dict:
        """Insert the identity on first sight, else refresh mutable fields. Keyed by the
        Supabase auth uid. Role/tier are preserved across upserts (never downgraded here)."""
        now = _now()
        existing = self.get(user_id)
        if existing:
            self.db.execute(
                "UPDATE users SET email=?, display_name=COALESCE(?, display_name), "
                "updated_at=? WHERE id=?",
                (email, display_name, now, user_id))
            return self.get(user_id) or existing
        self.db.execute(
            "INSERT INTO users (id, email, display_name, role, tier, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (user_id, email, display_name, "USER", "FREE", now, now))
        return self.get(user_id)  # type: ignore[return-value]

    def get(self, user_id: str) -> dict | None:
        rows = self.db.query("SELECT * FROM users WHERE id=?", (user_id,))
        return rows[0] if rows else None

    def set_tier(self, user_id: str, tier: str) -> None:
        self.db.execute("UPDATE users SET tier=?, updated_at=? WHERE id=?",
                        (tier, _now(), user_id))


class AiUsageRepository:
    """Daily AI Juice call count per user (UTC calendar day) — backs the enforced free/Pro
    rate limit. Same check-then-write pattern as UserRepository.upsert (this codebase's
    existing style here; not a hardened atomic increment, but a double-click race costs at
    worst one extra counted call, not a security issue)."""
    def __init__(self, db: Database):
        self.db = db

    def get_count(self, user_id: str, usage_date: str) -> int:
        rows = self.db.query(
            "SELECT count FROM ai_usage WHERE user_id=? AND usage_date=?", (user_id, usage_date))
        return rows[0]["count"] if rows else 0

    def increment(self, user_id: str, usage_date: str) -> int:
        """Bump today's count and return the new total."""
        existing = self.db.query(
            "SELECT count FROM ai_usage WHERE user_id=? AND usage_date=?", (user_id, usage_date))
        if not existing:
            self.db.execute(
                "INSERT INTO ai_usage (user_id, usage_date, count, updated_at) VALUES (?,?,1,?)",
                (user_id, usage_date, _now()))
            return 1
        new_count = existing[0]["count"] + 1
        self.db.execute(
            "UPDATE ai_usage SET count=?, updated_at=? WHERE user_id=? AND usage_date=?",
            (new_count, _now(), user_id, usage_date))
        return new_count


class WatchlistRepository:
    def __init__(self, db: Database):
        self.db = db

    def create(self, user_id: str, name: str) -> dict:
        wid = _uid()
        self.db.execute(
            "INSERT INTO watchlists (id, user_id, name, created_at) VALUES (?,?,?,?)",
            (wid, user_id, name, _now()))
        return {"id": wid, "user_id": user_id, "name": name}

    def list_for_user(self, user_id: str) -> list[dict]:
        lists = self.db.query(
            "SELECT * FROM watchlists WHERE user_id=? ORDER BY created_at", (user_id,))
        for wl in lists:
            wl["items"] = self.db.query(
                "SELECT * FROM watchlist_items WHERE watchlist_id=? ORDER BY created_at",
                (wl["id"],))
        return lists

    def _owns(self, user_id: str, watchlist_id: str) -> bool:
        return bool(self.db.query(
            "SELECT 1 FROM watchlists WHERE id=? AND user_id=?", (watchlist_id, user_id)))

    def add_item(self, user_id: str, watchlist_id: str, item: dict[str, Any]) -> dict | None:
        """Add a prop to a watchlist the user owns. Returns None if they don't own it."""
        if not self._owns(user_id, watchlist_id):
            return None
        iid = _uid()
        self.db.execute(
            "INSERT INTO watchlist_items (id, watchlist_id, line_id, sport, player, "
            "stat_type, note, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (iid, watchlist_id, item.get("line_id"), item.get("sport"),
             item.get("player"), item.get("stat_type"), item.get("note"), _now()))
        return {"id": iid, "watchlist_id": watchlist_id, **item}

    def remove_item(self, user_id: str, item_id: str) -> bool:
        """Delete an item only if it belongs to a watchlist the user owns."""
        owned = self.db.query(
            "SELECT wi.id FROM watchlist_items wi JOIN watchlists w "
            "ON wi.watchlist_id = w.id WHERE wi.id=? AND w.user_id=?", (item_id, user_id))
        if not owned:
            return False
        self.db.execute("DELETE FROM watchlist_items WHERE id=?", (item_id,))
        return True

    def delete(self, user_id: str, watchlist_id: str) -> bool:
        if not self._owns(user_id, watchlist_id):
            return False
        self.db.execute("DELETE FROM watchlist_items WHERE watchlist_id=?", (watchlist_id,))
        self.db.execute("DELETE FROM watchlists WHERE id=?", (watchlist_id,))
        return True


class PortfolioRepository:
    def __init__(self, db: Database):
        self.db = db

    def add(self, user_id: str, entry: dict[str, Any]) -> dict:
        eid = _uid()
        row = {
            "id": eid, "user_id": user_id,
            "line_id": entry.get("line_id"), "sport": entry.get("sport"),
            "player": entry.get("player"), "stat_type": entry.get("stat_type"),
            "side": (entry.get("side") or "over").lower(),
            "line_value": entry.get("line_value"),
            "stake": float(entry.get("stake", 1.0) or 1.0),
            "odds": entry.get("odds"), "status": "open", "result": None,
            "placed_at": _now(), "settled_at": None,
        }
        self.db.execute(
            "INSERT INTO portfolio_entries (id, user_id, line_id, sport, player, stat_type, "
            "side, line_value, stake, odds, status, result, placed_at, settled_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            tuple(row[k] for k in ("id", "user_id", "line_id", "sport", "player",
                                   "stat_type", "side", "line_value", "stake", "odds",
                                   "status", "result", "placed_at", "settled_at")))
        return row

    def list_for_user(self, user_id: str, status: str | None = None) -> list[dict]:
        if status:
            return self.db.query(
                "SELECT * FROM portfolio_entries WHERE user_id=? AND status=? "
                "ORDER BY placed_at DESC", (user_id, status))
        return self.db.query(
            "SELECT * FROM portfolio_entries WHERE user_id=? ORDER BY placed_at DESC",
            (user_id,))

    def settle(self, user_id: str, entry_id: str, status: str,
               result: float | None = None) -> bool:
        """Mark an entry won/lost/void. Scoped to the owner."""
        if status not in ("won", "lost", "void"):
            return False
        owned = self.db.query(
            "SELECT 1 FROM portfolio_entries WHERE id=? AND user_id=?", (entry_id, user_id))
        if not owned:
            return False
        self.db.execute(
            "UPDATE portfolio_entries SET status=?, result=?, settled_at=? "
            "WHERE id=? AND user_id=?", (status, result, _now(), entry_id, user_id))
        return True

    def remove(self, user_id: str, entry_id: str) -> bool:
        owned = self.db.query(
            "SELECT 1 FROM portfolio_entries WHERE id=? AND user_id=?", (entry_id, user_id))
        if not owned:
            return False
        self.db.execute("DELETE FROM portfolio_entries WHERE id=?", (entry_id,))
        return True


class AlertRepository:
    _METRICS = ("edge", "juice", "probability", "line_move")
    _COMPARATORS = ("gte", "lte")

    def __init__(self, db: Database):
        self.db = db

    def create(self, user_id: str, alert: dict[str, Any]) -> Optional[dict]:
        metric = (alert.get("metric") or "").lower()
        comparator = (alert.get("comparator") or "gte").lower()
        if metric not in self._METRICS or comparator not in self._COMPARATORS:
            return None
        try:
            threshold = float(alert.get("threshold"))
        except (TypeError, ValueError):
            return None
        aid = _uid()
        row = {
            "id": aid, "user_id": user_id, "line_id": alert.get("line_id"),
            "player": alert.get("player"), "stat_type": alert.get("stat_type"),
            "metric": metric, "comparator": comparator, "threshold": threshold,
            "active": 1, "created_at": _now(), "triggered_at": None,
        }
        self.db.execute(
            "INSERT INTO alerts (id, user_id, line_id, player, stat_type, metric, "
            "comparator, threshold, active, created_at, triggered_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            tuple(row[k] for k in ("id", "user_id", "line_id", "player", "stat_type",
                                   "metric", "comparator", "threshold", "active",
                                   "created_at", "triggered_at")))
        return row

    def list_for_user(self, user_id: str, active_only: bool = False) -> list[dict]:
        q = "SELECT * FROM alerts WHERE user_id=?"
        if active_only:
            q += " AND active=1"
        return self.db.query(q + " ORDER BY created_at DESC", (user_id,))

    def set_active(self, user_id: str, alert_id: str, active: bool) -> bool:
        owned = self.db.query(
            "SELECT 1 FROM alerts WHERE id=? AND user_id=?", (alert_id, user_id))
        if not owned:
            return False
        self.db.execute("UPDATE alerts SET active=? WHERE id=? AND user_id=?",
                        (1 if active else 0, alert_id, user_id))
        return True

    def remove(self, user_id: str, alert_id: str) -> bool:
        owned = self.db.query(
            "SELECT 1 FROM alerts WHERE id=? AND user_id=?", (alert_id, user_id))
        if not owned:
            return False
        self.db.execute("DELETE FROM alerts WHERE id=?", (alert_id,))
        return True
