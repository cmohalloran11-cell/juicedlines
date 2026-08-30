"""
cfb.player_status — the manual, admin-editable status override (cfb_player_status) and the
staleness rule board.py's real projection-attach step (once the modeling agent builds it) is
expected to check.

No CFBD/Odds-API injury report is mandated to exist for college football (unlike MLB's
statsapi or NFL's official designations), so this table IS the injury signal: a human sets it
via /api/cfb/player-status (ADMIN role, see routes_cfb.py), and `is_stale` tells a caller
whether that confirmation is old enough, relative to kickoff, that a projection built on it
should be flagged rather than trusted at face value.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from store.database import Database
from .repositories import PlayerStatusRepository

# A status confirmed more than this many hours before kickoff is stale -- close enough to
# game time that a lot can have changed (a Thursday "questionable" tag says little about
# Saturday availability). Deliberately generous (CFB status updates are infrequent compared
# to the NFL's daily practice-report cadence), and easy to tighten once real usage data
# exists -- see cfb/README.md's extension points.
STALE_AFTER_HOURS = 48.0


def get_status(db: Database, player_id: str) -> Optional[dict]:
    return PlayerStatusRepository(db).get(player_id)


def set_status(db: Database, player_id: str, status: str, note: Optional[str] = None,
              set_by: Optional[str] = None) -> dict:
    return PlayerStatusRepository(db).set(player_id, status, note=note, set_by=set_by)


def is_stale(status_row: Optional[dict], kickoff_iso: Optional[str],
            now: Optional[datetime] = None) -> bool:
    """True when a projection near kickoff should be flagged because the status is either
    UNKNOWN (no override ever set) or was last confirmed too long before kickoff to trust.
    `kickoff_iso` absent (schedule not yet matched) -> can't judge proximity, so this only
    flags the "never set" case."""
    if status_row is None:
        return True
    try:
        as_of = datetime.fromisoformat(status_row["as_of"])
    except (KeyError, TypeError, ValueError):
        return True
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    if not kickoff_iso:
        return False
    try:
        kickoff = datetime.fromisoformat(kickoff_iso)
    except (TypeError, ValueError):
        return False
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=timezone.utc)
    ref = now or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    hours_before_kickoff_confirmed = (kickoff - as_of).total_seconds() / 3600.0
    near_kickoff = (kickoff - ref).total_seconds() / 3600.0 <= 24.0
    return near_kickoff and hours_before_kickoff_confirmed > STALE_AFTER_HOURS
