"""
routes_cfb.py — HTTP surface for the CFB data layer (cfb/). Mounted by main.py
(app.include_router(cfb_router)).

Every route here returns OUR OWN canonical rows (cfb_teams/cfb_players/cfb_player_status), a
transform of the underlying CFBD/Odds-API data, never a raw upstream payload -- see
cfb/README.md's License constraints section (no bulk raw CFBD export, ever).

Auth follows CLAUDE.md's rule: viewing public data stays public (teams/players/status are all
already-transparent board context, not compute-triggering); anything that WRITES the manual
status override or resolves a fuzzy-match review row requires ADMIN, same as
/api/admin/calibration in main.py.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

import store
from auth import require_role
from cfb import player_status
from cfb.repositories import PlayerRepository, TeamRepository, UnmatchedPlayerRepository
from cfb.players_sync import SOURCE as _SYNC_SOURCE
from cfb.repositories import SyncLogRepository
from cfb.config import cfbd_api_key, odds_api_key

router = APIRouter(prefix="/api/cfb", tags=["cfb"])


def _db():
    return store.get_database()


@router.get("/status")
def cfb_status():
    """Transparency: which optional CFB data sources are configured, and how much of the
    canonical player table has been synced -- no raw CFBD/Odds-API data, just our own DB
    counts and boolean key presence (never the key values themselves)."""
    db = _db()
    return {
        "cfbd_configured": cfbd_api_key() is not None,
        "odds_api_configured": odds_api_key() is not None,
        "teams_synced": len(TeamRepository(db).all()),
        "players_synced": len(PlayerRepository(db).all()),
        "last_roster_sync": SyncLogRepository(db).last_synced(_SYNC_SOURCE),
    }


@router.get("/teams")
def list_teams(classification: str = Query("fbs", max_length=20)):
    return {"teams": TeamRepository(_db()).all(classification=classification or None)}


@router.get("/players")
def list_players(team: str = Query("", max_length=100),
                 position: str = Query("", max_length=10)):
    db = _db()
    if team:
        players = PlayerRepository(db).find_by_team(team)
    elif position:
        players = PlayerRepository(db).find_by_position(position.upper())
    else:
        players = PlayerRepository(db).all()
    return {"players": players}


@router.get("/player-status/{player_id}")
def get_player_status(player_id: str):
    row = player_status.get_status(_db(), player_id)
    return {"player_id": player_id, "status": row}


class PlayerStatusIn(BaseModel):
    status: str
    note: Optional[str] = None


@router.put("/player-status/{player_id}")
def set_player_status(player_id: str, body: PlayerStatusIn,
                      admin: dict = Depends(require_role("ADMIN", "SUPER_ADMIN"))):
    """Manual override -- no mandated CFB injury report exists (see cfb/player_status.py's
    module docstring), so this IS the availability signal. ADMIN-gated: it's a write that
    downstream projections/display will treat as ground truth."""
    if not body.status.strip():
        raise HTTPException(status_code=422, detail="status is required")
    row = player_status.set_status(_db(), player_id, body.status.strip(), note=body.note,
                                   set_by=admin.get("email") or admin.get("id"))
    return {"player_status": row}


@router.get("/unmatched-players")
def unmatched_players(admin: dict = Depends(require_role("ADMIN", "SUPER_ADMIN"))):
    return {"unmatched": UnmatchedPlayerRepository(_db()).list_pending()}


class ResolveIn(BaseModel):
    player_id: str


@router.post("/unmatched-players/{unmatched_id}/resolve")
def resolve_unmatched(unmatched_id: str, body: ResolveIn,
                      admin: dict = Depends(require_role("ADMIN", "SUPER_ADMIN"))):
    ok = UnmatchedPlayerRepository(_db()).resolve(unmatched_id, body.player_id)
    if not ok:
        raise HTTPException(status_code=404, detail="unmatched row not found")
    return {"resolved": True}


@router.post("/unmatched-players/{unmatched_id}/ignore")
def ignore_unmatched(unmatched_id: str,
                     admin: dict = Depends(require_role("ADMIN", "SUPER_ADMIN"))):
    ok = UnmatchedPlayerRepository(_db()).ignore(unmatched_id)
    if not ok:
        raise HTTPException(status_code=404, detail="unmatched row not found")
    return {"ignored": True}
