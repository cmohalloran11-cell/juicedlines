"""
routes_user.py — authenticated per-user endpoints (watchlists + portfolio).

Every route is gated by auth.get_current_user, and every repository call is scoped to the
caller's id, so a user can only ever read or mutate their own data (spec Ch. 10 — User Data
Security). Mounted under /api by main.py.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import store
from auth import get_current_user

router = APIRouter(prefix="/api", tags=["user"])


def _repos():
    db = store.get_database()
    return (store.WatchlistRepository(db), store.PortfolioRepository(db))


def _alert_repo():
    return store.AlertRepository(store.get_database())


# ── identity ──────────────────────────────────────────────────────────────────

@router.get("/me")
def me(user: dict = Depends(get_current_user)):
    """The authenticated user's profile (id, email, role, tier)."""
    return {"user": user}


# ── watchlists ────────────────────────────────────────────────────────────────

class WatchlistCreate(BaseModel):
    name: str


class WatchlistItemIn(BaseModel):
    line_id: str
    sport: Optional[str] = None
    player: Optional[str] = None
    stat_type: Optional[str] = None
    note: Optional[str] = None


@router.get("/watchlists")
def list_watchlists(user: dict = Depends(get_current_user)):
    wl, _ = _repos()
    return {"watchlists": wl.list_for_user(user["id"])}


@router.post("/watchlists")
def create_watchlist(body: WatchlistCreate, user: dict = Depends(get_current_user)):
    wl, _ = _repos()
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="name is required")
    return {"watchlist": wl.create(user["id"], name)}


@router.delete("/watchlists/{watchlist_id}")
def delete_watchlist(watchlist_id: str, user: dict = Depends(get_current_user)):
    wl, _ = _repos()
    if not wl.delete(user["id"], watchlist_id):
        raise HTTPException(status_code=404, detail="watchlist not found")
    return {"deleted": watchlist_id}


@router.post("/watchlists/{watchlist_id}/items")
def add_watchlist_item(watchlist_id: str, item: WatchlistItemIn,
                       user: dict = Depends(get_current_user)):
    wl, _ = _repos()
    added = wl.add_item(user["id"], watchlist_id, item.model_dump())
    if added is None:
        raise HTTPException(status_code=404, detail="watchlist not found")
    return {"item": added}


@router.delete("/watchlists/items/{item_id}")
def remove_watchlist_item(item_id: str, user: dict = Depends(get_current_user)):
    wl, _ = _repos()
    if not wl.remove_item(user["id"], item_id):
        raise HTTPException(status_code=404, detail="item not found")
    return {"deleted": item_id}


# ── portfolio ─────────────────────────────────────────────────────────────────

class PortfolioIn(BaseModel):
    line_id: str
    side: str = "over"
    sport: Optional[str] = None
    player: Optional[str] = None
    stat_type: Optional[str] = None
    line_value: Optional[float] = None
    stake: float = 1.0
    odds: Optional[float] = None


class SettleIn(BaseModel):
    status: str            # won | lost | void
    result: Optional[float] = None


@router.get("/portfolio")
def list_portfolio(status: Optional[str] = None, user: dict = Depends(get_current_user)):
    _, pf = _repos()
    return {"entries": pf.list_for_user(user["id"], status)}


@router.post("/portfolio")
def add_portfolio(entry: PortfolioIn, user: dict = Depends(get_current_user)):
    _, pf = _repos()
    if entry.side.lower() not in ("over", "under"):
        raise HTTPException(status_code=422, detail="side must be over|under")
    return {"entry": pf.add(user["id"], entry.model_dump())}


@router.post("/portfolio/{entry_id}/settle")
def settle_portfolio(entry_id: str, body: SettleIn, user: dict = Depends(get_current_user)):
    _, pf = _repos()
    if not pf.settle(user["id"], entry_id, body.status, body.result):
        raise HTTPException(status_code=400,
                            detail="entry not found or invalid status (won|lost|void)")
    return {"settled": entry_id, "status": body.status}


@router.delete("/portfolio/{entry_id}")
def remove_portfolio(entry_id: str, user: dict = Depends(get_current_user)):
    _, pf = _repos()
    if not pf.remove(user["id"], entry_id):
        raise HTTPException(status_code=404, detail="entry not found")
    return {"deleted": entry_id}


# ── alerts ────────────────────────────────────────────────────────────────────

class AlertIn(BaseModel):
    metric: str                      # edge | juice | probability | line_move
    threshold: float
    comparator: str = "gte"          # gte | lte
    line_id: Optional[str] = None
    player: Optional[str] = None
    stat_type: Optional[str] = None


@router.get("/alerts")
def list_alerts(active: bool = False, user: dict = Depends(get_current_user)):
    return {"alerts": _alert_repo().list_for_user(user["id"], active_only=active)}


@router.post("/alerts")
def create_alert(body: AlertIn, user: dict = Depends(get_current_user)):
    created = _alert_repo().create(user["id"], body.model_dump())
    if created is None:
        raise HTTPException(
            status_code=422,
            detail="metric must be edge|juice|probability|line_move, comparator gte|lte")
    return {"alert": created}


@router.post("/alerts/{alert_id}/toggle")
def toggle_alert(alert_id: str, active: bool = True,
                 user: dict = Depends(get_current_user)):
    if not _alert_repo().set_active(user["id"], alert_id, active):
        raise HTTPException(status_code=404, detail="alert not found")
    return {"alert_id": alert_id, "active": active}


@router.delete("/alerts/{alert_id}")
def remove_alert(alert_id: str, user: dict = Depends(get_current_user)):
    if not _alert_repo().remove(user["id"], alert_id):
        raise HTTPException(status_code=404, detail="alert not found")
    return {"deleted": alert_id}
