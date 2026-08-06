"""
routes_fantasy.py — HTTP surface for the fantasy football draft assistant (fantasy/). Mounted
by main.py (app.include_router(fantasy_router)).

Every Sleeper call happens here, server-side -- the browser never talks to api.sleeper.app
directly (task constraint). Caching: league reads use a 5-minute in-process TTL cache; draft-
picks polls use a 10-second TTL cache, both keyed by request args -- same _cache-dict pattern
as optimizer.py's _CACHE (module-level, no DB persistence, doesn't need to survive a restart).
"""
from __future__ import annotations

import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

import store
from auth import get_current_user
from fantasy import draft_state, scoring, vor
from fantasy import sleeper_client
from fantasy.repositories import LeagueRepository, PlayerRepository, ProjectionRepository, \
    UnmatchedPlayerRepository

router = APIRouter(prefix="/api/fantasy", tags=["fantasy"])

_LEAGUE_CACHE_TTL = 300.0   # 5 min
_DRAFT_CACHE_TTL = 10.0     # 10s, live-draft only
_cache: dict[str, tuple[float, Any]] = {}


def _cached(key: str, ttl: float, fetch):
    now = time.monotonic()
    hit = _cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    value = fetch()
    _cache[key] = (now, value)
    return value


def _db():
    return store.get_database()


# ── Sleeper passthrough (server-side only -- browser never calls api.sleeper.app) ──────────

@router.get("/sleeper/user/{username}")
def sleeper_user(username: str):
    user = sleeper_client.get_user(username)
    if not user:
        raise HTTPException(status_code=404, detail="Sleeper username not found")
    return user


@router.get("/sleeper/leagues/{sleeper_user_id}")
def sleeper_leagues(sleeper_user_id: str, season: str = Query(...)):
    key = f"leagues:{sleeper_user_id}:{season}"
    leagues = _cached(key, _LEAGUE_CACHE_TTL,
                      lambda: sleeper_client.get_user_leagues(sleeper_user_id, season))
    return {"leagues": leagues}


@router.get("/leagues/{sleeper_league_id}")
def league_detail(sleeper_league_id: str):
    key = f"league:{sleeper_league_id}"
    league = _cached(key, _LEAGUE_CACHE_TTL, lambda: sleeper_client.get_league(sleeper_league_id))
    if not league:
        raise HTTPException(status_code=404, detail="Sleeper league not found")
    return league


@router.get("/leagues/{sleeper_league_id}/drafts")
def league_drafts(sleeper_league_id: str):
    """The league's draft(s) -- lets the UI find a live draft_id without the user hunting it
    down manually. Most recent draft first (Sleeper's own ordering)."""
    key = f"drafts:{sleeper_league_id}"
    drafts = _cached(key, _LEAGUE_CACHE_TTL,
                     lambda: sleeper_client.get_league_drafts(sleeper_league_id))
    return {"drafts": drafts}


# ── league import ────────────────────────────────────────────────────────────────────────

class LeagueImportIn(BaseModel):
    sleeper_league_id: str


@router.post("/leagues/import")
def import_league(body: LeagueImportIn, user: dict = Depends(get_current_user)):
    """Reads the league's real scoring settings + roster shape from Sleeper and persists it
    for this user -- every downstream computation (scoring, VOR, draft board) reads from this
    saved config, never a hardcoded PPR/12-team assumption."""
    league = sleeper_client.get_league(body.sleeper_league_id)
    if not league:
        raise HTTPException(status_code=404, detail="Sleeper league not found")
    saved = LeagueRepository(_db()).upsert(
        user_id=user["id"], sleeper_league_id=body.sleeper_league_id,
        name=league.get("name"), season=league.get("season"),
        league_size=league.get("total_rosters"),
        scoring_settings=league.get("scoring_settings") or {},
        roster_positions=league.get("roster_positions") or [])
    return {"league": saved}


@router.get("/leagues")
def my_leagues(user: dict = Depends(get_current_user)):
    return {"leagues": LeagueRepository(_db()).list_for_user(user["id"])}


# ── draft board (VOR + tiers) ────────────────────────────────────────────────────────────

def _league_or_404(user_id: str, league_id: str) -> dict:
    league = LeagueRepository(_db()).get(user_id, league_id)
    if not league:
        raise HTTPException(status_code=404, detail="league not found or not imported")
    return league


def _scored_board(league: dict) -> list[dict]:
    """VOR-ranked, tiered board for one league's actual scoring rules + roster shape.
    fantasy_projections rows only carry player_id/stats (provider-agnostic, league-agnostic);
    position/name come from the canonical player table, joined here rather than duplicated
    onto every projection row."""
    season = int(league["season"]) if league.get("season") else time.gmtime().tm_year
    projections = ProjectionRepository(_db()).for_season(provider="nflverse", season=season)

    players_repo = PlayerRepository(_db())
    enriched = []
    for proj in projections:
        player = players_repo.get(proj["player_id"])
        if not player:
            continue
        enriched.append({
            "player_id": proj["player_id"],
            "name": player["full_name"],
            "position": player["position"],
            "team": player["team"],
            "stats": proj["stats"],
            "methodology": proj.get("methodology"),
        })

    scored = scoring.score_players(enriched, league["scoring_settings"])
    ranked = vor.compute_vor(scored, league["roster_positions"], league["league_size"] or 12)
    return vor.assign_tiers(ranked)


@router.get("/draft-board")
def draft_board(league_id: str = Query(...), user: dict = Depends(get_current_user)):
    league = _league_or_404(user["id"], league_id)
    return {"league_id": league_id, "board": _scored_board(league)}


# ── live draft mode ──────────────────────────────────────────────────────────────────────

@router.get("/draft/{draft_id}/live")
def live_draft(draft_id: str, league_id: str = Query(...),
               roster_id: Optional[str] = Query(
                   None, description="which team's needs to weight recommendations by"),
               user: dict = Depends(get_current_user)):
    league = _league_or_404(user["id"], league_id)
    picks = _cached(f"picks:{draft_id}", _DRAFT_CACHE_TTL,
                    lambda: sleeper_client.get_draft_picks(draft_id))

    board = _scored_board(league)
    board_by_player = {p["player_id"]: p for p in board}
    # annotate picks with position (from our board) for run-detection -- Sleeper's pick rows
    # carry it under metadata.position, but not every league's picks resolve to a player we
    # have a projection for, so fall back to the board lookup first.
    for pick in picks:
        if not pick.get("position"):
            meta = pick.get("metadata") or {}
            board_hit = board_by_player.get(pick.get("player_id"))
            pick["position"] = (board_hit or {}).get("position") or meta.get("position")

    available = draft_state.remove_drafted(board, picks)
    runs = draft_state.positional_runs(picks)

    drafted_by_user: list[dict] = []
    if roster_id is not None:
        drafted_by_user = [p for p in picks if str(p.get("roster_id")) == str(roster_id)]
    recs = draft_state.recommend(available, league["roster_positions"], drafted_by_user)

    return {
        "draft_id": draft_id, "picks_made": len(picks), "available": available,
        "positional_runs": runs, "recommendations": recs,
    }


# ── unmatched player review (admin) ──────────────────────────────────────────────────────

@router.get("/unmatched-players")
def unmatched_players(user: dict = Depends(get_current_user)):
    if user.get("role") not in ("ADMIN", "SUPER_ADMIN"):
        raise HTTPException(status_code=403, detail="admin only")
    return {"unmatched": UnmatchedPlayerRepository(_db()).list_pending()}
