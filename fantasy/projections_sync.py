"""
fantasy.projections_sync — populates fantasy_projections from the active ProjectionsProvider
(fantasy.projections.get_provider()). Nothing computes VOR/scoring against a provider directly
-- everything reads the persisted rows via ProjectionRepository, same boundary as the players
sync (fantasy.players_sync) keeps for Sleeper's player dump.

Season-long projections are synced lazily, on-demand, gated by the same once-a-day staleness
check as the players dump (SyncLogRepository.is_stale) -- a league's `season` comes from
whatever the user imported, so there's no single "current season" to pre-warm in a background
loop the way players_sync does. Weekly projections are the same: synced lazily per
(season, week) the first time a lineup is requested for that week, then cached.
"""
from __future__ import annotations

import logging
from typing import Optional

from store.database import Database
from .projections import get_provider
from .repositories import ProjectionRepository, SyncLogRepository

log = logging.getLogger(__name__)

SEASON_SOURCE = "fantasy_projections_season"
WEEK_SOURCE = "fantasy_projections_week"


def _season_key(season: int) -> str:
    return f"{SEASON_SOURCE}:{season}"


def _week_key(season: int, week: int) -> str:
    return f"{WEEK_SOURCE}:{season}:{week}"


def should_sync_season(db: Database, season: int, min_hours: float = 20.0) -> bool:
    return SyncLogRepository(db).is_stale(_season_key(season), min_hours)


def should_sync_week(db: Database, season: int, week: int, min_hours: float = 20.0) -> bool:
    return SyncLogRepository(db).is_stale(_week_key(season, week), min_hours)


def _upsert(db: Database, projections: list[dict], season: int, week: Optional[int]) -> int:
    repo = ProjectionRepository(db)
    for p in projections:
        repo.upsert(p["player_id"], p["provider"], season, week, p["stats"], p.get("methodology"))
    return len(projections)


def sync_season_projections(db: Database, season: int, provider_name: str | None = None) -> dict[str, int]:
    """Fetch season-long (redraft) projections from the active provider and upsert them.
    Player identity is already resolved inside the provider (ProjectionsProvider's contract)
    -- unmapped source rows never reach here, they're in fantasy_unmatched_players instead."""
    provider = get_provider(provider_name)
    projections = provider.get_projections(season=season)
    n = _upsert(db, projections, season, None)
    SyncLogRepository(db).record(_season_key(season), n)
    log.info("fantasy season projections sync: %d players (%s, %s)", n, provider.name, season)
    return {"players": n}


def sync_week_projections(db: Database, season: int, week: int,
                          provider_name: str | None = None) -> dict[str, int]:
    provider = get_provider(provider_name)
    projections = provider.get_projections(season=season, week=week)
    n = _upsert(db, projections, season, week)
    SyncLogRepository(db).record(_week_key(season, week), n)
    log.info("fantasy week projections sync: %d players (%s, %s wk%d)", n, provider.name, season, week)
    return {"players": n}


def ensure_season_projections(db: Database, season: int, provider_name: str | None = None) -> None:
    """Sync season projections if stale, best-effort -- a provider fetch failure must not take
    down the draft board; it just serves whatever's already persisted (possibly empty, which
    renders as an honest "no projections yet" state, never a fabricated fallback)."""
    if should_sync_season(db, season):
        try:
            sync_season_projections(db, season, provider_name)
        except Exception as exc:
            log.warning("season projections sync failed (%s): %s", season, exc)


def ensure_week_projections(db: Database, season: int, week: int,
                            provider_name: str | None = None) -> None:
    if should_sync_week(db, season, week):
        try:
            sync_week_projections(db, season, week, provider_name)
        except Exception as exc:
            log.warning("week projections sync failed (%s wk%d): %s", season, week, exc)
