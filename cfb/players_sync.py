"""
cfb.players_sync — the scheduled job that pulls CFBD's FBS teams + rosters and upserts them
into cfb_teams / cfb_players + the "cfbd" source mapping. NEVER called from a request path --
a full FBS roster sync is 134 teams' worth of /roster calls, the same "don't do bulk external
fetches on a request path" constraint fantasy.players_sync documents for Sleeper's dump.
Wired into main.py's lifespan as a daily loop, gated by cfb_sync_log so a restart doesn't
re-trigger it early.
"""
from __future__ import annotations

import logging

from store.database import Database
from .data.cfbd_client import CFBDClient
from .repositories import PlayerMappingRepository, PlayerRepository, SyncLogRepository, TeamRepository

log = logging.getLogger(__name__)

SOURCE = "cfbd_fbs_rosters"
_CURRENT_SEASON_FALLBACK = 2026


def sync_teams_and_rosters(db: Database, season: int = _CURRENT_SEASON_FALLBACK) -> dict[str, int]:
    """Fetch every FBS team + roster from CFBD and upsert into cfb_teams/cfb_players, linking
    the "cfbd" athlete-id mapping. No CFBD_API_KEY configured -> CFBDClient.teams() returns
    [], so this is a safe no-op (0/0/0), never a crash. Idempotent -- safe to re-run."""
    client = CFBDClient()
    teams = client.teams(season)
    team_repo, players_repo, mapper = TeamRepository(db), PlayerRepository(db), PlayerMappingRepository(db)

    for t in teams:
        team_repo.upsert(t.school, t.conference, t.classification, t.abbreviation, t.id)

    created = updated = skipped = 0
    for t in teams:
        for p in client.roster(t.school, season):
            if not p.name:
                skipped += 1
                continue
            existing_id = mapper.resolve("cfbd", p.id)
            first, _, last = p.name.partition(" ")
            fields = dict(full_name=p.name, first_name=first or None, last_name=last or None,
                         position=p.position or None, team=t.school, jersey=p.jersey)
            if existing_id:
                players_repo.update(existing_id, **fields)
                updated += 1
            else:
                row = players_repo.create(**fields)
                mapper.map(row["id"], "cfbd", p.id, confidence=1.0)
                created += 1

    SyncLogRepository(db).record(SOURCE, created + updated)
    log.info("cfb players sync: %d teams, %d players created, %d updated, %d skipped",
             len(teams), created, updated, skipped)
    return {"teams": len(teams), "created": created, "updated": updated, "skipped": skipped}


def should_sync(db: Database, min_hours: float = 20.0) -> bool:
    """True if the last successful sync is missing or older than `min_hours` -- survives
    process restarts (checked against cfb_sync_log, not in-memory state)."""
    return SyncLogRepository(db).is_stale(SOURCE, min_hours)
