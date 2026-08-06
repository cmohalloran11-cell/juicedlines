"""
fantasy.players_sync — the scheduled job that pulls Sleeper's /v1/players/nfl dump (~5MB) and
upserts it into the canonical player table + sleeper mapping. NEVER called from a request path
-- see fantasy.sleeper_client's module docstring for why. Wired into main.py's lifespan as a
daily loop, gated by fantasy_players_sync_log so a restart doesn't re-trigger it early.
"""
from __future__ import annotations

import logging

from store.database import Database
from . import sleeper_client
from .repositories import PlayerRepository, PlayerMappingRepository, SyncLogRepository

log = logging.getLogger(__name__)

SOURCE = "sleeper_players_nfl"
_RELEVANT_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DEF"}


def sync_players(db: Database) -> dict[str, int]:
    """Fetch Sleeper's player dump and upsert every fantasy-relevant (QB/RB/WR/TE/K/DEF)
    player into fantasy_players, linking/creating the sleeper mapping row. Returns counts for
    logging. Idempotent -- safe to re-run."""
    raw = sleeper_client.fetch_all_players()
    players_repo = PlayerRepository(db)
    mapper = PlayerMappingRepository(db)

    created = updated = skipped = 0
    for sleeper_id, p in raw.items():
        position = p.get("position")
        if position not in _RELEVANT_POSITIONS:
            skipped += 1
            continue
        full_name = p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
        if not full_name:
            skipped += 1
            continue

        existing_id = mapper.resolve("sleeper", sleeper_id)
        fields = dict(
            full_name=full_name, first_name=p.get("first_name"), last_name=p.get("last_name"),
            position=position, team=p.get("team"), status=p.get("status"),
            birth_date=p.get("birth_date"),
        )
        if existing_id:
            players_repo.update(existing_id, **fields)
            updated += 1
        else:
            row = players_repo.create(**fields)
            mapper.map(row["id"], "sleeper", sleeper_id, confidence=1.0)
            created += 1

    SyncLogRepository(db).record(SOURCE, created + updated)
    log.info("fantasy players sync: %d created, %d updated, %d skipped", created, updated, skipped)
    return {"created": created, "updated": updated, "skipped": skipped}


def should_sync(db: Database, min_hours: float = 20.0) -> bool:
    """True if the last successful sync is missing or older than `min_hours` -- the once-a-
    day guard, survives process restarts (checked against fantasy_players_sync_log, not
    in-memory state)."""
    return SyncLogRepository(db).is_stale(SOURCE, min_hours)
