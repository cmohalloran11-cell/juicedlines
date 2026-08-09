"""
fantasy_sync_job.py — scheduled entrypoint for .github/workflows/fantasy-sync.yml.

Populates the Supabase Postgres backing the Vercel serverless Fantasy Draft Assistant
(static/api/fantasy/*.js) with the same data the live-server path (routes_fantasy.py) would
otherwise sync lazily on-demand. Everything here reuses the existing, already-tested
fantasy/ package unmodified -- this file is only the entrypoint, run with DATABASE_URL
pointed at that Postgres (a GitHub secret, never a local default).
"""
from __future__ import annotations

import datetime
import logging
import os
import sys

import fantasy
import store
from fantasy import players_sync, projections_sync, sleeper_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fantasy_sync_job")


def main() -> int:
    if not os.getenv("DATABASE_URL", "").startswith(("postgres://", "postgresql://")):
        log.error("DATABASE_URL must point at Postgres for this job (got: %r)", os.getenv("DATABASE_URL"))
        return 1

    db = store.get_database()
    store.init_schema(db)
    fantasy.init_schema(db)

    if players_sync.should_sync(db):
        result = players_sync.sync_players(db)
        log.info("players sync: %s", result)
    else:
        log.info("players sync: skipped (synced recently)")

    state = sleeper_client.get_nfl_state() or {}
    season = int(state.get("season") or datetime.datetime.now(datetime.timezone.utc).year)
    if projections_sync.should_sync_season(db, season):
        result = projections_sync.sync_season_projections(db, season)
        log.info("season projections sync (%s): %s", season, result)
    else:
        log.info("season projections sync: skipped (synced recently)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
