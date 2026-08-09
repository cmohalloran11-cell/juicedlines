"""
Tests for fantasy_sync_job.py — the entrypoint .github/workflows/fantasy-sync.yml runs.
Everything it calls (players_sync, projections_sync, fantasy.init_schema) is already tested
elsewhere; this file only verifies the entrypoint's own control flow: the DATABASE_URL guard,
and that it calls the sync functions in the expected order with the expected gating.
"""
from __future__ import annotations

import pytest

import fantasy_sync_job
import store
from store.database import SQLiteDatabase


@pytest.fixture()
def db(tmp_path, monkeypatch):
    d = SQLiteDatabase(tmp_path / "fantasy_sync_job_test.db")
    store.reset_singleton()
    monkeypatch.setattr(store.database, "_singleton", d, raising=False)
    return d


def test_main_refuses_non_postgres_database_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "")
    assert fantasy_sync_job.main() == 1


def test_main_refuses_sqlite_database_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "/some/local/path.db")
    assert fantasy_sync_job.main() == 1


def test_main_runs_both_syncs_when_stale(monkeypatch, db):
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake/for-guard-only")
    calls = []
    monkeypatch.setattr(fantasy_sync_job.players_sync, "should_sync", lambda db: True)
    monkeypatch.setattr(fantasy_sync_job.players_sync, "sync_players",
                        lambda db: calls.append("players_sync") or {"created": 1})
    monkeypatch.setattr(fantasy_sync_job.projections_sync, "should_sync_season", lambda db, season: True)
    monkeypatch.setattr(fantasy_sync_job.projections_sync, "sync_season_projections",
                        lambda db, season: calls.append(("projections_sync", season)) or {"players": 1})
    monkeypatch.setattr(fantasy_sync_job.sleeper_client, "get_nfl_state", lambda: {"season": "2026", "week": 1})

    assert fantasy_sync_job.main() == 0
    assert calls == ["players_sync", ("projections_sync", 2026)]


def test_main_skips_both_syncs_when_fresh(monkeypatch, db):
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake/for-guard-only")
    calls = []
    monkeypatch.setattr(fantasy_sync_job.players_sync, "should_sync", lambda db: False)
    monkeypatch.setattr(fantasy_sync_job.players_sync, "sync_players",
                        lambda db: calls.append("players_sync"))
    monkeypatch.setattr(fantasy_sync_job.projections_sync, "should_sync_season", lambda db, season: False)
    monkeypatch.setattr(fantasy_sync_job.projections_sync, "sync_season_projections",
                        lambda db, season: calls.append("projections_sync"))
    monkeypatch.setattr(fantasy_sync_job.sleeper_client, "get_nfl_state", lambda: {"season": "2026", "week": 1})

    assert fantasy_sync_job.main() == 0
    assert calls == []


def test_main_falls_back_to_current_year_when_sleeper_state_unavailable(monkeypatch, db):
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake/for-guard-only")
    monkeypatch.setattr(fantasy_sync_job.players_sync, "should_sync", lambda db: False)
    seasons_checked = []
    monkeypatch.setattr(fantasy_sync_job.projections_sync, "should_sync_season",
                        lambda db, season: seasons_checked.append(season) or False)
    monkeypatch.setattr(fantasy_sync_job.sleeper_client, "get_nfl_state", lambda: None)

    assert fantasy_sync_job.main() == 0
    assert len(seasons_checked) == 1 and seasons_checked[0] >= 2026
