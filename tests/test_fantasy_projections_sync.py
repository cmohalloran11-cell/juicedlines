"""
Tests for fantasy.projections_sync — populating fantasy_projections from the active
ProjectionsProvider. The provider itself is monkeypatched (a fake object implementing
get_projections) so this never triggers a real nflverse fetch; that's covered separately by
tests/test_fantasy_projections_nflverse.py.
"""
from __future__ import annotations

import pytest

import fantasy
from store.database import SQLiteDatabase
from fantasy import projections_sync
from fantasy.repositories import ProjectionRepository, SyncLogRepository


@pytest.fixture()
def db(tmp_path):
    d = SQLiteDatabase(tmp_path / "fantasy_test.db")
    fantasy.init_schema(d)
    return d


class _FakeProvider:
    name = "fake"

    def __init__(self, rows):
        self._rows = rows

    def get_projections(self, season, week=None):
        return self._rows


def _patch_provider(monkeypatch, rows):
    monkeypatch.setattr(projections_sync, "get_provider", lambda name=None: _FakeProvider(rows))


def test_should_sync_season_true_when_never_synced(db):
    assert projections_sync.should_sync_season(db, 2026) is True


def test_sync_season_projections_upserts_and_records_log(db, monkeypatch):
    rows = [
        {"player_id": "p1", "position": "RB", "provider": "fake",
         "stats": {"rush_yd": 900.0}, "methodology": "m"},
        {"player_id": "p2", "position": "WR", "provider": "fake",
         "stats": {"rec_yd": 700.0}, "methodology": "m"},
    ]
    _patch_provider(monkeypatch, rows)

    result = projections_sync.sync_season_projections(db, 2026)
    assert result == {"players": 2}
    assert projections_sync.should_sync_season(db, 2026) is False

    stored = ProjectionRepository(db).for_season("fake", 2026)
    assert {r["player_id"] for r in stored} == {"p1", "p2"}


def test_sync_season_projections_is_scoped_per_season(db, monkeypatch):
    _patch_provider(monkeypatch, [{"player_id": "p1", "position": "RB", "provider": "fake",
                                   "stats": {}, "methodology": "m"}])
    projections_sync.sync_season_projections(db, 2026)
    # 2027 hasn't been synced yet, even though 2026 just was -- each season is its own gate
    assert projections_sync.should_sync_season(db, 2027) is True


def test_sync_week_projections_stored_separately_from_season(db, monkeypatch):
    _patch_provider(monkeypatch, [{"player_id": "p1", "position": "RB", "provider": "fake",
                                   "stats": {"rush_yd": 60.0}, "methodology": "m"}])
    projections_sync.sync_week_projections(db, 2026, 3)
    assert projections_sync.should_sync_week(db, 2026, 3) is False
    assert projections_sync.should_sync_week(db, 2026, 4) is True   # different week, own gate

    week_rows = ProjectionRepository(db).for_season("fake", 2026, week=3)
    season_rows = ProjectionRepository(db).for_season("fake", 2026, week=None)
    assert len(week_rows) == 1
    assert season_rows == []   # never touched the season-long (week=NULL) bucket


def test_ensure_season_projections_skips_when_fresh(db, monkeypatch):
    calls = {"n": 0}

    def fake_sync(*a, **k):
        calls["n"] += 1

    SyncLogRepository(db).record(projections_sync._season_key(2026), 5)
    monkeypatch.setattr(projections_sync, "sync_season_projections", fake_sync)
    projections_sync.ensure_season_projections(db, 2026)
    assert calls["n"] == 0   # already synced recently -- no redundant fetch


def test_ensure_season_projections_is_best_effort_on_provider_failure(db, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("nflverse unreachable")

    monkeypatch.setattr(projections_sync, "sync_season_projections", boom)
    projections_sync.ensure_season_projections(db, 2026)   # must not raise
