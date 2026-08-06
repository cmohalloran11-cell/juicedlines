"""
Tests for fantasy.players_sync — the daily Sleeper /v1/players/nfl sync job. Never calls
Sleeper for real (fetch_all_players is monkeypatched); focuses on the once-a-day gating logic
and idempotent upsert behavior, since a bug here either burns Sleeper's rate ask or silently
stops updating player data.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import fantasy
from store.database import SQLiteDatabase
from fantasy import players_sync, sleeper_client
from fantasy.repositories import PlayerRepository, PlayerMappingRepository, SyncLogRepository


@pytest.fixture()
def db(tmp_path):
    d = SQLiteDatabase(tmp_path / "fantasy_test.db")
    fantasy.init_schema(d)
    return d


_RAW_PLAYERS = {
    "SLP1": {"full_name": "Christian McCaffrey", "first_name": "Christian",
            "last_name": "McCaffrey", "position": "RB", "team": "SF", "status": "Active",
            "birth_date": "1996-06-07"},
    "SLP2": {"full_name": "Sam Howell", "first_name": "Sam", "last_name": "Howell",
            "position": "QB", "team": "SEA", "status": "Active", "birth_date": None},
    "SLP3": {"full_name": "Some Kicker", "position": "K", "team": "NYJ", "status": "Active"},
    "SLP4": {"position": "LS", "team": "DAL", "status": "Active"},   # irrelevant position
}


def test_sync_players_creates_relevant_positions_only(db, monkeypatch):
    monkeypatch.setattr(sleeper_client, "fetch_all_players", lambda: _RAW_PLAYERS)
    result = players_sync.sync_players(db)
    assert result == {"created": 3, "updated": 0, "skipped": 1}

    players = PlayerRepository(db).all()
    assert len(players) == 3
    assert {p["full_name"] for p in players} == {
        "Christian McCaffrey", "Sam Howell", "Some Kicker"}


def test_sync_players_is_idempotent_on_rerun(db, monkeypatch):
    monkeypatch.setattr(sleeper_client, "fetch_all_players", lambda: _RAW_PLAYERS)
    players_sync.sync_players(db)
    result = players_sync.sync_players(db)
    assert result == {"created": 0, "updated": 3, "skipped": 1}
    assert len(PlayerRepository(db).all()) == 3  # no duplicates on re-sync


def test_sync_players_records_sync_log(db, monkeypatch):
    monkeypatch.setattr(sleeper_client, "fetch_all_players", lambda: _RAW_PLAYERS)
    players_sync.sync_players(db)
    assert SyncLogRepository(db).last_synced(players_sync.SOURCE) is not None


def test_should_sync_true_when_never_synced(db):
    assert players_sync.should_sync(db) is True


def test_should_sync_false_within_the_last_20_hours(db):
    SyncLogRepository(db).record(players_sync.SOURCE, 100)
    assert players_sync.should_sync(db) is False


def test_should_sync_true_once_stale(db, monkeypatch):
    old = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat(timespec="seconds")
    monkeypatch.setattr(SyncLogRepository, "last_synced", lambda self, source: old)
    assert players_sync.should_sync(db) is True
