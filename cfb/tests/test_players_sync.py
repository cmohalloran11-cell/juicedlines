"""
Tests for cfb.players_sync — the daily CFBD FBS teams+rosters sync job. Never calls CFBD for
real (CFBDClient.teams/roster are monkeypatched); focuses on the once-a-day gating logic and
idempotent upsert behavior, same shape as tests/test_fantasy_players_sync.py, since a bug here
either burns CFBD's rate budget (134 teams' worth of roster calls) or silently stops updating
rosters.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from cfb import init_schema
from cfb import players_sync
from cfb.data.base import PlayerRef, TeamRef
from cfb.data.cfbd_client import CFBDClient
from cfb.repositories import PlayerMappingRepository, PlayerRepository, SyncLogRepository, TeamRepository
from store.database import SQLiteDatabase


@pytest.fixture()
def db(tmp_path):
    d = SQLiteDatabase(tmp_path / "cfb_sync_test.db")
    init_schema(d)
    return d


_TEAMS = [
    TeamRef(id="1", school="Ohio State", conference="Big Ten", classification="fbs", abbreviation="OSU"),
    TeamRef(id="2", school="Georgia", conference="SEC", classification="fbs", abbreviation="UGA"),
]

_ROSTERS = {
    "Ohio State": [
        PlayerRef(id="cfbd-100", name="Will Howard", team="Ohio State", position="QB", jersey=18),
        PlayerRef(id="cfbd-101", name="", team="Ohio State", position="WR"),  # no name -> skipped
    ],
    "Georgia": [
        PlayerRef(id="cfbd-200", name="Carson Beck", team="Georgia", position="QB", jersey=15),
    ],
}


def _patch_client(monkeypatch):
    monkeypatch.setattr(CFBDClient, "teams", lambda self, season, classification="fbs": _TEAMS)
    monkeypatch.setattr(CFBDClient, "roster", lambda self, team, season: _ROSTERS[team])


def test_sync_creates_teams_and_named_players_only(db, monkeypatch):
    _patch_client(monkeypatch)
    result = players_sync.sync_teams_and_rosters(db, season=2026)
    assert result == {"teams": 2, "created": 2, "updated": 0, "skipped": 1}

    assert {t["school"] for t in TeamRepository(db).all()} == {"Ohio State", "Georgia"}
    players = PlayerRepository(db).all()
    assert {p["full_name"] for p in players} == {"Will Howard", "Carson Beck"}


def test_sync_maps_cfbd_ids_for_resolution(db, monkeypatch):
    _patch_client(monkeypatch)
    players_sync.sync_teams_and_rosters(db, season=2026)

    mapper = PlayerMappingRepository(db)
    howard_id = mapper.resolve("cfbd", "cfbd-100")
    beck_id = mapper.resolve("cfbd", "cfbd-200")
    assert howard_id is not None and beck_id is not None
    assert howard_id != beck_id


def test_sync_is_idempotent_on_rerun(db, monkeypatch):
    _patch_client(monkeypatch)
    players_sync.sync_teams_and_rosters(db, season=2026)
    result = players_sync.sync_teams_and_rosters(db, season=2026)
    assert result == {"teams": 2, "created": 0, "updated": 2, "skipped": 1}
    assert len(PlayerRepository(db).all()) == 2  # no duplicates on re-sync


def test_sync_records_sync_log(db, monkeypatch):
    _patch_client(monkeypatch)
    players_sync.sync_teams_and_rosters(db, season=2026)
    assert SyncLogRepository(db).is_stale(players_sync.SOURCE) is False


def test_sync_is_a_safe_no_op_without_an_api_key(db, monkeypatch):
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    result = players_sync.sync_teams_and_rosters(db, season=2026)
    assert result == {"teams": 0, "created": 0, "updated": 0, "skipped": 0}
    assert TeamRepository(db).all() == []


def test_should_sync_true_when_never_synced(db):
    assert players_sync.should_sync(db) is True


def test_should_sync_false_within_the_last_20_hours(db, monkeypatch):
    _patch_client(monkeypatch)
    players_sync.sync_teams_and_rosters(db, season=2026)
    assert players_sync.should_sync(db) is False


def test_should_sync_true_once_stale(db, monkeypatch):
    old = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat(timespec="seconds")
    monkeypatch.setattr(SyncLogRepository, "last_synced", lambda self, source: old)
    assert players_sync.should_sync(db) is True
