"""
Tests for fantasy.repositories + fantasy.schema — the canonical player table, source-id
mapping layer, unmatched-player review log, league import storage, and projection upsert.
Each test gets a fresh temp SQLite DB (store.database.SQLiteDatabase), matching this repo's
test-isolation convention (see tests/test_dashboard.py's fixture).
"""
from __future__ import annotations

import pytest

import fantasy
from store.database import SQLiteDatabase
from fantasy.repositories import (
    PlayerRepository, PlayerMappingRepository, UnmatchedPlayerRepository,
    SyncLogRepository, LeagueRepository, ProjectionRepository,
)


@pytest.fixture()
def db(tmp_path):
    d = SQLiteDatabase(tmp_path / "fantasy_test.db")
    fantasy.init_schema(d)
    return d


def test_player_create_and_get_roundtrip(db):
    repo = PlayerRepository(db)
    created = repo.create(full_name="Ja'Marr Chase", first_name="Ja'Marr", last_name="Chase",
                          position="WR", team="CIN", status="Active")
    fetched = repo.get(created["id"])
    assert fetched["full_name"] == "Ja'Marr Chase"
    assert fetched["position"] == "WR"


def test_player_update_bumps_updated_at_and_leaves_created_at(db):
    repo = PlayerRepository(db)
    created = repo.create(full_name="Test Player", position="RB")
    repo.update(created["id"], team="KC", status="Active")
    fetched = repo.get(created["id"])
    assert fetched["team"] == "KC"
    assert fetched["created_at"] == created["created_at"]


def test_player_mapping_resolve_missing_returns_none(db):
    mapper = PlayerMappingRepository(db)
    assert mapper.resolve("sleeper", "does-not-exist") is None


def test_player_mapping_map_then_resolve(db):
    players = PlayerRepository(db)
    mapper = PlayerMappingRepository(db)
    player = players.create(full_name="Bijan Robinson", position="RB")
    mapper.map(player["id"], "sleeper", "SLP123")
    assert mapper.resolve("sleeper", "SLP123") == player["id"]


def test_player_mapping_remap_refreshes_confidence(db):
    players = PlayerRepository(db)
    mapper = PlayerMappingRepository(db)
    player = players.create(full_name="Puka Nacua", position="WR")
    mapper.map(player["id"], "nflverse_gsis", "00-7777777", confidence=0.9)
    mapper.map(player["id"], "nflverse_gsis", "00-7777777", confidence=1.0)
    rows = mapper.ids_for_player(player["id"])
    assert len(rows) == 1  # remapping the same source/source_id updates, doesn't duplicate
    assert rows[0]["confidence"] == 1.0


def test_unmatched_player_log_and_resolve(db):
    unmatched = UnmatchedPlayerRepository(db)
    players = PlayerRepository(db)
    row = unmatched.log(source="nflverse_gsis", source_id="00-1234567",
                        raw_name="Some Rookie", raw_team="SEA", raw_position="WR")
    assert row["status"] == "pending"
    assert len(unmatched.list_pending()) == 1

    player = players.create(full_name="Some Rookie", position="WR", team="SEA")
    assert unmatched.resolve(row["id"], player["id"]) is True
    assert unmatched.list_pending() == []
    mapper = PlayerMappingRepository(db)
    assert mapper.resolve("nflverse_gsis", "00-1234567") == player["id"]


def test_unmatched_player_ignore(db):
    unmatched = UnmatchedPlayerRepository(db)
    row = unmatched.log(source="nflverse_gsis", source_id="00-9999999", raw_name="Nobody")
    assert unmatched.ignore(row["id"]) is True
    assert unmatched.list_pending() == []


def test_sync_log_gates_on_missing_and_recorded(db):
    sync = SyncLogRepository(db)
    assert sync.last_synced("sleeper_players_nfl") is None
    sync.record("sleeper_players_nfl", 2500)
    assert sync.last_synced("sleeper_players_nfl") is not None


def test_league_upsert_is_idempotent_and_scoped_to_user(db):
    leagues = LeagueRepository(db)
    scoring_settings = {"rec": 1.0, "pass_td": 4.0}
    roster = ["QB", "RB", "RB", "WR", "WR", "FLEX", "BN"]

    first = leagues.upsert("user-1", "sleeper-league-1", "My League", "2026", 12,
                           scoring_settings, roster)
    again = leagues.upsert("user-1", "sleeper-league-1", "My League (renamed)", "2026", 12,
                           scoring_settings, roster)
    assert first["id"] == again["id"]  # same league re-imported, not duplicated
    assert again["name"] == "My League (renamed)"

    other_user = leagues.upsert("user-2", "sleeper-league-1", "Their View", "2026", 12,
                                scoring_settings, roster)
    assert other_user["id"] != first["id"]  # same Sleeper league, different local rows per user

    assert leagues.get("user-1", first["id"])["scoring_settings"] == scoring_settings
    assert leagues.get("user-2", first["id"]) is None  # cross-user access denied


def test_league_list_for_user(db):
    leagues = LeagueRepository(db)
    leagues.upsert("user-1", "league-a", "A", "2026", 10, {}, [])
    leagues.upsert("user-1", "league-b", "B", "2026", 10, {}, [])
    rows = leagues.list_for_user("user-1")
    assert {r["sleeper_league_id"] for r in rows} == {"league-a", "league-b"}


def test_projection_upsert_updates_in_place_for_season_long(db):
    proj = ProjectionRepository(db)
    first = proj.upsert("player-1", "nflverse", 2026, None, {"rush_yd": 900.0}, "v1")
    second = proj.upsert("player-1", "nflverse", 2026, None, {"rush_yd": 950.0}, "v2")
    assert first["id"] == second["id"]  # re-run of the sync updates the same row, no duplicate

    rows = proj.for_season("nflverse", 2026)
    assert len(rows) == 1
    assert rows[0]["stats"]["rush_yd"] == 950.0


def test_projection_week_null_and_weekly_do_not_collide(db):
    proj = ProjectionRepository(db)
    proj.upsert("player-1", "nflverse", 2026, None, {"rush_yd": 900.0}, "season")
    proj.upsert("player-1", "nflverse", 2026, 3, {"rush_yd": 60.0}, "week3")
    season_rows = proj.for_season("nflverse", 2026, week=None)
    week_rows = proj.for_season("nflverse", 2026, week=3)
    assert len(season_rows) == 1 and season_rows[0]["stats"]["rush_yd"] == 900.0
    assert len(week_rows) == 1 and week_rows[0]["stats"]["rush_yd"] == 60.0
