"""
Tests for fantasy.projections.nflverse_adapter — the nflverse ProjectionsProvider.

aggregate_seasons/project_players are pure functions of already-fetched rows (fetch_rows is
isolated to its own function precisely so these never need real HTTP) -- this file feeds
synthetic rows and checks the shrinkage math exactly, then checks the provider's identity-
resolution behavior (skip-unmapped, never mis-map) against a real temp DB.
"""
from __future__ import annotations

import pytest

import fantasy
import store
from store.database import SQLiteDatabase
from fantasy.repositories import PlayerRepository, PlayerMappingRepository
from fantasy.projections import nflverse_adapter as nfl
from fantasy.projections.nflverse_adapter import NflverseProjectionsProvider


def _row(gsis_id, season, week, position, rushing_yards=0, receiving_yards=0, receptions=0,
        season_type="REG", team="SEA", name="Test Player"):
    return {
        "player_id": gsis_id, "player_display_name": name, "position": position,
        "recent_team": team, "season": str(season), "week": str(week), "season_type": season_type,
        "rushing_yards": str(rushing_yards), "receiving_yards": str(receiving_yards),
        "receptions": str(receptions),
    }


def test_aggregate_seasons_sums_games_and_filters_reg_season_offense_only():
    rows = [
        _row("00-A", 2025, 1, "RB", rushing_yards=100),
        _row("00-A", 2025, 2, "RB", rushing_yards=80),
        _row("00-A", 2025, 3, "RB", rushing_yards=10, season_type="POST"),  # excluded
        _row("00-B", 2025, 1, "K"),   # excluded -- not an offense skill position
        _row("00-A", 2023, 1, "RB", rushing_yards=999),  # outside the requested season set
    ]
    agg = nfl.aggregate_seasons(rows, seasons={2025})
    assert set(agg.keys()) == {("00-A", 2025)}
    a = agg[("00-A", 2025)]
    assert a["games"] == 2
    assert a["totals"]["rush_yd"] == 180.0


def test_project_players_shrinkage_math_matches_hand_calculation():
    per_player_season = {
        ("00-A", 2025): {"gsis_id": "00-A", "season": 2025, "position": "RB",
                         "name": "Player A", "team": "SEA", "games": 2,
                         "totals": {"rush_yd": 180.0}},
        ("00-B", 2025): {"gsis_id": "00-B", "season": 2025, "position": "RB",
                         "name": "Player B", "team": "SEA", "games": 1,
                         "totals": {"rush_yd": 50.0}},
    }
    projected = nfl.project_players(per_player_season, target_season=2026,
                                    projected_games=17, shrinkage_k=6.0)
    by_id = {p["gsis_id"]: p for p in projected}

    # prior = mean per-game rate across both player-seasons = (90 + 50) / 2 = 70
    # A: shrunk = (90*2 + 70*6) / 8 = 75.0  -> projected = 75 * 17 = 1275.0
    assert by_id["00-A"]["stats"]["rush_yd"] == 1275.0
    # B: shrunk = (50*1 + 70*6) / 7 = 67.142857...  -> * 17 = 1141.43 (rounded)
    assert by_id["00-B"]["stats"]["rush_yd"] == 1141.43
    assert "not a predictive model" in by_id["00-A"]["methodology"]


def test_project_players_only_uses_seasons_before_target():
    per_player_season = {
        ("00-A", 2026): {"gsis_id": "00-A", "season": 2026, "position": "RB", "name": "A",
                         "team": "SEA", "games": 5, "totals": {"rush_yd": 500.0}},
    }
    # target_season=2026 itself must never be used as "history" -- it hasn't happened yet.
    projected = nfl.project_players(per_player_season, target_season=2026)
    assert projected == []


def test_project_players_uses_most_recent_available_season():
    per_player_season = {
        ("00-A", 2024): {"gsis_id": "00-A", "season": 2024, "position": "WR", "name": "A",
                         "team": "SEA", "games": 10, "totals": {"rec_yd": 700.0}},
        ("00-A", 2025): {"gsis_id": "00-A", "season": 2025, "position": "WR", "name": "A",
                         "team": "SEA", "games": 10, "totals": {"rec_yd": 900.0}},
    }
    projected = nfl.project_players(per_player_season, target_season=2026)
    assert "2025 season" in projected[0]["methodology"]


@pytest.fixture()
def db(tmp_path, monkeypatch):
    d = SQLiteDatabase(tmp_path / "fantasy_test.db")
    fantasy.init_schema(d)
    store.reset_singleton()
    monkeypatch.setattr(store.database, "_singleton", d, raising=False)
    return d


def test_provider_skips_players_without_a_canonical_mapping(db, monkeypatch):
    players = PlayerRepository(db)
    mapper = PlayerMappingRepository(db)
    mapped_player = players.create(full_name="Mapped Guy", position="RB", team="SEA")
    mapper.map(mapped_player["id"], "nflverse_gsis", "00-MAPPED")

    rows = [
        _row("00-MAPPED", 2025, 1, "RB", rushing_yards=100, name="Mapped Guy"),
        _row("00-MAPPED", 2025, 2, "RB", rushing_yards=100, name="Mapped Guy"),
        _row("00-UNMAPPED", 2025, 1, "RB", rushing_yards=50, name="Unmapped Guy"),
    ]
    monkeypatch.setattr(nfl, "fetch_rows", lambda *a, **k: iter(rows))

    provider = NflverseProjectionsProvider()
    out = provider.get_projections(season=2026)

    ids = {p["player_id"] for p in out}
    assert ids == {mapped_player["id"]}  # the unmapped GSIS id is silently omitted, not guessed
    row = out[0]
    assert row["provider"] == "nflverse"
    assert row["position"] == "RB"
    assert "stats" in row and "methodology" in row


def test_provider_weekly_projection_is_unmultiplied_per_game_rate(db, monkeypatch):
    players = PlayerRepository(db)
    mapper = PlayerMappingRepository(db)
    p = players.create(full_name="Weekly Guy", position="RB", team="SEA")
    mapper.map(p["id"], "nflverse_gsis", "00-WEEKLY")

    rows = [
        _row("00-WEEKLY", 2025, 1, "RB", rushing_yards=100, name="Weekly Guy"),
        _row("00-WEEKLY", 2025, 2, "RB", rushing_yards=100, name="Weekly Guy"),
    ]
    monkeypatch.setattr(nfl, "fetch_rows", lambda *a, **k: iter(rows))

    provider = NflverseProjectionsProvider()
    season_out = provider.get_projections(season=2026)[0]
    week_out = provider.get_projections(season=2026, week=3)[0]

    # week value = season value / PROJECTED_GAMES (17) since it's the same shrunk per-game
    # rate, just not multiplied across a season -- not a different, week-specific model.
    assert week_out["stats"]["rush_yd"] == round(season_out["stats"]["rush_yd"] / 17, 2)
    assert "flat to every week" in week_out["methodology"]
