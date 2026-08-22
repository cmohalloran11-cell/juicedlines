"""
Offline parsing tests for the balldontlie.io adapter — WNBA's third data-source attempt
(see basketball/data/balldontlie.py's module docstring for why the first two, ESPN and
stats.wnba.com, were replaced). All deterministic — `_get` is monkeypatched, no network,
mirroring test_espn_gamelog.py / test_wnba_stats.py's established pattern for this package.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from basketball.data import balldontlie as B


@pytest.fixture(autouse=True)
def _clear_memo_cache(monkeypatch):
    monkeypatch.setenv("BALLDONTLIE_API_KEY", "test-key")
    B._memory.clear()
    yield
    B._memory.clear()


_TEAMS_PAGE = {"data": [
    {"id": 1, "full_name": "Las Vegas Aces", "abbreviation": "LV", "city": "Las Vegas"},
    {"id": 2, "full_name": "New York Liberty", "abbreviation": "NY", "city": "New York"},
], "meta": {"next_cursor": None}}

_PLAYERS_PAGE = {"data": [
    {"id": 101, "first_name": "A'ja", "last_name": "Wilson", "position": "F",
     "team": {"id": 1, "full_name": "Las Vegas Aces"}},
    {"id": 102, "first_name": "Chelsea", "last_name": "Gray", "position": "G",
     "team": {"id": 1, "full_name": "Las Vegas Aces"}},
], "meta": {"next_cursor": None}}

_STATS_PAGE = {"data": [
    {"min": "32:15", "pts": 24, "reb": 9, "oreb": 2, "dreb": 7, "ast": 3, "stl": 1, "blk": 2,
     "turnover": 2, "fgm": 9, "fga": 17, "fg3m": 1, "fg3a": 3, "ftm": 5, "fta": 6, "pf": 2,
     "team": {"id": 1}, "game": {"id": 5001, "date": "2026-07-20"}},
    {"min": "0", "pts": 0, "reb": 0, "oreb": 0, "dreb": 0, "ast": 0, "stl": 0, "blk": 0,
     "turnover": 0, "fgm": 0, "fga": 0, "fg3m": 0, "fg3a": 0, "ftm": 0, "fta": 0, "pf": 0,
     "team": {"id": 1}, "game": {"id": 5002, "date": "2026-07-15"}},   # DNP — must be dropped
], "meta": {"next_cursor": None}}

_TEAM_STATS_PAGE = {"data": [
    {"pts": 88, "fga": 70, "fta": 18, "oreb": 9, "turnover": 12, "team": {"id": 1},
     "game": {"id": 5001, "date": "2026-07-20", "home_team_id": 1, "visitor_team_id": 2,
              "home_team_score": 88, "visitor_team_score": 78}},
], "meta": {"next_cursor": None}}

_GAMES_PAGE = {"data": [
    {"id": 5003, "home_team_id": 1, "visitor_team_id": 2},
], "meta": {"next_cursor": None}}


def _fake_get(path, params=None, ttl=900):
    if path == "/teams":
        return _TEAMS_PAGE
    if path == "/players":
        return _PLAYERS_PAGE
    if path == "/stats" and params and "player_ids[]" in params:
        return _STATS_PAGE
    if path == "/stats" and params and "team_ids[]" in params:
        return _TEAM_STATS_PAGE
    if path == "/games":
        return _GAMES_PAGE
    return None


def test_no_api_key_degrades_to_empty_everywhere(monkeypatch):
    # Deliberately does NOT monkeypatch _get — exercises the real _key() guard, which must
    # return None before any network call happens (no real HTTP call is made by this test).
    monkeypatch.delenv("BALLDONTLIE_API_KEY", raising=False)
    adapter = B.BallDontLie()
    assert adapter.teams("WNBA") == {}
    assert adapter.players("WNBA") == {}
    assert adapter.gamelog("WNBA", "101") == []
    assert adapter.injuries("WNBA") == {}


def test_missing_api_key_prints_a_diagnostic_exactly_once_per_process(monkeypatch, capsys):
    monkeypatch.delenv("BALLDONTLIE_API_KEY", raising=False)
    monkeypatch.setattr(B, "_warned_no_key", False)
    adapter = B.BallDontLie()
    adapter.teams("WNBA")
    adapter.players("WNBA")
    adapter.gamelog("WNBA", "101")
    out = capsys.readouterr().out
    assert out.count("BALLDONTLIE_API_KEY is not set") == 1, (
        "must warn once per process, not once per request -- a refresh cycle makes "
        "hundreds of calls while the key is unset")


def test_team_table_and_assets(monkeypatch):
    monkeypatch.setattr(B, "_get", _fake_get)
    adapter = B.BallDontLie()
    t = adapter._team_table("WNBA")
    assert t["1"]["name"] == "Las Vegas Aces" and t["1"]["abbr"] == "LV"
    assets = adapter.team_assets("WNBA")
    assert assets[B._norm_name("Las Vegas Aces")]["id"] == "1"
    assert assets[B._norm_name("LV")]["id"] == "1"
    # No image field in balldontlie's payload — logo must stay honestly None, never guessed.
    assert assets[B._norm_name("LV")]["logo"] is None


def test_players_maps_position_and_team(monkeypatch):
    monkeypatch.setattr(B, "_get", _fake_get)
    adapter = B.BallDontLie()
    players = adapter.players("WNBA")
    wilson = players[B._norm_name("A'ja Wilson")]
    assert wilson.id == "101" and wilson.position == "F" and wilson.team_id == "1"


def test_gamelog_parses_mmss_minutes_and_drops_dnp(monkeypatch):
    monkeypatch.setattr(B, "_get", _fake_get)
    adapter = B.BallDontLie()
    games = adapter.gamelog("WNBA", "101")
    assert len(games) == 1
    g = games[0]
    assert g.minutes == pytest.approx(32.25)
    assert g.pts == 24 and g.orb == 2 and g.drb == 7


def test_minutes_parses_plain_numeric_and_empty():
    assert B._minutes("32") == 32.0
    assert B._minutes("32:30") == pytest.approx(32.5)
    assert B._minutes(None) == 0.0
    assert B._minutes("") == 0.0
    assert B._minutes("DNP") == 0.0


def test_team_pace_sums_player_rows_and_reads_opponent_score_from_game(monkeypatch):
    monkeypatch.setattr(B, "_get", _fake_get)
    adapter = B.BallDontLie()
    tp = adapter.team_pace("WNBA", "1")
    assert tp is not None and tp.games == 1
    # home team (id 1) allowed the visitor's score (78)
    assert tp.pace > 0 and tp.off_rtg > tp.def_rtg


def test_upcoming_opponents_maps_both_directions(monkeypatch):
    monkeypatch.setattr(B, "_get", _fake_get)
    adapter = B.BallDontLie()
    opp = adapter.upcoming_opponents("WNBA")
    assert opp == {"1": "2", "2": "1"}


def test_injuries_and_headshots_are_honestly_empty(monkeypatch):
    monkeypatch.setattr(B, "_get", _fake_get)
    adapter = B.BallDontLie()
    assert adapter.injuries("WNBA") == {}
    assert adapter.all_headshots("WNBA") == {}


def test_paged_stops_on_missing_data_key(monkeypatch):
    monkeypatch.setattr(B, "_get", lambda path, params=None, ttl=900: {"unexpected": "shape"})
    assert B._paged("/teams", {}, 900) == []


def test_get_returns_none_on_http_error(monkeypatch):
    class _Resp:
        status_code = 429
        text = "rate limited"

    monkeypatch.setattr(B.requests, "get", lambda *a, **k: _Resp())
    assert B._get("/teams") is None
