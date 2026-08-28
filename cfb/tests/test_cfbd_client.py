"""
Offline parsing tests for the CFBD REST adapter — `_get` is monkeypatched, no network,
mirroring basketball/tests/test_balldontlie.py's established pattern for this repo's
keyed-adapter modules (warn-once-per-process on a missing key, honest-empty on any failure).
"""
from __future__ import annotations

import pytest

from cfb.data import cfbd_client as C


@pytest.fixture(autouse=True)
def _clear_memo(monkeypatch):
    monkeypatch.setenv("CFBD_API_KEY", "test-key")
    C._memory.clear()
    yield
    C._memory.clear()


_TEAMS = [
    {"id": 1, "school": "Ohio State", "conference": "Big Ten", "classification": "fbs",
     "abbreviation": "OSU"},
    {"id": 2, "school": "Michigan", "conference": "Big Ten", "classification": "fbs",
     "abbreviation": "MICH"},
    {"id": 3, "school": "Some FCS School", "conference": "FCS Conf", "classification": "fcs",
     "abbreviation": "SFC"},
]

_ROSTER = [
    {"id": 101, "first_name": "Will", "last_name": "Howard", "position": "QB", "jersey": 18},
    {"id": 102, "first_name": "TreVeyon", "last_name": "Henderson", "position": "RB", "jersey": 32},
]

_GAMES = [
    {"id": 5001, "season": 2026, "week": 1, "seasonType": "regular",
     "startDate": "2026-08-30T19:30:00.000Z", "homeTeam": "Ohio State",
     "awayTeam": "Michigan", "homeClassification": "fbs", "awayClassification": "fbs",
     "completed": False},
]

_LINES = [
    {"id": 5001, "lines": [{"provider": "DraftKings", "spread": -6.5, "overUnder": 54.5}]},
]

_GAME_PLAYERS = [
    {"id": 5001, "teams": [
        {"team": "Ohio State", "category": "passing", "types": [
            {"name": "C/ATT", "athletes": [{"id": 101, "name": "Will Howard", "stat": "22/30"}]},
            {"name": "YDS", "athletes": [{"id": 101, "name": "Will Howard", "stat": "280"}]},
            {"name": "TD", "athletes": [{"id": 101, "name": "Will Howard", "stat": "3"}]},
            {"name": "INT", "athletes": [{"id": 101, "name": "Will Howard", "stat": "1"}]},
        ]},
        {"team": "Ohio State", "category": "rushing", "types": [
            {"name": "CAR", "athletes": [{"id": 102, "name": "TreVeyon Henderson", "stat": "18"}]},
            {"name": "YDS", "athletes": [{"id": 102, "name": "TreVeyon Henderson", "stat": "112"}]},
            {"name": "TD", "athletes": [{"id": 102, "name": "TreVeyon Henderson", "stat": "2"}]},
        ]},
    ]},
]

_ADVANCED = [
    {"gameId": 5001, "week": 1, "team": "Ohio State", "opponent": "Michigan",
     "offense": {"ppa": 0.31, "successRate": 0.47}, "defense": {"ppa": 0.02, "successRate": 0.33},
     "plays": 68},
]


def _fake_get(path, params=None, ttl=900):
    if path == "/teams":
        return _TEAMS
    if path == "/roster":
        return _ROSTER
    if path == "/games":
        return _GAMES
    if path == "/lines":
        return _LINES
    if path == "/games/players":
        return _GAME_PLAYERS
    if path == "/stats/game/advanced":
        return _ADVANCED
    return None


def test_no_api_key_degrades_to_empty_everywhere(monkeypatch):
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    client = C.CFBDClient()
    assert client.teams(2026) == []
    assert client.roster("Ohio State", 2026) == []
    assert client.schedule(2026) == []
    assert client.player_game_stats(2026) == []
    assert client.team_efficiency(2026) == []


def test_missing_api_key_prints_diagnostic_once(monkeypatch, capsys):
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    monkeypatch.setattr(C, "_warned_no_key", False)
    client = C.CFBDClient()
    client.teams(2026)
    client.roster("Ohio State", 2026)
    out = capsys.readouterr().out
    assert out.count("CFBD_API_KEY is not set") == 1


def test_teams_filters_to_fbs_by_default(monkeypatch):
    monkeypatch.setattr(C, "_get", _fake_get)
    teams = C.CFBDClient().teams(2026)
    assert {t.school for t in teams} == {"Ohio State", "Michigan"}
    assert all(t.classification == "fbs" for t in teams)


def test_teams_classification_filter_can_be_widened(monkeypatch):
    monkeypatch.setattr(C, "_get", _fake_get)
    teams = C.CFBDClient().teams(2026, classification="fcs")
    assert [t.school for t in teams] == ["Some FCS School"]


def test_roster_parses_names_and_positions(monkeypatch):
    monkeypatch.setattr(C, "_get", _fake_get)
    roster = C.CFBDClient().roster("Ohio State", 2026)
    assert len(roster) == 2
    qb = next(p for p in roster if p.id == "101")
    assert qb.name == "Will Howard" and qb.position == "QB" and qb.jersey == 18


def test_schedule_merges_market_spread_and_total(monkeypatch):
    monkeypatch.setattr(C, "_get", _fake_get)
    games = C.CFBDClient().schedule(2026, week=1)
    assert len(games) == 1
    g = games[0]
    assert g.home_team == "Ohio State" and g.away_team == "Michigan"
    assert g.spread == -6.5 and g.over_under == 54.5


def test_player_game_stats_merges_passing_and_rushing_categories(monkeypatch):
    monkeypatch.setattr(C, "_get", _fake_get)
    rows = C.CFBDClient().player_game_stats(2026, week=1)
    by_id = {r.player_id: r for r in rows}
    qb = by_id["101"]
    assert qb.pass_completions == 22 and qb.pass_attempts == 30
    assert qb.pass_yards == 280 and qb.pass_tds == 3 and qb.interceptions == 1
    rb = by_id["102"]
    assert rb.rush_attempts == 18 and rb.rush_yards == 112 and rb.rush_tds == 2
    assert rb.stat("Rushing Yards") == 112
    assert qb.stat("Passing Yards") == 280
    assert qb.stat("Anytime TD") == 0.0   # a passing TD alone doesn't count as anytime TD


def test_anytime_td_stat_true_when_rush_or_rec_td_present(monkeypatch):
    monkeypatch.setattr(C, "_get", _fake_get)
    rows = C.CFBDClient().player_game_stats(2026, week=1)
    rb = next(r for r in rows if r.player_id == "102")
    assert rb.stat("Anytime TD") == 1.0


def test_team_efficiency_parses_offense_and_defense(monkeypatch):
    monkeypatch.setattr(C, "_get", _fake_get)
    eff = C.CFBDClient().team_efficiency(2026, week=1)
    assert len(eff) == 1
    row = eff[0]
    assert row.offense_ppa == 0.31 and row.defense_ppa == 0.02
    assert row.offense_success_rate == 0.47 and row.plays == 68


def test_get_returns_none_on_http_error(monkeypatch):
    class _Resp:
        status_code = 429
        text = "rate limited"

    monkeypatch.setattr(C.requests, "get", lambda *a, **k: _Resp())
    assert C._get("/teams") is None


def test_get_returns_none_on_network_exception(monkeypatch):
    def _raise(*a, **k):
        raise ConnectionError("boom")

    monkeypatch.setattr(C.requests, "get", _raise)
    assert C._get("/teams") is None
