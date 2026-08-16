"""
Offline parsing tests for the stats.wnba.com/cdn.wnba.com adapter that replaced ESPN as
WNBA's data source (see basketball/data/wnba_stats.py's module docstring for why). All
deterministic — `_get` is monkeypatched, no network, mirroring test_espn_gamelog.py's
established pattern for this package.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from basketball.data import wnba_stats as W


@pytest.fixture(autouse=True)
def _clear_memo_cache():
    # _team_table/players/team_pace/etc each memoize under their own key in the SAME
    # module-level dict `_get` also (would) cache into — without clearing, one test's
    # cached result leaks into the next test's differently-shaped fake data.
    W._memory.clear()
    yield
    W._memory.clear()


def _envelope(headers, rows):
    return {"resultSets": [{"headers": headers, "rowSet": rows}]}


_TEAM_STATS = _envelope(
    ["TEAM_ID", "TEAM_NAME"],
    [["17", "Las Vegas Aces"], ["9", "New York Liberty"]],
)

_ROSTER = _envelope(
    ["PLAYER_ID", "PLAYER", "POSITION"],
    [["201", "A'ja Wilson", "F"], ["202", "Chelsea Gray", "G"]],
)

_GAMELOG = _envelope(
    ["GAME_DATE", "MATCHUP", "MIN", "PTS", "REB", "OREB", "DREB", "AST", "STL", "BLK",
     "TOV", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "PF"],
    [
        ["2026-07-20", "LVA vs. NYL", "32", "24", "9", "2", "7", "3", "1", "2", "2",
         "9", "17", "1", "3", "5", "6", "2"],
        ["2026-07-15", "LVA @ SEA", "0", "0", "0", "0", "0", "0", "0", "0", "0",
         "0", "0", "0", "0", "0", "0", "0"],   # DNP — must be dropped (minutes == 0)
    ],
)

_TEAM_GAMELOG = _envelope(
    ["GAME_DATE", "FGA", "FTA", "OREB", "TOV", "PTS", "PLUS_MINUS"],
    [
        ["2026-07-20", "70", "18", "9", "12", "88", "10"],
        ["2026-07-17", "68", "16", "8", "14", "82", "-4"],
    ],
)

_SCOREBOARD = _envelope(["HOME_TEAM_ID", "VISITOR_TEAM_ID"], [["17", "9"]])


def _fake_get(url, ttl=900):
    if "leaguedashteamstats" in url:
        return _TEAM_STATS
    if "commonteamroster" in url:
        return _ROSTER
    if "playergamelog" in url:
        return _GAMELOG
    if "teamgamelog" in url:
        return _TEAM_GAMELOG
    if "scoreboardv2" in url:
        return _SCOREBOARD
    return None


def test_team_table_maps_real_ids_to_static_identity(monkeypatch):
    monkeypatch.setattr(W, "_get", _fake_get)
    adapter = W.WnbaStats()
    t = adapter._team_table("WNBA")
    assert t["17"]["name"] == "Las Vegas Aces" and t["17"]["abbr"] == "LV"
    assert t["9"]["name"] == "New York Liberty"


def test_team_assets_keys_by_name_and_abbr_and_alias(monkeypatch):
    monkeypatch.setattr(W, "_get", _fake_get)
    adapter = W.WnbaStats()
    assets = adapter.team_assets("WNBA")
    assert assets[W._norm_name("Las Vegas Aces")]["id"] == "17"
    assert assets[W._norm_name("LV")]["id"] == "17"
    assert assets[W._norm_name("las vegas")]["id"] == "17"   # alias table


def test_roster_maps_position_first_letter(monkeypatch):
    monkeypatch.setattr(W, "_get", _fake_get)
    adapter = W.WnbaStats()
    refs = adapter._roster("WNBA", "17", "Las Vegas Aces")
    assert {r.name for r in refs} == {"A'ja Wilson", "Chelsea Gray"}
    wilson = next(r for r in refs if r.name == "A'ja Wilson")
    assert wilson.position == "F" and wilson.team_id == "17"


def test_gamelog_drops_dnp_rows_and_parses_the_matchup_opponent(monkeypatch):
    monkeypatch.setattr(W, "_get", _fake_get)
    adapter = W.WnbaStats()
    games = adapter.gamelog("WNBA", "201")
    assert len(games) == 1
    g = games[0]
    assert g.opp == "NYL" and g.pts == 24 and g.orb == 2 and g.drb == 7
    assert g.minutes == 32


def test_gamelog_sorts_most_recent_first(monkeypatch):
    two_games = _envelope(
        _GAMELOG["resultSets"][0]["headers"],
        [
            ["2026-07-10", "LVA vs. NYL", "30", "10", "5", "1", "4", "2", "1", "1", "1",
             "4", "8", "1", "2", "1", "2", "1"],
            ["2026-07-20", "LVA @ SEA", "28", "18", "6", "2", "4", "3", "0", "0", "2",
             "7", "14", "0", "1", "4", "4", "3"],
        ],
    )
    monkeypatch.setattr(W, "_get", lambda url, ttl=900: two_games if "playergamelog" in url else None)
    adapter = W.WnbaStats()
    games = adapter.gamelog("WNBA", "201")
    assert [g.date for g in games] == ["2026-07-20", "2026-07-10"]


def test_team_pace_derives_opponent_points_from_plus_minus(monkeypatch):
    monkeypatch.setattr(W, "_get", _fake_get)
    adapter = W.WnbaStats()
    tp = adapter.team_pace("WNBA", "17")
    assert tp is not None
    assert tp.games == 2
    # game 1: pts=88, plus_minus=10 -> allowed 78; game 2: pts=82, plus_minus=-4 -> allowed 86
    assert tp.off_rtg > 0 and tp.def_rtg > 0


def test_upcoming_opponents_maps_both_directions(monkeypatch):
    monkeypatch.setattr(W, "_get", _fake_get)
    adapter = W.WnbaStats()
    opp = adapter.upcoming_opponents("WNBA")
    assert opp == {"17": "9", "9": "17"}


def test_injuries_is_honestly_empty_not_fabricated(monkeypatch):
    monkeypatch.setattr(W, "_get", _fake_get)
    adapter = W.WnbaStats()
    assert adapter.injuries("WNBA") == {}


def test_all_headshots_keyed_by_normalized_player_name(monkeypatch):
    monkeypatch.setattr(W, "_get", _fake_get)
    adapter = W.WnbaStats()
    heads = adapter.all_headshots("WNBA")
    assert W._norm_name("A'ja Wilson") in heads
    assert heads[W._norm_name("A'ja Wilson")].endswith("/201.png")


def test_get_handles_a_missing_resultset_without_crashing(monkeypatch):
    monkeypatch.setattr(W, "_get", lambda url, ttl=900: {"resultSets": []})
    adapter = W.WnbaStats()
    assert adapter.gamelog("WNBA", "999") == []
    assert adapter._team_table("WNBA") == {}
