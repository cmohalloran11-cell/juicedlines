"""
Regression test: the WNBA/basketball ESPN adapter must exclude exhibition games (the
All-Star Game) from a player's game log and from the completed-games list.

Real incident: ESPN gives the All-Star Game no distinct season/game type — it's just
another event, pitting fake draft "teams" (e.g. "Team Coop" vs "Team Spoon") against
each other. Left unfiltered, it was pulled into every All-Star's recency-weighted game
log as if it were a normal game (reduced/atypical minutes, no real defense), corrupting
their projections right after the break. The fix: only a real franchise's roster
endpoint lists real team ids, so any competitor NOT in that set is an exhibition.

All deterministic / offline — _get is monkeypatched, no network.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from basketball.data import espn as E

# Two real WNBA franchises + one fake "All-Star draft team" (mirrors ESPN's real shape).
_REAL_TEAMS = {
    "sports": [{"leagues": [{"teams": [
        {"team": {"id": "17", "displayName": "Las Vegas Aces", "abbreviation": "LV"}},
        {"team": {"id": "9", "displayName": "New York Liberty", "abbreviation": "NY"}},
    ]}]}]
}

_GAMELOG = {
    "names": ["minutes", "points"],
    "events": {
        "1": {"gameDate": "2026-07-20T00:00:00Z", "opponent": {"id": "9", "displayName": "NY"}},
        "2": {"gameDate": "2026-07-26T00:00:00Z", "opponent": {"id": "133384", "displayName": "Team Coop"}},
    },
    "seasonTypes": [{"categories": [{"events": [
        {"eventId": "1", "stats": ["32", "20"]},   # real game — should survive
        {"eventId": "2", "stats": ["18", "9"]},    # All-Star Game — should be dropped
    ]}]}],
}

_SCOREBOARD = {
    "events": [
        {"id": "gid1", "competitions": [{
            "status": {"type": {"state": "post"}},
            "competitors": [{"team": {"id": "17"}}, {"team": {"id": "9"}}],
        }]},
        {"id": "gid2", "competitions": [{   # the All-Star Game — both competitors fake
            "status": {"type": {"state": "post"}},
            "competitors": [{"team": {"id": "133384"}}, {"team": {"id": "133383"}}],
        }]},
    ]
}


def _fake_get(url, ttl=1800):
    if "/teams" in url:
        return _REAL_TEAMS
    if "/gamelog" in url:
        return _GAMELOG
    if "/scoreboard" in url:
        return _SCOREBOARD
    return None


def test_athlete_gamelog_excludes_the_all_star_game(monkeypatch):
    monkeypatch.setattr(E, "_get", _fake_get)
    adapter = E.EspnBasketball()
    games = adapter._athlete_gamelog("WNBA", "player1")
    assert len(games) == 1
    assert games[0].opp == "NY"


def test_completed_games_excludes_the_all_star_game(monkeypatch):
    monkeypatch.setattr(E, "_get", _fake_get)
    adapter = E.EspnBasketball()
    ids = adapter._completed_games("WNBA")
    assert ids == ["gid1"]


def test_valid_team_ids_matches_the_real_roster(monkeypatch):
    monkeypatch.setattr(E, "_get", _fake_get)
    adapter = E.EspnBasketball()
    assert adapter._valid_team_ids("WNBA") == {"17", "9"}
