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
from basketball.model import rates as R, priors as PR

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


# ── gamelog ordering ──────────────────────────────────────────────────────────
# GameLogSource.gamelog's contract is most-recent-first, and fit_rates/project_minutes
# weight purely by list index (0.5**(i/halflife)) — so an unsorted feed order doesn't
# degrade the recency weighting, it INVERTS it. balldontlie, wnba_stats and this adapter's
# own _boxscore_index all sort; _athlete_gamelog did not, leaving the whole weighting at
# the mercy of ESPN's nested seasonTypes→categories→events iteration order.

_ORDERED = {
    "names": ["minutes", "points"],
    "events": {
        "1": {"gameDate": "2026-07-05T00:00:00Z", "opponent": {"id": "9", "displayName": "NY"}},
        "2": {"gameDate": "2026-07-12T00:00:00Z", "opponent": {"id": "9", "displayName": "NY"}},
        "3": {"gameDate": "2026-07-19T00:00:00Z", "opponent": {"id": "9", "displayName": "NY"}},
    },
    # emitted OLDEST-first, split across two seasonTypes (ESPN's real nesting)
    "seasonTypes": [
        {"categories": [{"events": [{"eventId": "1", "stats": ["30", "4"]},
                                    {"eventId": "2", "stats": ["30", "4"]}]}]},
        {"categories": [{"events": [{"eventId": "3", "stats": ["30", "26"]}]}]},
    ],
}


def _fake_get_ordered(url, ttl=1800):
    return _REAL_TEAMS if "/teams" in url else (_ORDERED if "/gamelog" in url else None)


def test_athlete_gamelog_returns_most_recent_first(monkeypatch):
    monkeypatch.setattr(E, "_get", _fake_get_ordered)
    games = E.EspnBasketball()._athlete_gamelog("WNBA", "player1")
    assert [g.date for g in games] == ["2026-07-19", "2026-07-12", "2026-07-05"]


def test_athlete_gamelog_order_puts_the_recency_weight_on_the_newest_game(monkeypatch):
    monkeypatch.setattr(E, "_get", _fake_get_ordered)
    games = E.EspnBasketball()._athlete_gamelog("WNBA", "player1")
    prior = PR.positional_prior_poss("G", 96.0, "WNBA")
    weighted = R.fit_rates(games, "WNBA", prior, 40, 96.0, 0.0, 1.0).per_poss["pts"]
    flat = sum(g.pts for g in games) / sum(
        R.player_possessions(g.minutes, 40, 96.0) for g in games)
    # 26-point game is the newest, so a half-life of 1 game must pull the rate ABOVE the
    # unweighted average; served oldest-first it landed below it instead.
    assert weighted > flat
