"""
Offline parsing tests for the-odds-api CFB adapter — `_get` is monkeypatched, no network,
mirroring cfb/tests/test_cfbd_client.py / basketball/tests/test_balldontlie.py's pattern.
"""
from __future__ import annotations

import pytest

from cfb.data import odds_provider as O
from cfb.config import ODDS_MARKET_TO_STAT


@pytest.fixture(autouse=True)
def _clear_memo(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    O._memory.clear()
    yield
    O._memory.clear()


_EVENTS = [
    {"id": "evt1", "commence_time": "2026-08-30T19:30:00Z",
     "home_team": "Ohio State Buckeyes", "away_team": "Michigan Wolverines"},
]

_EVENT_ODDS = {
    "id": "evt1", "commence_time": "2026-08-30T19:30:00Z",
    "home_team": "Ohio State Buckeyes", "away_team": "Michigan Wolverines",
    "bookmakers": [
        {"key": "draftkings", "title": "DraftKings", "markets": [
            {"key": "player_pass_yds", "outcomes": [
                {"name": "Over", "description": "Will Howard", "price": -115, "point": 245.5},
                {"name": "Under", "description": "Will Howard", "price": -105, "point": 245.5},
            ]},
            {"key": "player_anytime_td", "outcomes": [
                {"name": "Over", "description": "TreVeyon Henderson", "price": 120, "point": 0.5},
            ]},
        ]},
    ],
}


def _fake_get(path, params=None, ttl=60.0):
    if path.endswith("/events"):
        return _EVENTS
    if path.endswith("/odds"):
        return _EVENT_ODDS
    return None


def test_no_api_key_degrades_to_empty(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    adapter = O.TheOddsApiAdapter()
    assert adapter.events() == []
    assert adapter.player_props("evt1", list(ODDS_MARKET_TO_STAT)) == []


def test_missing_api_key_prints_diagnostic_once(monkeypatch, capsys):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.setattr(O, "_warned_no_key", False)
    adapter = O.TheOddsApiAdapter()
    adapter.events()
    adapter.player_props("evt1", ["player_pass_yds"])
    out = capsys.readouterr().out
    assert out.count("ODDS_API_KEY is not set") == 1


def test_events_parses_id_and_teams(monkeypatch):
    monkeypatch.setattr(O, "_get", _fake_get)
    events = O.TheOddsApiAdapter().events()
    assert events == [{"id": "evt1", "commence_time": "2026-08-30T19:30:00Z",
                       "home_team": "Ohio State Buckeyes", "away_team": "Michigan Wolverines"}]


def test_player_props_pairs_over_under_by_player_and_market(monkeypatch):
    monkeypatch.setattr(O, "_get", _fake_get)
    props = O.TheOddsApiAdapter().player_props("evt1", list(ODDS_MARKET_TO_STAT))
    by_key = {(p.player, p.market): p for p in props}

    howard = by_key[("Will Howard", "player_pass_yds")]
    assert howard.line == 245.5 and howard.book == "draftkings"
    assert howard.over_price == "-115" and howard.under_price == "-105"

    henderson = by_key[("TreVeyon Henderson", "player_anytime_td")]
    assert henderson.line == 0.5 and henderson.over_price == "+120"
    assert henderson.under_price is None   # book only posted the Over side


def test_player_props_ignores_markets_not_requested(monkeypatch):
    monkeypatch.setattr(O, "_get", _fake_get)
    props = O.TheOddsApiAdapter().player_props("evt1", ["player_rush_yds"])
    assert props == []   # neither market in the fixture is player_rush_yds


def test_get_returns_none_on_http_error(monkeypatch):
    class _Resp:
        status_code = 401
        text = "bad key"

    monkeypatch.setattr(O.requests, "get", lambda *a, **k: _Resp())
    assert O._get("/sports/x/events") is None
