"""
Tests for cfb.lines.fetch_cfb_props — the Odds-API-to-board-Line-dict transform, and the
"only what a book actually posted" gating rule.
"""
from __future__ import annotations

import pytest

import cfb
import store
from store.database import SQLiteDatabase
from cfb import lines as L
from cfb.data.odds_provider import PlayerProp
from cfb.repositories import PlayerRepository, UnmatchedPlayerRepository


@pytest.fixture(autouse=True)
def _temp_store(tmp_path, monkeypatch):
    store.reset_singleton()
    db = SQLiteDatabase(tmp_path / "cfb_lines_test.db")
    cfb.init_schema(db)
    monkeypatch.setattr(store, "get_database", lambda url=None: db)
    yield db
    store.reset_singleton()


def test_no_odds_api_key_returns_empty_no_error(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    out, err = L.fetch_cfb_props()
    assert out == [] and err is None


def test_sport_filter_for_a_different_sport_short_circuits(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    out, err = L.fetch_cfb_props(sport_filter="MLB")
    assert out == [] and err is None


class _FakeAdapter:
    def events(self):
        return [{"id": "evt1", "commence_time": "2026-08-30T19:30:00Z",
                "home_team": "Ohio State", "away_team": "Michigan"}]

    def player_props(self, event_id, markets):
        return [PlayerProp(event_id="evt1", commence_time="2026-08-30T19:30:00Z",
                          home_team="Ohio State", away_team="Michigan",
                          player="Will Howard", market="player_pass_yds", book="draftkings",
                          line=245.5, over_price="-115", under_price="-105")]


def test_builds_a_board_line_dict_for_a_posted_market(monkeypatch, _temp_store):
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    monkeypatch.setattr(L, "TheOddsApiAdapter", lambda: _FakeAdapter())

    out, err = L.fetch_cfb_props()
    assert err is None
    assert len(out) == 1
    row = out[0]
    assert row["sport"] == "CFB"
    assert row["player"] == "Will Howard"
    assert row["stat_type"] == "Passing Yards"
    assert row["line"] == 245.5
    assert row["source"] == "draftkings"
    assert row["over_price"] == "-115" and row["under_price"] == "-105"
    assert row["over_implied"] is not None and row["under_implied"] is not None
    assert row["matchup"] == "Michigan @ Ohio State"
    # unresolved player (no canonical row exists yet) -- team/position honestly None, not
    # guessed, and the raw name got logged for human review instead of silently dropped.
    assert row["team"] is None and row["position"] is None
    assert len(UnmatchedPlayerRepository(_temp_store).list_pending()) == 1


def test_resolved_player_gets_team_and_position_from_canonical_row(monkeypatch, _temp_store):
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    monkeypatch.setattr(L, "TheOddsApiAdapter", lambda: _FakeAdapter())
    PlayerRepository(_temp_store).create(full_name="Will Howard", position="QB", team="Ohio State")

    out, err = L.fetch_cfb_props()
    assert err is None
    row = out[0]
    assert row["team"] == "Ohio State" and row["position"] == "QB"
    assert UnmatchedPlayerRepository(_temp_store).list_pending() == []


class _EmptyMarketAdapter:
    def events(self):
        return [{"id": "evt1", "commence_time": "2026-08-30T19:30:00Z",
                "home_team": "Ohio State", "away_team": "Michigan"}]

    def player_props(self, event_id, markets):
        return []   # no book posted any of the requested markets for this event


def test_no_props_posted_yields_no_lines(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    monkeypatch.setattr(L, "TheOddsApiAdapter", lambda: _EmptyMarketAdapter())
    out, err = L.fetch_cfb_props()
    assert out == [] and err is None
