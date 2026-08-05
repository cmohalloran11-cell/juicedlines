"""
Tests for analytics.py's player name resolution — specifically the last-name+team fallback
added 2026-08-05 after a live production audit found real active players (Vladimir Guerrero
Jr., listed by prop feeds as "Vlad Guerrero Jr.") going entirely unprojected because a feed's
short/nickname first name doesn't match statsapi's official fullName even after
mlb._norm_name's accent/suffix normalization.
"""
from __future__ import annotations

import pytest

import analytics


@pytest.fixture(autouse=True)
def _clear_cache():
    # _player_meta/_last_name_index are memoized in analytics._CACHE by season-scoped key;
    # without clearing it, one test's monkeypatched _player_meta would leak into the next.
    analytics._CACHE.clear()
    yield
    analytics._CACHE.clear()


def _fake_player_meta(monkeypatch):
    meta = {
        "vladimir guerrero": [{"id": 665489, "name": "Vladimir Guerrero Jr.",
                               "pos_type": "Infielder", "pos_abbr": "1B", "team_id": 141}],
        "max muncy": [
            {"id": 1, "name": "Max Muncy", "pos_type": "Infielder", "team_id": 119},   # LAD
            {"id": 2, "name": "Max Muncy", "pos_type": "Infielder", "team_id": 133},   # ATH
        ],
    }
    monkeypatch.setattr(analytics, "_player_meta", lambda: meta)
    teams = {"TOR": {"id": 141, "name": "Blue Jays", "abbr": "TOR"},
             "LAD": {"id": 119, "name": "Dodgers", "abbr": "LAD"},
             "ATH": {"id": 133, "name": "Athletics", "abbr": "ATH"},
             "_by_id": {141: {"abbr": "TOR"}, 119: {"abbr": "LAD"}, 133: {"abbr": "ATH"}}}
    monkeypatch.setattr(analytics, "_team_map", lambda: teams)


def test_nickname_resolves_via_last_name_team_fallback(monkeypatch):
    _fake_player_meta(monkeypatch)
    m = analytics._resolve_player("Vlad Guerrero Jr.", "TOR", False)
    assert m is not None and m["id"] == 665489


def test_official_full_name_still_resolves_directly(monkeypatch):
    # Regression guard: a name that already matches exactly must keep working unchanged —
    # the fallback only activates when the exact lookup already came up empty.
    _fake_player_meta(monkeypatch)
    m = analytics._resolve_player("Vladimir Guerrero Jr.", "TOR", False)
    assert m is not None and m["id"] == 665489


def test_fallback_requires_team_to_disambiguate(monkeypatch):
    # No team_abbr -> can't safely narrow a last-name-only match, so it stays None rather
    # than guessing at one of potentially many same-surname players league-wide.
    _fake_player_meta(monkeypatch)
    assert analytics._resolve_player("Vlad Guerrero Jr.", None, False) is None


def test_nonexistent_player_stays_none(monkeypatch):
    _fake_player_meta(monkeypatch)
    assert analytics._resolve_player("Totally Fake Player", "TOR", False) is None


def test_fallback_disambiguates_shared_last_names_by_team(monkeypatch):
    _fake_player_meta(monkeypatch)
    # Two "Max Muncy"s (real case this codebase already handles for the exact-name path —
    # see _resolve_player's docstring) — the fallback path must respect team scoping too.
    lad = analytics._resolve_player("Muncy", "LAD", False)
    ath = analytics._resolve_player("Muncy", "ATH", False)
    assert lad is not None and lad["id"] == 1
    assert ath is not None and ath["id"] == 2


def test_attach_projections_returns_subsport_errors_instead_of_swallowing_them(monkeypatch):
    # 2026-08-05 finding: attach_basketball raised for a full production cycle (0/2650 WNBA
    # props projected) and the only trace was a print() into a CI log nobody could read.
    # attach_projections must now surface that failure to its caller.
    monkeypatch.setattr(analytics, "_MLB_OK", False)

    def _boom(lines):
        raise RuntimeError("espn unavailable")
    monkeypatch.setattr("tennis.board.attach_tennis", lambda lines: None, raising=False)
    monkeypatch.setattr("basketball.board.attach_basketball", _boom, raising=False)
    monkeypatch.setattr("provenance.stamp_lines", lambda lines: None, raising=False)

    errs = analytics.attach_projections([])
    assert errs.get("basketball") == "espn unavailable"
    assert "tennis" not in errs
