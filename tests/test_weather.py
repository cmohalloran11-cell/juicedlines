"""
Tests for weather.slate()'s wall-clock budget.

Regression guard for a real incident: with real Xweather credentials configured for the
first time, a degraded/slow provider made slate()'s per-team loop (each call individually
timeout-bounded, but the LOOP itself uncapped) stall a full build cycle for 12+ minutes.
The fix caps total loop time at _SLATE_BUDGET regardless of how many teams remain to try.
"""
from __future__ import annotations

import weather


def _lines(teams):
    return [{"sport": "MLB", "team": t, "start_time": None} for t in teams]


def test_slate_stops_once_the_time_budget_is_exhausted(monkeypatch):
    monkeypatch.setattr(weather, "available", lambda: True)
    monkeypatch.setattr(weather, "for_team", lambda team: {"tempF": 70})
    # Simulate: start at t=0, then every time.time() call after the first advances past budget.
    calls = {"n": 0}

    def fake_time():
        calls["n"] += 1
        return 0.0 if calls["n"] == 1 else weather._SLATE_BUDGET + 1
    monkeypatch.setattr(weather.time, "time", fake_time)

    lines = _lines(["ARI", "ATL", "BAL", "BOS", "CHC"])
    out = weather.slate(lines, limit=16)
    assert out["available"] is True
    # Budget exhausted after the very first check inside the loop → at most 1 game fetched.
    assert len(out["games"]) <= 1


def test_slate_returns_all_games_when_within_budget(monkeypatch):
    monkeypatch.setattr(weather, "available", lambda: True)
    monkeypatch.setattr(weather, "for_team", lambda team: {"tempF": 70})
    monkeypatch.setattr(weather.time, "time", lambda: 0.0)  # no time ever elapses
    lines = _lines(["ARI", "ATL", "BAL"])
    out = weather.slate(lines, limit=16)
    assert len(out["games"]) == 3


def test_slate_unavailable_without_credentials(monkeypatch):
    monkeypatch.setattr(weather, "available", lambda: False)
    out = weather.slate(_lines(["ARI"]))
    assert out["available"] is False and out["games"] == []
