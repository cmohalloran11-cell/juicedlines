"""
_full_logs() feeds the drawer's "Recent Games & Hit Rate" section straight from
statsapi.mlb.com, separately from whatever data source produced the board's own
model_proj/model_n (a different pipeline -- see analytics.py's module docstring). It only
had diagnostics for two failure shapes (request exception, unresolved personId) -- a THIRD
shape went completely silent: a personId that resolves fine but whose gameLog hydrate comes
back with an empty splits array, observed live for an established player the model itself
had 30 games of history for. That's statsapi under-hydrating under the bulk sequential load
a full build puts on it (~5000 lookups in a few minutes), not a real zero-game player. This
adds a bounded one-shot retry plus a diagnostic for whichever way it resolves.
"""
from __future__ import annotations

import pytest

import analytics
import mlb_model as mlb


@pytest.fixture(autouse=True)
def _clear_cache(monkeypatch):
    analytics._CACHE.clear()
    monkeypatch.setattr(analytics.time, "sleep", lambda s: None)   # no real delay in tests
    yield
    analytics._CACHE.clear()


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


_ONE_GAME = {"people": [{"stats": [{"splits": [
    {"date": "2026-08-20", "opponent": {"id": 111, "name": "Rays"}, "isHome": True,
     "stat": {"runs": 1}},
]}]}]}
_EMPTY_SPLITS = {"people": [{"stats": [{"splits": []}]}]}
_NO_PEOPLE = {"people": []}


def test_retries_once_and_recovers_when_the_first_hydrate_comes_back_empty(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(1)
        return _FakeResp(_EMPTY_SPLITS if len(calls) == 1 else _ONE_GAME)

    monkeypatch.setattr(mlb._session, "get", fake_get)
    out = analytics._full_logs(12345, "hitting")
    assert len(calls) == 2, "must retry exactly once on an empty-but-resolved response"
    assert len(out) == 1 and out[0]["stat"] == {"runs": 1}


def test_logs_a_diagnostic_and_returns_empty_when_still_empty_after_retry(monkeypatch, capsys):
    monkeypatch.setattr(mlb._session, "get", lambda url, params=None, timeout=None:
                        _FakeResp(_EMPTY_SPLITS))
    out = analytics._full_logs(12345, "hitting")
    assert out == []
    err = capsys.readouterr().out
    assert "[RecentGames]" in err and "empty after" in err and "12345" in err


def test_unresolved_person_id_is_not_retried_and_stays_diagnosable(monkeypatch, capsys):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(1)
        return _FakeResp(_NO_PEOPLE)

    monkeypatch.setattr(mlb._session, "get", fake_get)
    out = analytics._full_logs(99999, "hitting")
    assert len(calls) == 1, "an unresolved personId is a different failure -- retrying won't fix it"
    assert out == []
    assert "people' array is empty" in capsys.readouterr().out


def test_a_healthy_first_response_is_used_directly_with_no_retry(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(1)
        return _FakeResp(_ONE_GAME)

    monkeypatch.setattr(mlb._session, "get", fake_get)
    out = analytics._full_logs(12345, "hitting")
    assert len(calls) == 1
    assert len(out) == 1
