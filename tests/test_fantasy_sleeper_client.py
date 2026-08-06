"""
Tests for fantasy.sleeper_client — the server-side-only Sleeper API client. No real network
calls: requests.get is monkeypatched with a fake response, matching this repo's isolation
convention for external-API adapters (see tests/test_pullers_isolation.py).
"""
from __future__ import annotations

import pytest

from fantasy import sleeper_client


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_get_user_returns_none_on_404(monkeypatch):
    monkeypatch.setattr(sleeper_client.requests, "get",
                        lambda *a, **k: _FakeResponse(status_code=404))
    assert sleeper_client.get_user("nonexistent_user") is None


def test_get_user_returns_payload_on_200(monkeypatch):
    payload = {"user_id": "123", "username": "commish"}
    monkeypatch.setattr(sleeper_client.requests, "get",
                        lambda *a, **k: _FakeResponse(status_code=200, payload=payload))
    assert sleeper_client.get_user("commish") == payload


def test_get_league_raises_sleeper_error_on_non_200_non_404(monkeypatch):
    monkeypatch.setattr(sleeper_client.requests, "get",
                        lambda *a, **k: _FakeResponse(status_code=500))
    with pytest.raises(sleeper_client.SleeperError):
        sleeper_client.get_league("league123")


def test_get_draft_picks_defaults_to_empty_list_not_none(monkeypatch):
    monkeypatch.setattr(sleeper_client.requests, "get",
                        lambda *a, **k: _FakeResponse(status_code=404))
    assert sleeper_client.get_draft_picks("draft123") == []


def test_request_failure_wraps_in_sleeper_error(monkeypatch):
    import requests as real_requests

    def _raise(*a, **k):
        raise real_requests.ConnectionError("boom")

    monkeypatch.setattr(sleeper_client.requests, "get", _raise)
    with pytest.raises(sleeper_client.SleeperError):
        sleeper_client.get_user("anyone")


def test_fetch_all_players_uses_longer_timeout(monkeypatch):
    captured = {}

    def fake_get(url, headers=None, timeout=None):
        captured["timeout"] = timeout
        return _FakeResponse(status_code=200, payload={"1": {"full_name": "Test"}})

    monkeypatch.setattr(sleeper_client.requests, "get", fake_get)
    result = sleeper_client.fetch_all_players()
    assert result == {"1": {"full_name": "Test"}}
    assert captured["timeout"] == 60.0
