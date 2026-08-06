"""
Integration tests for routes_fantasy.py, mounted standalone (not the whole main.py app) so
these stay fast and don't depend on the live-board snapshot loop. Sleeper is always
monkeypatched -- no real network. Auth uses FastAPI's dependency_overrides, same mechanism
FastAPI itself recommends for testing Depends()-gated routes.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import fantasy
import store
from store.database import SQLiteDatabase
from fantasy.repositories import LeagueRepository, PlayerRepository, PlayerMappingRepository, \
    ProjectionRepository
from fantasy import sleeper_client
import routes_fantasy
from auth import get_current_user

USER = {"id": "user-1", "email": "coach@example.com", "role": "USER", "tier": "FREE"}
ADMIN = {"id": "admin-1", "email": "admin@example.com", "role": "ADMIN", "tier": "ELITE"}


@pytest.fixture()
def db(tmp_path, monkeypatch):
    d = SQLiteDatabase(tmp_path / "fantasy_test.db")
    fantasy.init_schema(d)
    store.reset_singleton()
    monkeypatch.setattr(store.database, "_singleton", d, raising=False)
    routes_fantasy._cache.clear()  # TTL cache is module-level; don't leak between tests
    return d


@pytest.fixture()
def app_as(db):
    def _build(user):
        app = FastAPI()
        app.include_router(routes_fantasy.router)
        app.dependency_overrides[get_current_user] = lambda: user
        return TestClient(app)
    return _build


def test_import_league_requires_auth(db):
    app = FastAPI()
    app.include_router(routes_fantasy.router)
    client = TestClient(app)
    resp = client.post("/api/fantasy/leagues/import", json={"sleeper_league_id": "L1"})
    assert resp.status_code == 401


def test_sleeper_user_lookup_404_when_not_found(app_as, monkeypatch):
    monkeypatch.setattr(sleeper_client, "get_user", lambda username: None)
    client = app_as(USER)
    resp = client.get("/api/fantasy/sleeper/user/nobody")
    assert resp.status_code == 404


def test_league_detail_is_cached_within_ttl(app_as, monkeypatch):
    calls = {"n": 0}

    def fake_get_league(league_id):
        calls["n"] += 1
        return {"name": "Test League", "scoring_settings": {}, "roster_positions": []}

    monkeypatch.setattr(sleeper_client, "get_league", fake_get_league)
    client = app_as(USER)
    client.get("/api/fantasy/leagues/L1")
    client.get("/api/fantasy/leagues/L1")
    assert calls["n"] == 1  # second call served from the 5-min TTL cache, no second Sleeper hit


def test_import_league_persists_real_scoring_and_roster_shape(app_as, monkeypatch, db):
    league_payload = {
        "name": "Dynasty Warriors", "season": "2026", "total_rosters": 10,
        "scoring_settings": {"rec": 0.5, "pass_td": 4.0, "rush_yd": 0.1},
        "roster_positions": ["QB", "RB", "RB", "WR", "FLEX", "BN"],
    }
    monkeypatch.setattr(sleeper_client, "get_league", lambda league_id: league_payload)
    client = app_as(USER)
    resp = client.post("/api/fantasy/leagues/import", json={"sleeper_league_id": "L1"})
    assert resp.status_code == 200
    body = resp.json()["league"]
    assert body["scoring_settings"] == league_payload["scoring_settings"]  # not hardcoded PPR
    assert body["roster_positions"] == league_payload["roster_positions"]
    assert body["league_size"] == 10

    saved = LeagueRepository(db).list_for_user(USER["id"])
    assert len(saved) == 1


def test_league_drafts_returns_sleeper_drafts_list(app_as, monkeypatch):
    drafts = [{"draft_id": "D1", "status": "drafting"}]
    monkeypatch.setattr(sleeper_client, "get_league_drafts", lambda league_id: drafts)
    client = app_as(USER)
    resp = client.get("/api/fantasy/leagues/L1/drafts")
    assert resp.status_code == 200
    assert resp.json()["drafts"] == drafts


def test_import_league_404s_on_unknown_sleeper_league(app_as, monkeypatch):
    monkeypatch.setattr(sleeper_client, "get_league", lambda league_id: None)
    client = app_as(USER)
    resp = client.post("/api/fantasy/leagues/import", json={"sleeper_league_id": "ghost"})
    assert resp.status_code == 404


def _seed_league_and_players(db):
    LeagueRepository(db).upsert(
        USER["id"], "L1", "Test League", "2026", league_size=2,
        scoring_settings={"rec": 1.0, "rec_yd": 0.1, "rush_yd": 0.1, "rush_td": 6.0},
        roster_positions=["RB", "WR", "BN"])

    players = PlayerRepository(db)
    mapper = PlayerMappingRepository(db)
    proj = ProjectionRepository(db)

    rb1 = players.create(full_name="Star RB", position="RB", team="SF")
    rb2 = players.create(full_name="Backup RB", position="RB", team="NYJ")
    wr1 = players.create(full_name="Star WR", position="WR", team="MIA")
    for p, sleeper_id in ((rb1, "s1"), (rb2, "s2"), (wr1, "s3")):
        mapper.map(p["id"], "sleeper", sleeper_id)

    proj.upsert(rb1["id"], "nflverse", 2026, None, {"rush_yd": 1200, "rush_td": 10}, "m")
    proj.upsert(rb2["id"], "nflverse", 2026, None, {"rush_yd": 400, "rush_td": 2}, "m")
    proj.upsert(wr1["id"], "nflverse", 2026, None, {"rec": 90, "rec_yd": 1100}, "m")
    return rb1, rb2, wr1


def test_draft_board_scores_players_under_league_settings(app_as, db):
    rb1, rb2, wr1 = _seed_league_and_players(db)
    league = LeagueRepository(db).list_for_user(USER["id"])[0]

    client = app_as(USER)
    resp = client.get("/api/fantasy/draft-board", params={"league_id": league["id"]})
    assert resp.status_code == 200
    board = {p["player_id"]: p for p in resp.json()["board"]}

    # rush_yd*0.1 + rush_td*6 = 1200*0.1 + 10*6 = 180.0
    assert board[rb1["id"]]["fantasy_points"] == 180.0
    # rec*1 + rec_yd*0.1 = 90 + 110 = 200.0
    assert board[wr1["id"]]["fantasy_points"] == 200.0
    assert board[rb1["id"]]["tier"] >= 1


def test_draft_board_404_for_league_not_owned_by_caller(app_as, db):
    _seed_league_and_players(db)
    league = LeagueRepository(db).list_for_user(USER["id"])[0]
    other_client = app_as({"id": "someone-else", "role": "USER"})
    resp = other_client.get("/api/fantasy/draft-board", params={"league_id": league["id"]})
    assert resp.status_code == 404


def test_live_draft_removes_drafted_players_and_flags_runs(app_as, monkeypatch, db):
    rb1, rb2, wr1 = _seed_league_and_players(db)
    league = LeagueRepository(db).list_for_user(USER["id"])[0]

    picks = [
        {"player_id": rb1["id"], "roster_id": "1", "metadata": {"position": "RB"}},
        {"player_id": "unrelated-1", "roster_id": "2", "metadata": {"position": "RB"}},
        {"player_id": "unrelated-2", "roster_id": "1", "metadata": {"position": "RB"}},
    ]
    monkeypatch.setattr(sleeper_client, "get_draft_picks", lambda draft_id: picks)

    client = app_as(USER)
    resp = client.get(f"/api/fantasy/draft/D1/live",
                      params={"league_id": league["id"], "roster_id": "1"})
    assert resp.status_code == 200
    body = resp.json()
    available_ids = {p["player_id"] for p in body["available"]}
    assert rb1["id"] not in available_ids           # drafted player removed from the board
    assert rb2["id"] in available_ids and wr1["id"] in available_ids
    assert body["positional_runs"].get("RB") == 3    # 3 RB picks in the recent window


def test_unmatched_players_requires_admin(app_as, db):
    from fantasy.repositories import UnmatchedPlayerRepository
    UnmatchedPlayerRepository(db).log(source="nflverse_gsis", source_id="00-1", raw_name="X")

    user_client = app_as(USER)
    assert user_client.get("/api/fantasy/unmatched-players").status_code == 403

    admin_client = app_as(ADMIN)
    resp = admin_client.get("/api/fantasy/unmatched-players")
    assert resp.status_code == 200
    assert len(resp.json()["unmatched"]) == 1
