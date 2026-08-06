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
    # _scored_board() calls these before reading fantasy_projections -- tests seed
    # ProjectionRepository directly (see _seed_league_and_players), so the provider sync
    # itself is a no-op here; this only stops a real network call to nflverse on every test.
    monkeypatch.setattr(routes_fantasy.projections_sync, "ensure_season_projections",
                        lambda *a, **k: None)
    monkeypatch.setattr(routes_fantasy.projections_sync, "ensure_week_projections",
                        lambda *a, **k: None)
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


def test_league_rosters_route_maps_owner_display_names(app_as, monkeypatch, db):
    rosters = [{"roster_id": 1, "owner_id": "u1"}, {"roster_id": 2, "owner_id": "u2"}]
    users = [{"user_id": "u1", "display_name": "Alice", "metadata": {"team_name": "Alice's Team"}},
            {"user_id": "u2", "username": "bobbyb"}]
    monkeypatch.setattr(sleeper_client, "get_league_rosters", lambda league_id: rosters)
    monkeypatch.setattr(sleeper_client, "get_league_users", lambda league_id: users)

    client = app_as(USER)
    resp = client.get("/api/fantasy/leagues/L1/rosters")
    assert resp.status_code == 200
    body = {r["roster_id"]: r for r in resp.json()["rosters"]}
    assert body[1]["owner_display_name"] == "Alice"
    assert body[1]["team_name"] == "Alice's Team"
    assert body[2]["owner_display_name"] == "bobbyb"   # falls back to username, no display_name


def test_waivers_excludes_every_rostered_player_and_ranks_by_need(app_as, monkeypatch, db):
    rb1, rb2, wr1 = _seed_league_and_players(db)
    league = LeagueRepository(db).list_for_user(USER["id"])[0]

    rosters = [
        {"roster_id": "1", "players": ["s1"]},   # mine: rb1 (Sleeper id s1)
        {"roster_id": "2", "players": ["s2"]},   # someone else's: rb2 (Sleeper id s2)
    ]
    monkeypatch.setattr(sleeper_client, "get_league_rosters", lambda league_id: rosters)

    client = app_as(USER)
    resp = client.get("/api/fantasy/waivers",
                      params={"league_id": league["id"], "roster_id": "1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["available_count"] == 1   # only wr1 (s3) is unrostered anywhere in the league
    assert body["recommendations"][0]["player_id"] == wr1["id"]
    # my only open starting need is WR (I already have my one RB slot filled by rb1)
    assert body["recommendations"][0]["need_weight"] == 1.15


def test_waivers_404_for_league_not_owned_by_caller(app_as, db):
    _seed_league_and_players(db)
    league = LeagueRepository(db).list_for_user(USER["id"])[0]
    other_client = app_as({"id": "someone-else", "role": "USER"})
    resp = other_client.get("/api/fantasy/waivers",
                            params={"league_id": league["id"], "roster_id": "1"})
    assert resp.status_code == 404


def test_trade_analyze_reports_vor_gain_direction(app_as, db):
    rb1, rb2, wr1 = _seed_league_and_players(db)
    league = LeagueRepository(db).list_for_user(USER["id"])[0]

    client = app_as(USER)
    resp = client.post("/api/fantasy/trade/analyze", json={
        "league_id": league["id"], "side_a": [rb1["id"]], "side_b": [rb2["id"], wr1["id"]]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["side_a"]["total_vor"] == 128.0    # 180 - replacement(52)
    assert body["side_b"]["total_vor"] == 0.0       # rb2 and wr1 are each their own replacement
    assert body["verdict"] == "Side A gains 128.0 VOR"


def test_trade_analyze_rejects_empty_side(app_as, db):
    _seed_league_and_players(db)
    league = LeagueRepository(db).list_for_user(USER["id"])[0]
    client = app_as(USER)
    resp = client.post("/api/fantasy/trade/analyze",
                       json={"league_id": league["id"], "side_a": [], "side_b": ["x"]})
    assert resp.status_code == 422


def _seed_lineup_league(db):
    LeagueRepository(db).upsert(
        USER["id"], "L2", "Lineup League", "2026", league_size=2,
        scoring_settings={"pass_yd": 0.04, "pass_td": 4.0, "rush_yd": 0.1, "rush_td": 6.0,
                          "rec": 1.0, "rec_yd": 0.1},
        roster_positions=["QB", "RB", "WR", "BN"])
    players = PlayerRepository(db)
    mapper = PlayerMappingRepository(db)
    proj = ProjectionRepository(db)

    qb1 = players.create(full_name="Star QB", position="QB", team="KC")
    rb1 = players.create(full_name="Star RB", position="RB", team="SF")
    wr1 = players.create(full_name="Star WR", position="WR", team="MIA")
    for p, sid in ((qb1, "s1"), (rb1, "s2"), (wr1, "s3")):
        mapper.map(p["id"], "sleeper", sid)

    proj.upsert(qb1["id"], "nflverse", 2026, 1, {"pass_yd": 300, "pass_td": 2}, "m")   # 20.0
    proj.upsert(rb1["id"], "nflverse", 2026, 1, {"rush_yd": 100, "rush_td": 1}, "m")   # 16.0
    proj.upsert(wr1["id"], "nflverse", 2026, 1, {"rec": 5, "rec_yd": 50}, "m")         # 10.0
    return qb1, rb1, wr1


def test_lineup_optimizes_and_reports_delta_against_current_starters(app_as, monkeypatch, db):
    qb1, rb1, wr1 = _seed_lineup_league(db)
    league = LeagueRepository(db).list_for_user(USER["id"])[0]

    roster = {"roster_id": "1", "players": ["s1", "s2", "s3"], "starters": ["s1", "s2"]}
    monkeypatch.setattr(sleeper_client, "get_league_rosters", lambda league_id: [roster])

    client = app_as(USER)
    resp = client.get("/api/fantasy/lineup",
                      params={"league_id": league["id"], "roster_id": "1", "week": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["week"] == 1
    assert body["projected_points"] == 46.0   # 20 + 16 + 10
    started_ids = {s["player"]["player_id"] for s in body["starters"] if s["player"]}
    assert started_ids == {qb1["id"], rb1["id"], wr1["id"]}
    assert body["delta"]["current_points"] == 36.0     # 20 + 16, wr1 wasn't started
    assert body["delta"]["points_gained"] == 10.0
    assert body["delta"]["start"] == [wr1["id"]]


def test_lineup_defaults_week_from_sleeper_nfl_state(app_as, monkeypatch, db):
    qb1, rb1, wr1 = _seed_lineup_league(db)
    league = LeagueRepository(db).list_for_user(USER["id"])[0]
    roster = {"roster_id": "1", "players": ["s1", "s2", "s3"], "starters": []}
    monkeypatch.setattr(sleeper_client, "get_league_rosters", lambda league_id: [roster])
    monkeypatch.setattr(sleeper_client, "get_nfl_state", lambda: {"season": "2026", "week": 1})

    client = app_as(USER)
    resp = client.get("/api/fantasy/lineup", params={"league_id": league["id"], "roster_id": "1"})
    assert resp.status_code == 200
    assert resp.json()["week"] == 1   # no ?week= given -- resolved from Sleeper's own clock


def test_lineup_404_for_unknown_roster(app_as, monkeypatch, db):
    _seed_lineup_league(db)
    league = LeagueRepository(db).list_for_user(USER["id"])[0]
    monkeypatch.setattr(sleeper_client, "get_league_rosters", lambda league_id: [])
    client = app_as(USER)
    resp = client.get("/api/fantasy/lineup",
                      params={"league_id": league["id"], "roster_id": "99", "week": 1})
    assert resp.status_code == 404


def test_unmatched_players_requires_admin(app_as, db):
    from fantasy.repositories import UnmatchedPlayerRepository
    UnmatchedPlayerRepository(db).log(source="nflverse_gsis", source_id="00-1", raw_name="X")

    user_client = app_as(USER)
    assert user_client.get("/api/fantasy/unmatched-players").status_code == 403

    admin_client = app_as(ADMIN)
    resp = admin_client.get("/api/fantasy/unmatched-players")
    assert resp.status_code == 200
    assert len(resp.json()["unmatched"]) == 1
