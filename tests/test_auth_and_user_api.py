"""
Integration tests for server-side auth + per-user endpoints.

Uses a real Supabase-style HS256 JWT (signed with a test secret) against the live FastAPI
app via TestClient. Verifies: unauthenticated access is refused (fail-closed), a valid token
grants access and provisions the user, and one user can never see or mutate another's data.

The store is pointed at a temp SQLite file so nothing touches real data, and the app's
background snapshot loop is neutralized via USE_MOCK so tests stay fast and offline.
"""
from __future__ import annotations

import os
import time
import importlib
import tempfile

import jwt
import pytest

SECRET = "test-jwt-secret-0123456789-abcdefghijklmnop"  # ≥32 bytes for HS256


def _token(sub: str, email: str, exp_offset: int = 3600) -> str:
    payload = {"sub": sub, "email": email, "aud": "authenticated",
               "exp": int(time.time()) + exp_offset, "iat": int(time.time())}
    return jwt.encode(payload, SECRET, algorithm="HS256")


@pytest.fixture()
def client(monkeypatch, tmp_path):
    # Configure env BEFORE importing the app/auth so module-level config picks it up.
    monkeypatch.setenv("SUPABASE_JWT_SECRET", SECRET)
    monkeypatch.setenv("SUPABASE_URL", "")         # force the HS256 path (not JWKS) for tests
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("USE_MOCK", "1")            # no live pulls in the background loop
    monkeypatch.setenv("FALLBACK_TO_MOCK", "1")

    # Point the relational store at a temp SQLite file (and make get_database() return it).
    import store
    store.reset_singleton()
    sdb = store.get_database(str(tmp_path / "juiced_test.db"))
    monkeypatch.setattr(store.database, "_singleton", sdb, raising=False)
    store.init_schema(sdb)

    # Redirect the legacy ledger DB so the background loop never writes to the real one.
    import db as legacy_db
    monkeypatch.setattr(legacy_db, "DB_PATH", tmp_path / "history_test.db", raising=False)

    # (Re)import auth + app fresh so SUPABASE_JWT_SECRET is read from the patched env, and
    # pin the secret directly (belt-and-suspenders against import-order surprises).
    import auth
    importlib.reload(auth)
    monkeypatch.setattr(auth, "SUPABASE_JWT_SECRET", SECRET, raising=False)
    monkeypatch.setattr(auth, "SUPABASE_URL", "", raising=False)
    monkeypatch.setattr(auth, "CONFIGURED", True, raising=False)
    import main
    importlib.reload(main)
    monkeypatch.setattr(main.store.database, "_singleton", sdb, raising=False)

    from fastapi.testclient import TestClient
    with TestClient(main.app) as c:
        yield c


def test_unauthenticated_is_rejected(client):
    assert client.get("/api/me").status_code == 401
    assert client.get("/api/watchlists").status_code == 401
    assert client.post("/api/portfolio", json={"line_id": "L1"}).status_code == 401


def test_me_provisions_user(client):
    h = {"Authorization": f"Bearer {_token('uid-alice', 'alice@x.com')}"}
    r = client.get("/api/me", headers=h)
    assert r.status_code == 200
    u = r.json()["user"]
    assert u["id"] == "uid-alice" and u["email"] == "alice@x.com"
    assert u["role"] == "USER" and u["tier"] == "FREE"


def test_invalid_and_expired_tokens_rejected(client):
    assert client.get("/api/me", headers={"Authorization": "Bearer garbage"}).status_code == 401
    expired = _token("uid-x", "x@x.com", exp_offset=-10)
    assert client.get("/api/me", headers={"Authorization": f"Bearer {expired}"}).status_code == 401


def test_watchlist_flow_and_isolation(client):
    alice = {"Authorization": f"Bearer {_token('uid-alice', 'alice@x.com')}"}
    bob = {"Authorization": f"Bearer {_token('uid-bob', 'bob@x.com')}"}

    wid = client.post("/api/watchlists", json={"name": "MLB"}, headers=alice).json()["watchlist"]["id"]
    item = client.post(f"/api/watchlists/{wid}/items",
                       json={"line_id": "L1", "sport": "MLB", "player": "Judge"},
                       headers=alice)
    assert item.status_code == 200

    # Alice sees her list + item; Bob sees nothing.
    assert len(client.get("/api/watchlists", headers=alice).json()["watchlists"]) == 1
    assert client.get("/api/watchlists", headers=bob).json()["watchlists"] == []

    # Bob cannot add to Alice's list nor delete it.
    assert client.post(f"/api/watchlists/{wid}/items", json={"line_id": "L9"},
                       headers=bob).status_code == 404
    assert client.delete(f"/api/watchlists/{wid}", headers=bob).status_code == 404


def test_portfolio_flow_and_settlement(client):
    alice = {"Authorization": f"Bearer {_token('uid-alice', 'alice@x.com')}"}
    eid = client.post("/api/portfolio",
                      json={"line_id": "L1", "side": "over", "line_value": 1.5,
                            "stake": 2, "odds": -110, "sport": "MLB"},
                      headers=alice).json()["entry"]["id"]
    assert len(client.get("/api/portfolio", headers=alice).json()["entries"]) == 1
    ok = client.post(f"/api/portfolio/{eid}/settle", json={"status": "won", "result": 1.9},
                     headers=alice)
    assert ok.status_code == 200
    entries = client.get("/api/portfolio?status=won", headers=alice).json()["entries"]
    assert len(entries) == 1 and entries[0]["status"] == "won"


def test_version_endpoint_is_public(client):
    r = client.get("/api/version")
    assert r.status_code == 200
    assert "model_versions" in r.json() and "feature_version" in r.json()


def test_alerts_flow_and_validation(client):
    alice = {"Authorization": f"Bearer {_token('uid-alice', 'alice@x.com')}"}
    bob = {"Authorization": f"Bearer {_token('uid-bob', 'bob@x.com')}"}
    # unauth is refused
    assert client.get("/api/alerts").status_code == 401
    # bad metric → 422
    assert client.post("/api/alerts", json={"metric": "bogus", "threshold": 1},
                       headers=alice).status_code == 422
    # create a valid alert
    aid = client.post("/api/alerts",
                      json={"metric": "juice", "threshold": 70, "comparator": "gte",
                            "player": "Judge"},
                      headers=alice).json()["alert"]["id"]
    assert len(client.get("/api/alerts", headers=alice).json()["alerts"]) == 1
    # isolation: bob sees none and cannot toggle/delete alice's
    assert client.get("/api/alerts", headers=bob).json()["alerts"] == []
    assert client.post(f"/api/alerts/{aid}/toggle?active=false",
                       headers=bob).status_code == 404
    # owner can toggle + delete
    assert client.post(f"/api/alerts/{aid}/toggle?active=false",
                       headers=alice).status_code == 200
    assert client.delete(f"/api/alerts/{aid}", headers=alice).status_code == 200


def test_new_public_endpoints_respond(client):
    assert client.get("/api/model/health").status_code == 200
    assert client.get("/api/data/health").status_code == 200
    # admin calibration requires ADMIN — a normal user is forbidden
    alice = {"Authorization": f"Bearer {_token('uid-alice', 'alice@x.com')}"}
    assert client.get("/api/admin/calibration", headers=alice).status_code == 403
    assert client.get("/api/admin/calibration").status_code == 401


def test_simulation_endpoint_unknown_id_404(client):
    assert client.get("/api/simulation?id=nope").status_code == 404
    assert client.get("/api/model/health?sport=MLB").status_code == 200
