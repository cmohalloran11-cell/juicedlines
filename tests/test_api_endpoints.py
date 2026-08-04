"""
Basic integration tests for the primary PUBLIC API endpoints, via a real TestClient
against the live FastAPI app -- not just the underlying functions. Added 2026-08
(launch-readiness pass): these are the endpoints every user/client actually hits, and
before this they were only exercised indirectly through unit tests of the functions
behind them, never through the actual route (query-param handling, response shape,
status codes).

Uses USE_MOCK=1 (mock_lines(), instant, no network) so this stays fast and offline --
same pattern as tests/test_auth_and_user_api.py's fixture, kept separate here so a
failure in one file's fixture setup doesn't mask the other's tests.
"""
from __future__ import annotations

import importlib
import tempfile

import pytest


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("USE_MOCK", "1")
    monkeypatch.setenv("FALLBACK_TO_MOCK", "1")
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("SUPABASE_URL", "")
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "")
    monkeypatch.setenv("SENTRY_DSN", "")           # never let a test run talk to real Sentry

    import store
    store.reset_singleton()
    sdb = store.get_database(str(tmp_path / "juiced_test.db"))
    monkeypatch.setattr(store.database, "_singleton", sdb, raising=False)
    store.init_schema(sdb)

    import db as legacy_db
    monkeypatch.setattr(legacy_db, "DB_PATH", tmp_path / "history_test.db", raising=False)

    import auth
    importlib.reload(auth)
    import main
    importlib.reload(main)
    monkeypatch.setattr(main.store.database, "_singleton", sdb, raising=False)

    from fastapi.testclient import TestClient
    with TestClient(main.app) as c:
        yield c


# ── static pages ────────────────────────────────────────────────────────────────

def test_landing_and_app_pages_load(client):
    for path in ("/", "/app.html", "/dashboard", "/terms.html", "/privacy.html"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert "text/html" in r.headers["content-type"]


def test_health_endpoints(client):
    for path in ("/health", "/healthz"):
        r = client.get(path)
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


# ── core board API ───────────────────────────────────────────────────────────────

def test_status_endpoint(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body.get("mock_mode") is True


def test_lines_endpoint_returns_mock_lines(client):
    r = client.get("/api/lines")
    assert r.status_code == 200
    body = r.json()
    assert "lines" in body and "updated_at" in body and "errors" in body
    assert len(body["lines"]) > 0
    assert all("id" in l and "player" in l for l in body["lines"])


def test_lines_endpoint_sport_filter(client):
    r = client.get("/api/lines?sport=MLB")
    assert r.status_code == 200
    lines = r.json()["lines"]
    assert all(l.get("sport") == "MLB" for l in lines)


def test_projections_endpoint(client):
    r = client.get("/api/projections?sort=juice&limit=50")
    assert r.status_code == 200
    body = r.json()
    assert "projections" in body and "total" in body and "updated_at" in body
    for p in body["projections"]:
        assert "juiceScore" in p and "confidence" in p


def test_dashboard_endpoint(client):
    r = client.get("/api/dashboard?sport=all")
    assert r.status_code == 200
    body = r.json()
    assert "tiles" in body
    assert "coach_context" in body


def test_optimizer_endpoint(client):
    # This endpoint previously existed only inside build_static.py (the static-deploy JSON
    # export) -- the live server had no route for it, so the live Entry Optimizer page 404'd.
    r = client.get("/api/optimizer/today?sport=all")
    assert r.status_code == 200
    body = r.json()
    assert "todaysBest" in body and "topEntries" in body
    assert set(body["todaysBest"].keys()) == {"2_power", "3_power", "4_power", "5_power",
                                               "2_flex", "3_flex", "4_flex", "5_flex", "6_flex"}


def test_ai_coach_endpoint_shape(client):
    r = client.get("/api/ai/coach?sport=all")
    assert r.status_code == 200
    assert "ai" in r.json()


def test_injuries_and_books_endpoints(client):
    assert client.get("/api/injuries").status_code == 200
    assert client.get("/api/books").status_code == 200


def test_version_endpoint_shape(client):
    r = client.get("/api/version")
    assert r.status_code == 200
    body = r.json()
    for key in ("model_versions", "feature_version", "simulation_version", "schema_version"):
        assert key in body
    assert set(body["model_versions"].keys()) >= {"MLB", "WNBA", "Tennis"}


def test_auth_status_endpoint_never_leaks_secrets(client):
    r = client.get("/api/auth/status")
    assert r.status_code == 200
    body = r.json()
    # Only the PUBLIC anon key (publishable by design) may appear; no server secret keys.
    dumped = str(body)
    assert "SUPABASE_JWT_SECRET" not in dumped
    assert "service_role" not in dumped.lower()


# ── model health / data health (public transparency surface) ───────────────────

def test_model_health_endpoint(client):
    r = client.get("/api/model/health")
    assert r.status_code == 200
    body = r.json()
    assert "sports" in body and "versions" in body


def test_data_health_endpoint(client):
    r = client.get("/api/data/health")
    assert r.status_code == 200


def test_model_dashboard_summary_and_detail(client):
    summary = client.get("/api/model/dashboard")
    assert summary.status_code == 200
    assert "sports" in summary.json()

    detail = client.get("/api/model/dashboard?sport=MLB")
    assert detail.status_code == 200
    assert detail.json().get("sport") == "MLB"


# ── error handling ──────────────────────────────────────────────────────────────

def test_unknown_line_id_404s_cleanly(client):
    r = client.get("/api/simulation?id=does-not-exist")
    assert r.status_code == 404
    assert "error" in r.json() or "detail" in r.json()


def test_unknown_route_404s(client):
    assert client.get("/api/this-route-does-not-exist").status_code == 404
