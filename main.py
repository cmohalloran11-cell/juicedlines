"""
main.py — FastAPI backend for the Sports Edge dashboard.

Endpoints:
  GET /              → serves index.html
  GET /api/lines     → current normalized lines (MLB + Tennis + WNBA)
  GET /api/lines/history?id=<line_id>  → SQLite movement history
  GET /api/status    → connector health
  POST /api/snapshot → manual snapshot trigger (dev)

Background: snapshots both feeds to SQLite every SNAPSHOT_INTERVAL seconds.
"""

from __future__ import annotations

import envload  # noqa: F401  — loads .env into os.environ BEFORE auth/ai_juice read config

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import db
import analytics
import provenance
import store
import ai_juice
import valuation
import model_health
import dataos
import dashboard as dashboard_mod
import optimizer as optimizer_mod
import weather as weather_mod
import books
import middleware
import fantasy
from fantasy import players_sync as fantasy_players_sync
from auth import require_role, optional_user, get_current_user
from routes_user import router as user_router
from routes_optimizer import router as optimizer_router
from routes_fantasy import router as fantasy_router
from pullers import fetch_prizepicks, fetch_underdog, mock_lines

FALLBACK_TO_MOCK = os.getenv("FALLBACK_TO_MOCK", "1").lower() not in ("0", "false", "no")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── error monitoring (optional, env-gated) ──────────────────────────────────────
# Fully opt-in: with no SENTRY_DSN set (the default -- nothing set locally or in most
# deploys), this block does nothing and the app never talks to Sentry. Set SENTRY_DSN on
# a deployment to get real-time alerting on unhandled exceptions instead of only finding
# out about a production issue from a user report or by reading logs after the fact.
_SENTRY_DSN = os.getenv("SENTRY_DSN", "")
if _SENTRY_DSN:
    try:
        import sentry_sdk
        sentry_sdk.init(
            dsn=_SENTRY_DSN,
            environment=os.getenv("SENTRY_ENVIRONMENT", "production"),
            release=None,   # provenance.code_sha() isn't available yet at this import point;
                            # Sentry still groups issues fine without an explicit release tag.
            # Off by default -- performance tracing samples request bodies/timings, which is
            # more than this app needs by default. Set SENTRY_TRACES_SAMPLE_RATE (0.0-1.0) to
            # opt in to tracing on a deployment that wants it.
            traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.0")),
            send_default_pii=False,   # never send request bodies/user IPs/cookies by default
        )
        log.info("Sentry error monitoring enabled (environment=%s)",
                os.getenv("SENTRY_ENVIRONMENT", "production"))
    except Exception as exc:
        # Never let a monitoring-tool problem take down the app it's supposed to be
        # watching -- log and continue exactly as if SENTRY_DSN had been unset.
        log.warning("Sentry init failed, continuing without it: %s", exc)


def _int_env(name: str, default: int) -> int:
    """int(os.getenv(...)) with a malformed value logging a clear warning and falling back
    to `default`, instead of a raw ValueError traceback crashing the process at import time
    (2026-08 production-readiness audit finding -- these are operational tuning knobs, not
    security-critical config, so fail-soft with visibility beats fail-hard here)."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("Env var %s=%r is not a valid integer; using default %d", name, raw, default)
        return default


SNAPSHOT_INTERVAL = _int_env("SNAPSHOT_INTERVAL", 180)  # seconds
USE_MOCK = os.getenv("USE_MOCK", "").lower() in ("1", "true", "yes")

# Rolling retention for the line-movement snapshot table. Unbounded, it reached
# 9.6M rows / 1.77GB in a month; movement charts only ever read recent rows. See
# db.prune_history. Set HISTORY_RETENTION_DAYS=0 to disable pruning.
HISTORY_RETENTION_DAYS = _int_env("HISTORY_RETENTION_DAYS", 14)
# Run DB maintenance (prune history + stale ungraded CLV) about once a day.
_MAINT_EVERY_CYCLES = max(1, 86_400 // max(SNAPSHOT_INTERVAL, 1))

# ─────────────────────────────────────────────────── in-memory line cache ────

_cache: dict[str, Any] = {
    "lines": [],
    "updated_at": None,
    "errors": {},
}

# id → line, kept in sync with _cache["lines"] so /api/analytics can look up
# any line (including ones filtered out of the current board view).
_line_by_id: dict[str, dict] = {}


def _reindex(lines: list[dict]) -> None:
    # build fully, then rebind atomically — a concurrent request reads either the
    # old or the new complete index, never a half-cleared one
    global _line_by_id
    _line_by_id = {l["id"]: l for l in lines}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def refresh_lines(sport: str = "all") -> dict[str, Any]:
    """Pull both sources, merge, cache, return result."""
    if USE_MOCK:
        lines = mock_lines()
        _cache["lines"] = lines
        _cache["updated_at"] = _now_iso()
        _cache["errors"] = {"mock": "Using mock data (USE_MOCK=true)"}
        _reindex(lines)
        return _cache.copy()

    errors: dict[str, str] = {}

    sf = sport if sport != "all" else None

    ud_lines, uerr = fetch_underdog(sport_filter=sf)
    if uerr:
        errors["underdog"] = uerr
        log.warning("Underdog error: %s", uerr)

    pp_lines, perr = fetch_prizepicks(sport_filter=sf)
    if perr:
        errors["prizepicks"] = perr
        log.warning("PrizePicks error: %s", perr)

    # Extra books (Sleeper live; DK Pick6 / ParlayPlay / Chalkboard bot-walled — see books.py).
    extra_lines, book_errs = books.fetch_extra_books(sport_filter=sf)
    for k, v in book_errs.items():
        errors[k] = v

    lines = ud_lines + pp_lines + extra_lines
    # If real sources came up completely empty, fall back to mock data so the
    # UI shows its full layout. Set FALLBACK_TO_MOCK=0 to disable.
    if not lines and FALLBACK_TO_MOCK:
        lines = mock_lines()
        errors["mock_fallback"] = "No live lines found — showing mock data. Set FALLBACK_TO_MOCK=0 to disable."
        log.info("No real lines found, using mock fallback.")

    # Attach headshots/logos/flags for the board (cached lookups, no extra HTTP).
    try:
        analytics.enrich_lines(lines)
    except Exception as exc:
        log.warning("enrich_lines failed: %s", exc)

    # Attach per-row model projections. This
    # also pre-warms the game-log cache, so the research drawer opens fast.
    try:
        for k, v in analytics.attach_projections(lines).items():
            errors[k] = v
            log.warning("%s attach failed: %s", k, v)
    except Exception as exc:
        log.warning("attach_projections failed: %s", exc)

    # juice_score's measured-per-stat-trust signal + the MLB shrinkage/losing-side flag it
    # carries (see analytics.attach_stat_trust) — build_static.py already calls this for the
    # static deploy; the live server never did, so this path was missing both the shrinkage
    # and the confirmed-losing-side suppression entirely.
    try:
        analytics.attach_stat_trust(lines)
    except Exception as exc:
        log.warning("attach_stat_trust failed: %s", exc)

    _cache["lines"] = lines
    _cache["updated_at"] = _now_iso()
    _cache["errors"] = errors
    _reindex(lines)
    return _cache.copy()


async def _snapshot_loop() -> None:
    loop = asyncio.get_event_loop()
    i = 0
    while True:
        try:
            log.info("Refreshing lines …")
            # refresh_lines() does blocking HTTP (Underdog/PP/statsapi/ESPN);
            # always run it in a thread so it never freezes the event loop.
            data = await loop.run_in_executor(None, refresh_lines)
            if data["lines"]:
                await loop.run_in_executor(
                    None, db.snapshot_lines, data["lines"], data["updated_at"])
                # CLV ledger: log line + model projection (open→close) per prop-day.
                n = await loop.run_in_executor(
                    None, db.log_clv, data["lines"], data["updated_at"])
                log.info("Snapshotted %d lines (CLV-logged %d)", len(data["lines"]), n)
            # Grade settled props ~every 30 min (10 cycles), off the event loop.
            # 2026-08 (production-readiness audit): grade_basketball/grade_tennis existed,
            # fully implemented ("best-effort; never raises into the build" by their own
            # docstrings) and mirror grade_pending's exact pattern, but were never actually
            # called from anywhere -- only MLB props were ever graded automatically. WNBA/
            # Tennis props got logged to the ledger every build and then sat ungraded
            # forever, which is why this session's Model Health work kept showing
            # insufficient_data for those two sports regardless of how much time passed.
            if i % 10 == 0:
                any_graded = False
                for grader in (analytics.grade_pending, analytics.grade_basketball,
                              analytics.grade_tennis):
                    res = await loop.run_in_executor(None, grader)
                    if res.get("graded") or res.get("voided"):
                        log.info("%s: graded %d, voided %d",
                                grader.__name__, res["graded"], res["voided"])
                        any_graded = True
                if any_graded:
                    # model_health.py caches its aggregates for _TTL seconds (2026-08) --
                    # a fresh grading pass should be reflected immediately, not after a
                    # stale window expires.
                    model_health.clear_cache()
            # DB maintenance ~once a day: bound the snapshot table + drop stale
            # ungraded ledger rows. Cheap DELETEs — no VACUUM in the hot loop.
            if HISTORY_RETENTION_DAYS > 0 and i % _MAINT_EVERY_CYCLES == 0:
                hn = await loop.run_in_executor(
                    None, db.prune_history, HISTORY_RETENTION_DAYS)
                cn = await loop.run_in_executor(None, db.prune_clv)
                log.info("Maintenance: pruned %d history rows, %d stale CLV rows", hn, cn)
                # Model registry (Vol V Ch 178): snapshot each sport's measured accuracy so
                # performance is tracked over time and drift is detectable.
                try:
                    import backtest
                    for sp in ("MLB", "WNBA", "Tennis"):
                        await loop.run_in_executor(None, backtest.record_run, sp, "baseline", None)
                except Exception as exc:
                    log.warning("model registry snapshot failed: %s", exc)
        except Exception as exc:
            log.exception("Snapshot failed: %s", exc)
        i += 1
        await asyncio.sleep(SNAPSHOT_INTERVAL)


async def _fantasy_sync_loop() -> None:
    """Checks about once an hour whether Sleeper's /v1/players/nfl dump needs re-fetching
    (fantasy_players_sync_log-gated, so the once-a-day request-cap survives restarts) and, if
    so, runs it off the event loop. Sleeper's ~5MB player dump must never be fetched from a
    request path -- see fantasy.sleeper_client's module docstring.

    Respects USE_MOCK like refresh_lines() -- otherwise every test/CI process that boots the
    full app (TestClient(main.app), same pattern as test_api_endpoints.py) would fire a real,
    uncontrolled Sleeper request on startup."""
    if USE_MOCK:
        log.info("USE_MOCK set -- fantasy players sync loop disabled.")
        return
    loop = asyncio.get_event_loop()
    while True:
        try:
            db_ = store.get_database()
            if await loop.run_in_executor(None, fantasy_players_sync.should_sync, db_):
                log.info("Fantasy players sync due — fetching Sleeper /v1/players/nfl …")
                result = await loop.run_in_executor(None, fantasy_players_sync.sync_players, db_)
                log.info("Fantasy players sync complete: %s", result)
        except Exception as exc:
            log.exception("Fantasy players sync failed: %s", exc)
        await asyncio.sleep(3600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    # Juiced 2.0 relational store (users / watchlists / portfolio) — SQLite locally,
    # Postgres in production via DATABASE_URL. Idempotent.
    try:
        store.init_schema(store.get_database())
        fantasy.init_schema(store.get_database())
    except Exception as exc:
        log.warning("store schema init failed: %s", exc)
    # Don't block startup on the (slow) warm-up — the server accepts connections
    # immediately and the loop's first iteration fills the cache in a thread. The
    # board shows "warming up" for a few seconds instead of being unreachable.
    task = asyncio.create_task(_snapshot_loop())
    fantasy_task = asyncio.create_task(_fantasy_sync_loop())
    yield
    task.cancel()
    fantasy_task.cancel()


app = FastAPI(title="Sports Edge", lifespan=lifespan)

# Allow the page to reach the API even when index.html is opened directly as a
# file (Origin: null) — a read-only local line board, so wildcard CORS is fine.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

app.include_router(user_router)
app.include_router(optimizer_router)
app.include_router(fantasy_router)

# Backend hardening: request-id + timing logs, error envelope, optional rate limit (Vol III).
middleware.install(app)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ──────────────────────────────────────────────────────────── routes ──────────

# HTML is served with no-cache so edits show on refresh (avoids stale-cache confusion).
_HTML_HEADERS = {"Cache-Control": "no-cache, no-store, must-revalidate"}


@app.get("/health")
@app.get("/healthz")
def health_check():
    """Pure process-liveness check for a load balancer / uptime monitor -- no disk read, no
    DB query, no dependency on upstream feeds being up. Deliberately distinct from
    /api/status (reports on cached line data) and /api/data/health (reports on data
    freshness/quality): those correctly go unhealthy when PrizePicks/Underdog/statsapi are
    down even though this process is completely fine, which is the wrong signal for
    "should the load balancer restart/stop routing to this instance." Added 2026-08 --
    render.yaml previously pointed healthCheckPath at "/", which does a FileResponse disk
    read of static/index.html on every check instead of a true liveness ping."""
    return {"status": "ok"}


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html", headers=_HTML_HEADERS)


@app.get("/app.html")
def app_page():
    # The landing page (index.html) links to ./app.html for "Launch app". In the static
    # deploy both files are siblings; on the FastAPI server we map it explicitly so the
    # launch buttons resolve instead of 404ing.
    return FileResponse(STATIC_DIR / "app.html", headers=_HTML_HEADERS)


@app.get("/terms.html")
def terms_page():
    return FileResponse(STATIC_DIR / "terms.html", headers=_HTML_HEADERS)


@app.get("/privacy.html")
def privacy_page():
    return FileResponse(STATIC_DIR / "privacy.html", headers=_HTML_HEADERS)


@app.get("/dashboard")
def dashboard_page():
    """The Juiced 2.0 terminal dashboard (Vol VI Pt3)."""
    return FileResponse(STATIC_DIR / "dashboard.html", headers=_HTML_HEADERS)


@app.get("/api/dashboard")
def api_dashboard(sport: str = Query("all", description="All | MLB | WNBA | Tennis")):
    """One-shot aggregation for the dashboard home: tiles, top Juice drops, movers,
    upcoming games, model-confidence factors, and data/model health."""
    return dashboard_mod.build(
        _cache.get("lines") or [], _cache.get("updated_at"),
        _cache.get("errors"), sport=sport)


@app.get("/api/projections")
def api_projections(sport: str = Query("all"), stat: str = Query(""),
                    sort: str = Query("juice", description="juice|edge|confidence|projection"),
                    limit: int = Query(1000, ge=1, le=15000),
                    boosted: bool = Query(False, description="demon/goblin lane instead of standard/boosted")):
    """Full enriched projected-props feed (Projections / Props Center / Top Movers)."""
    odds_types = ("demon", "goblin") if boosted else ("standard", "boosted")
    rows = dashboard_mod.projections(
        _cache.get("lines") or [], sport=sport, stat=stat or None, sort=sort, limit=limit,
        odds_types=odds_types)
    return {"projections": rows, "total": len(rows), "updated_at": _cache.get("updated_at")}


@app.get("/api/injuries")
def api_injuries():
    """Scratched / returning-from-IL players flagged on today's board."""
    return {"injuries": dashboard_mod.injuries(_cache.get("lines") or [])}


@app.get("/api/books")
def api_books():
    """Sportsbook connector status (which books are live vs bot-walled)."""
    return {"books": books.status()}


@app.get("/api/weather")
async def api_weather():
    """Live ballpark conditions (Xweather) for today's outdoor MLB games."""
    lines = _cache.get("lines") or []
    return await asyncio.get_event_loop().run_in_executor(None, weather_mod.slate, lines)


@app.get("/api/auth/status")
def api_auth_status():
    """What identity + AI features are configured, so the UI can adapt (no secrets leaked)."""
    import auth as _auth
    return {
        "auth_configured": _auth.CONFIGURED,
        "auth_method": ("jwks" if _auth.SUPABASE_URL else
                        "hs256" if _auth.SUPABASE_JWT_SECRET else None),
        "dev_mode": _auth.AUTH_DEV and not _auth.CONFIGURED,
        # Public Supabase client config (anon key is publishable by design) so the browser
        # can init supabase-js and run the real login flow.
        "supabase_url": _auth.SUPABASE_URL or None,
        "supabase_anon_key": os.getenv("SUPABASE_ANON_KEY") or None,
        "ai": ai_juice.available(),
        "ai_provider": ai_juice.PROVIDER,
        "ai_rpm_limit": ai_juice.status().get("rpm_limit"),
    }


@app.get("/api/lines")
def api_lines(
    sport: str = Query("all", description="MLB | Tennis | WNBA | all"),
    source: str = Query("all", description="underdog | prizepicks | all"),
    stat_type: str = Query("", description="filter by stat type substring"),
    player: str = Query("", description="filter by player name substring"),
):
    lines: list[dict] = _cache.get("lines") or []

    if sport != "all":
        lines = [l for l in lines if l.get("sport", "").lower() == sport.lower()]
    if source != "all":
        lines = [l for l in lines if l.get("source", "") == source]
    if stat_type:
        st = stat_type.lower()
        lines = [l for l in lines if st in (l.get("stat_type") or "").lower()]
    if player:
        pl = player.lower()
        lines = [l for l in lines if pl in (l.get("player") or "").lower()]

    return {
        "lines": lines,
        "total": len(lines),
        "updated_at": _cache.get("updated_at"),
        "errors": _cache.get("errors", {}),
    }


@app.get("/api/lines/history")
def api_history(
    id: str = Query(..., description="line_id to fetch history for"),
    limit: int = Query(200, ge=1, le=1000),
):
    rows = db.get_history(id, limit=limit)
    return {"line_id": id, "history": rows, "count": len(rows)}


@app.get("/api/analytics")
async def api_analytics(id: str = Query(..., description="line_id to analyze")):
    """Per-player analytics for the research drawer (recent form, vs-team, hit-rate)."""
    line = _line_by_id.get(id)
    if not line:
        raise HTTPException(status_code=404, detail=f"Unknown line id: {id}")
    # analytics does blocking HTTP to statsapi — run off the event loop
    result = await asyncio.get_event_loop().run_in_executor(None, analytics.analyze, line)
    return {"line_id": id, "analytics": result}


@app.get("/api/version")
def api_version():
    """Current model/feature/simulation/calibration versions + code sha (transparency)."""
    return provenance.versions()


@app.get("/api/ai/status")
def api_ai_status():
    """Whether AI Juice is configured (graceful-degradation probe for the UI)."""
    return ai_juice.status()


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _ai_quota_check(user: dict) -> Optional[dict]:
    """None (and consumes one call) if `user` has AI Juice quota left today; else the
    rejection payload, same {available, reason} shape ai_juice's own failures use. The
    client-shown limit (ai_juice.daily_limit) is a business decision, not a UI hint — this
    is the actual enforcement point, checked server-side against store.AiUsageRepository."""
    usage = store.AiUsageRepository(store.get_database())
    today = _today_utc()
    limit = ai_juice.daily_limit(user.get("tier"))
    used = usage.get_count(user["id"], today)
    if used >= limit:
        return {"available": False,
                "reason": f"Daily AI Juice limit reached ({used}/{limit}). Resets at midnight UTC."}
    usage.increment(user["id"], today)
    return None


@app.get("/api/ai/usage")
def api_ai_usage(user: Optional[dict] = Depends(optional_user)):
    """Today's real AI Juice usage for the signed-in user — what the header indicator
    displays. Never increments (that only happens on an actual explain/coach call)."""
    if not user:
        return {"signed_in": False}
    usage = store.AiUsageRepository(store.get_database())
    limit = ai_juice.daily_limit(user.get("tier"))
    used = usage.get_count(user["id"], _today_utc())
    return {"signed_in": True, "used": used, "limit": limit, "tier": user.get("tier")}


@app.get("/api/simulation")
def api_simulation(id: str = Query(..., description="line_id to value")):
    """SimulationOS: the projection's distribution object + EV / Kelly / Juice / confidence."""
    line = _line_by_id.get(id)
    if not line:
        raise HTTPException(status_code=404, detail=f"Unknown line id: {id}")
    return {
        "line_id": id,
        "simulation": valuation.simulation_object(line),
        "valuation": valuation.valuation(line),
    }


@app.get("/api/model/health")
def api_model_health(sport: str = Query("", description="MLB | WNBA | Tennis | NFL | empty=all"),
                     proj_kind: str = Query(
                         "", description="empty=no filter | e.g. nfl_regular | nfl_preseason "
                                         "(only meaningful combined with a specific sport)")):
    """Public transparency: measured accuracy, CLV, and calibration from the graded ledger."""
    return model_health.health(sport or None, proj_kind or None)


@app.get("/api/backtest")
async def api_backtest(sport: str = Query("MLB"),
                       replay: int = Query(0, ge=0, le=800,
                                           description="if >0, run a leakage-free replay of N historical props"),
                       proj_kind: str = Query(
                           "", description="empty=no filter | e.g. nfl_regular | nfl_preseason"),
                       user: dict = Depends(optional_user)):
    """Model measurement ruler (Vol V Pt4): current accuracy from the ledger, and optionally
    a leakage-free replay of the live model on N past prop-days.

    `replay` re-runs the real projection engine up to 800 times per call (one live
    Monte-Carlo simulation per historical prop-day) -- a materially more expensive,
    triggerable action than just reading the ledger, and unlike current_accuracy it isn't
    cached (see model_health.py). Viewing current_accuracy (replay=0, the default) stays
    fully public, matching the "transparency" principle every other model-health endpoint
    follows; actually triggering a replay requires being logged in (any account, not
    ADMIN-only) so it can't be hit anonymously in a loop."""
    if replay and not user:
        raise HTTPException(status_code=401,
                            detail="Sign in to run a backtest replay (viewing current "
                                   "accuracy without replay doesn't require it).")
    import backtest
    loop = asyncio.get_event_loop()
    pk = proj_kind or None
    out = {"current": await loop.run_in_executor(
        None, lambda: backtest.current_accuracy(sport, proj_kind=pk))}
    if replay:
        out["replay"] = await loop.run_in_executor(None, backtest.replay, sport, replay)
    return out


@app.get("/api/model/drift")
def api_model_drift(sport: str = Query("MLB"),
                    proj_kind: str = Query(
                        "", description="empty=no filter | e.g. nfl_regular | nfl_preseason")):
    """Drift detection (Vol V Ch 182): recent vs prior accuracy — is the live model degrading?"""
    import backtest
    return backtest.drift(sport, proj_kind=proj_kind or None)


@app.get("/api/model/registry")
def api_model_registry(sport: str = Query("", description="empty = all sports")):
    """Model registry (Vol V Ch 178): recorded backtest runs over time."""
    import backtest
    return {"runs": backtest.registry(sport or None)}


@app.get("/api/model/dashboard")
async def api_model_dashboard(sport: str = Query("", description="empty = all-sport summary"),
                              proj_kind: str = Query(
                                  "", description="empty=no filter | e.g. nfl_regular | "
                                                  "nfl_preseason (only used with a specific sport)")):
    """Model Health dashboard bundle: the self-monitoring surface. No sport = a
    lightweight cross-sport summary (top-line accuracy + drift status per sport, for the
    landing view); a sport = the full deep-dive (accuracy, reliability diagram, drift,
    per-stat drift, archetype breakdown, version history, changelog, daily-snapshot
    trend). One call per view, matching the existing /api/dashboard pattern."""
    loop = asyncio.get_event_loop()
    if not sport:
        return await loop.run_in_executor(None, model_health.dashboard_summary)
    pk = proj_kind or None
    return await loop.run_in_executor(None, lambda: model_health.dashboard_detail(sport, pk))


@app.get("/api/model/reliability")
def api_model_reliability(sport: str = Query("MLB"),
                          stat: str = Query("", description="empty = whole sport"),
                          proj_kind: str = Query(
                              "", description="empty=no filter | e.g. nfl_regular | nfl_preseason")):
    """Reliability diagram data for one sport (optionally one stat): predicted P(over)
    vs realized frequency per bucket — the direct evidence for "are the probabilities
    actually calibrated."""
    import backtest
    return backtest.reliability_diagram(sport, stat or None, proj_kind=proj_kind or None)


@app.get("/api/model/drift/by-stat")
def api_model_drift_by_stat(sport: str = Query("MLB"),
                            proj_kind: str = Query(
                                "", description="empty=no filter | e.g. nfl_regular | nfl_preseason")):
    """Per-stat drift: which specific statistics are degrading, not just the sport
    overall (a sport-wide 'stable' can hide one stat quietly worsening)."""
    import backtest
    return backtest.drift_by_stat(sport, proj_kind=proj_kind or None)


@app.get("/api/model/archetypes")
def api_model_archetypes(sport: str = Query("MLB"),
                         proj_kind: str = Query(
                             "", description="empty=no filter | e.g. nfl_regular | nfl_preseason")):
    """Accuracy by player sample depth (model_n at grading time) — the measured answer
    to "which player archetypes perform worst": thin-sample rookies/call-ups/injury
    returns vs deep-sample established regulars."""
    import backtest
    return backtest.by_sample_depth(sport, proj_kind=proj_kind or None)


@app.get("/api/model/changelog")
def api_model_changelog(sport: str = Query("", description="empty = all sports")):
    """What changed in each model version, in plain language — pair with
    /api/model/dashboard's accuracy_by_model_version to see whether a change helped."""
    return provenance.changelog(sport or None)


@app.get("/api/data/health")
def api_data_health():
    """DataOS: freshness + coverage + a composite quality score for the current snapshot."""
    return dataos.health(
        _cache.get("lines") or [], _cache.get("updated_at"), _cache.get("errors"))


@app.get("/api/admin/calibration")
def api_admin_calibration(sport: str = Query("MLB"),
                          proj_kind: str = Query(
                              "", description="empty=no filter | e.g. nfl_regular | nfl_preseason"),
                          _admin: dict = Depends(require_role("ADMIN", "SUPER_ADMIN"))):
    """AdminOS: full per-stat calibration stack. Requires an ADMIN role."""
    return model_health.calibration_detail(sport, proj_kind=proj_kind or None)


@app.get("/api/ai/explain")
async def api_ai_explain(id: str = Query(..., description="line_id to explain"),
                         user: dict = Depends(get_current_user)):
    """A grounded, plain-language explanation of one projection. AI Juice may cite ONLY
    that projection's own numbers (no fabrication); degrades cleanly when unconfigured.
    Requires sign-in and consumes one of the user's daily AI Juice calls (see
    ai_juice.AI_DAILY_LIMITS) — the shared Gemini key has no per-user isolation otherwise."""
    line = _line_by_id.get(id)
    if not line:
        raise HTTPException(status_code=404, detail=f"Unknown line id: {id}")
    quota_err = _ai_quota_check(user)
    if quota_err:
        return {"line_id": id, "ai": quota_err}
    # Off the event loop — the AI call is blocking network I/O.
    result = await asyncio.get_event_loop().run_in_executor(None, ai_juice.explain, line)
    return {"line_id": id, "ai": result}


@app.get("/api/ai/coach")
async def api_ai_coach(sport: str = Query("all"), user: dict = Depends(get_current_user)):
    """Grounded slate-level strategy recommendation (AI Coach panel) — built from the same
    real aggregate numbers already shown on the dashboard, never invented. Requires sign-in
    and consumes one of the user's daily AI Juice calls — see api_ai_explain."""
    quota_err = _ai_quota_check(user)
    if quota_err:
        return {"ai": quota_err}
    lines = _cache.get("lines") or []
    d = dashboard_mod.build(lines, _cache.get("updated_at"), _cache.get("errors"), sport=sport)
    ctx = dict(d.get("coach_context") or {})
    top = optimizer_mod.top_overall(optimizer_mod.build_pool(lines), limit=1, sport=sport)
    if top:
        best = top[0]
        ctx["best_entry_format"] = best["format"]
        ctx["best_entry_quality_score"] = best["qualityScore"]
        ctx["best_entry_expected_roi_pct"] = round(best["simulation"]["expectedROI"] * 100, 1)
        ctx["best_entry_chance_of_profit_pct"] = round(best["simulation"]["chanceOfProfit"] * 100, 1)
    result = await asyncio.get_event_loop().run_in_executor(None, ai_juice.coach, ctx)
    return {"ai": result}


@app.get("/api/status")
def api_status():
    lines = _cache.get("lines") or []
    by_source: dict[str, int] = {}
    by_sport: dict[str, int] = {}
    for l in lines:
        s = l.get("source", "unknown")
        by_source[s] = by_source.get(s, 0) + 1
        sp = l.get("sport", "other")
        by_sport[sp] = by_sport.get(sp, 0) + 1
    return {
        "updated_at": _cache.get("updated_at"),
        "total_lines": len(lines),
        "by_source": by_source,
        "by_sport": by_sport,
        "errors": _cache.get("errors", {}),
        "mock_mode": USE_MOCK,
        "snapshot_interval_s": SNAPSHOT_INTERVAL,
    }


@app.get("/api/scorecard")
def api_scorecard(sport: str = Query("", description="MLB | Tennis | WNBA | NFL | empty = all"),
                  proj_kind: str = Query(
                      "", description="empty=no filter | e.g. nfl_regular | nfl_preseason")):
    """Running model-vs-market scorecard: hit-rate vs the close, plays, and CLV.
    Builds up over time as the CLV ledger logs lines and grades outcomes."""
    return db.scorecard(sport or None, proj_kind=proj_kind or None)


@app.post("/api/snapshot")
async def api_snapshot(sport: str = "all"):
    """Manually trigger a refresh + snapshot. Useful during development."""
    data = await asyncio.get_event_loop().run_in_executor(None, lambda: refresh_lines(sport))
    if data["lines"]:
        db.snapshot_lines(data["lines"], data["updated_at"])
    return {
        "snapshotted": len(data["lines"]),
        "updated_at": data["updated_at"],
        "errors": data["errors"],
    }


if __name__ == "__main__":
    import os
    import uvicorn
    # Hosts (Render/Railway/Fly/Docker) inject the port via $PORT. Reload only in dev
    # (set DEV=1) — it must be off in production so the background refresh loop is stable.
    uvicorn.run("main:app", host="0.0.0.0",
                port=int(os.environ.get("PORT", "8001")),
                reload=bool(os.environ.get("DEV")))
