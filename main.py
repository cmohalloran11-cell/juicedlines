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

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import db
import analytics
from pullers import fetch_prizepicks, fetch_underdog, mock_lines

FALLBACK_TO_MOCK = os.getenv("FALLBACK_TO_MOCK", "1").lower() not in ("0", "false", "no")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SNAPSHOT_INTERVAL = int(os.getenv("SNAPSHOT_INTERVAL", "180"))  # seconds
USE_MOCK = os.getenv("USE_MOCK", "").lower() in ("1", "true", "yes")

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

    lines = ud_lines + pp_lines
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
        analytics.attach_projections(lines)
    except Exception as exc:
        log.warning("attach_projections failed: %s", exc)

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
            if i % 10 == 0:
                res = await loop.run_in_executor(None, analytics.grade_pending)
                if res.get("graded") or res.get("voided"):
                    log.info("CLV graded %d, voided %d", res["graded"], res["voided"])
        except Exception as exc:
            log.exception("Snapshot failed: %s", exc)
        i += 1
        await asyncio.sleep(SNAPSHOT_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    # Don't block startup on the (slow) warm-up — the server accepts connections
    # immediately and the loop's first iteration fills the cache in a thread. The
    # board shows "warming up" for a few seconds instead of being unreachable.
    task = asyncio.create_task(_snapshot_loop())
    yield
    task.cancel()


app = FastAPI(title="Sports Edge", lifespan=lifespan)

# Allow the page to reach the API even when index.html is opened directly as a
# file (Origin: null) — a read-only local line board, so wildcard CORS is fine.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ──────────────────────────────────────────────────────────── routes ──────────

@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


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
def api_scorecard(sport: str = Query("", description="MLB | Tennis | WNBA | empty = all")):
    """Running model-vs-market scorecard: hit-rate vs the close, plays, and CLV.
    Builds up over time as the CLV ledger logs lines and grades outcomes."""
    return db.scorecard(sport or None)


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
