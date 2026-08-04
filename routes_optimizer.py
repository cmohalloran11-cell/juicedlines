"""
routes_optimizer.py — HTTP surface for the Entry Optimizer (optimizer.py). Mounted by
main.py (`app.include_router(optimizer_router)`).

Reads the live line cache from `main._cache` lazily (import happens inside the request
handler, not at module load) — this file can be imported and its router built with zero
import-order dependency on main.py; the cache is only read once main.py has actually run and
populated it, which is true by the time any real request arrives.
"""
from __future__ import annotations

from fastapi import APIRouter, Query

import optimizer

router = APIRouter(prefix="/api", tags=["optimizer"])

_BOOK_DESC = "Every leg in an entry comes from the same book. " + " | ".join(optimizer.SUPPORTED_BOOKS)


def _current_lines() -> list[dict]:
    import main  # local import: avoids a hard import-time dependency on main.py
    return main._cache.get("lines") or []


@router.get("/optimizer/today")
def api_optimizer_today(sport: str = Query("all", description="All | MLB | WNBA | Tennis"),
                        book: str = Query(optimizer.DEFAULT_BOOK, description=_BOOK_DESC)):
    """Today's Best grid (one top entry per format `book` supports) + a cross-format Top
    Entries leaderboard — see optimizer.today_report for the full pipeline."""
    return optimizer.today_report(_current_lines(), sport=None if sport.lower() == "all" else sport,
                                  book=book)


@router.get("/optimizer/format")
def api_optimizer_format(picks: int = Query(..., ge=2, le=6),
                         kind: str = Query(..., description="power | flex"),
                         top_n: int = Query(3, ge=1, le=10),
                         sport: str = Query("all", description="All | MLB | WNBA | Tennis"),
                         book: str = Query(optimizer.DEFAULT_BOOK, description=_BOOK_DESC)):
    """Ranked entries for one specific format (e.g. picks=4&kind=flex) — used by the Entry
    Optimizer page when a user drills into a single format instead of the daily overview."""
    pool = optimizer.build_pool(_current_lines())
    sp = None if sport.lower() == "all" else sport
    entries = optimizer.best_entries(pool, picks, kind.lower(), top_n=top_n, sport=sp, book=book)
    return {"picks": picks, "kind": kind.lower(), "book": book.lower(), "entries": entries}


@router.get("/optimizer/books")
def api_optimizer_books():
    """Which books have a verified multi-pick payout table (see optimizer.BOOK_PAYOUTS) —
    lets the frontend disable/label books it can't build real entries for yet."""
    return {"books": [{"id": b, "available": bool(optimizer.BOOK_PAYOUTS.get(b))}
                      for b in optimizer.SUPPORTED_BOOKS]}
