"""
Live ESPN adapter for tennis — free, no key (same public API basketball/data/espn.py
already uses for WNBA). Fills the freshness gap the Sackmann mirror can't reach: the
mirror is frozen at 2022, so today's actual top players (Sinner, Alcaraz — literally
#1 and #2 ATP as of writing) have ZERO rows in it and would otherwise never get a
model-driven projection. This adapter gives recently-completed match results (for live
Elo updates + automatic grading) and the current tour ranking (a tier prior for players
the mirror has never seen).

Honest limits: ESPN's free tennis endpoint does NOT expose point-level serve stats
(aces/DFs/serve points won) or a real injury report (the `/injuries` endpoint returns an
empty shell for tennis) or match surface directly. Serve/return RATES therefore still
come from the Sackmann historical fit; surface is inferred from a small known-tournament
table (falls back to "Hard", same default as before — see `surface_for`).
"""

from __future__ import annotations

import time
import unicodedata
from datetime import date, timedelta

import requests

_SITE = "https://site.api.espn.com/apis/site/v2/sports/tennis/{tour_path}"
_S = requests.Session()
_S.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
_cache: dict = {}

_GRAND_SLAMS = {"australian open", "french open", "roland garros", "wimbledon", "us open"}

# Known non-Hard tournaments (majors + the main clay/grass swings). Best-effort — ESPN's
# free feed doesn't expose surface directly. Anything not listed defaults to Hard, same
# as the pre-ESPN behavior, so this can only ever IMPROVE surface accuracy, never regress it.
_CLAY = {"french open", "roland garros", "monte carlo", "madrid open", "italian open",
         "internazionali", "barcelona open", "estoril", "geneva open", "hamburg",
         "bastad", "kitzbuhel", "gstaad", "umag", "rio open", "argentina open",
         "chile open", "brasil open", "acapulco"}
_GRASS = {"wimbledon", "queen's club", "queens club", "halle open", "eastbourne",
          "mallorca championships", "s-hertogenbosch", "newport", "berlin"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace(".", "").replace("'", "").split())


def _get(url: str, ttl: float = 900) -> dict | None:
    now = time.time()
    hit = _cache.get(url)
    if hit and now - hit[0] < ttl:
        return hit[1]
    try:
        d = _S.get(url, timeout=20).json()
    except Exception:
        d = None
    _cache[url] = (now, d)
    return d


def _tour_path(tour: str) -> str:
    return "atp" if tour == "ATP" else "wta"


def best_of_for(tour: str, tournament: str) -> int:
    return 5 if (tour == "ATP" and any(g in _norm(tournament) for g in _GRAND_SLAMS)) else 3


def surface_for(tournament: str) -> str:
    t = _norm(tournament)
    if any(g in t for g in _CLAY):
        return "Clay"
    if any(g in t for g in _GRASS):
        return "Grass"
    return "Hard"


def _matches(tour: str, dates: str | None) -> list[dict]:
    """Raw ESPN scoreboard events -> flat list of singles matches (completed + upcoming)."""
    path = _SITE.format(tour_path=_tour_path(tour)) + "/scoreboard"
    if dates:
        path += f"?dates={dates}"
    d = _get(path, ttl=900)
    # Some tour stops co-locate an ATP + WTA event under one ESPN scoreboard entry (e.g. a
    # combined 500-series week) — querying the ATP endpoint can still hand back a
    # "womens-singles" grouping for that shared event. Filter to the SINGLES grouping that
    # actually matches the tour being queried, not just "any singles, not doubles".
    want_slug = "mens-singles" if tour == "ATP" else "womens-singles"
    out = []
    for ev in (d or {}).get("events", []):
        tname = ev.get("name", "")
        for grp in ev.get("groupings", []) or []:
            if (grp.get("grouping") or {}).get("slug") != want_slug:
                continue                                            # skip doubles / other tour
            for comp in grp.get("competitions", []) or []:
                competitors = comp.get("competitors") or []
                if len(competitors) != 2:
                    continue
                a, b = competitors
                status = ((comp.get("status") or {}).get("type") or {})
                out.append({
                    "id": comp.get("id"), "tournament": tname,
                    "date": (comp.get("date") or "")[:10],
                    "round": (grp.get("grouping") or {}).get("displayName", ""),
                    "surface": surface_for(tname),
                    "completed": bool(status.get("completed")),
                    "best_of": best_of_for(tour, tname),
                    "a_id": str((a.get("athlete") or {}).get("id") or a.get("id") or ""),
                    "a_name": (a.get("athlete") or {}).get("displayName", ""),
                    "a_winner": bool(a.get("winner")),
                    "a_games": [int(ls["value"]) for ls in (a.get("linescores") or [])
                                if ls.get("value") is not None],
                    "b_id": str((b.get("athlete") or {}).get("id") or b.get("id") or ""),
                    "b_name": (b.get("athlete") or {}).get("displayName", ""),
                    "b_winner": bool(b.get("winner")),
                    "b_games": [int(ls["value"]) for ls in (b.get("linescores") or [])
                                if ls.get("value") is not None],
                })
    return out


def completed_matches(tour: str, days_back: int = 14) -> list[dict]:
    """Recently-completed singles matches — feeds live Elo updates and grading."""
    today = date.today()
    span = f"{today - timedelta(days=days_back):%Y%m%d}-{today:%Y%m%d}"
    return [m for m in _matches(tour, span) if m["completed"]]


def upcoming(tour: str, days_ahead: int = 2) -> list[dict]:
    """Today's (+ near-future) scheduled/live singles matches."""
    today = date.today()
    span = f"{today:%Y%m%d}-{today + timedelta(days=days_ahead):%Y%m%d}"
    return [m for m in _matches(tour, span) if not m["completed"]]


def rankings(tour: str) -> dict[str, int]:
    """{normalized player name: current rank} — a tier prior for players the Sackmann
    mirror has never seen (e.g. Sinner/Alcaraz: 0 historical rows, but a real #1/#2 rank)."""
    d = _get(_SITE.format(tour_path=_tour_path(tour)) + "/rankings", ttl=6 * 3600)
    out: dict[str, int] = {}
    try:
        for rk in (d or {}).get("rankings", []):
            for r in rk.get("ranks", []) or []:
                nm = _norm((r.get("athlete") or {}).get("displayName", ""))
                cur = r.get("current")
                if nm and cur:
                    out[nm] = int(cur)
    except Exception:
        pass
    return out
