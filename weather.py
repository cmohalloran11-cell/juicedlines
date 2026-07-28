"""
weather.py — Vaisala Xweather (Aeris API) integration for outdoor-game conditions.

Powers the Weather page and the model's park/weather factor. MLB is the sport that matters
(outdoor); basketball/tennis are indoor or not weather-driven here, so we map each MLB team
to its ballpark city and pull current conditions. Degrades cleanly when unconfigured.

Config (env): XWEATHER_CLIENT_ID, XWEATHER_CLIENT_SECRET.
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

import requests

_BASE = "https://api.aerisapi.com/conditions/{loc}"

# MLB team abbrev → ballpark city,state for the Xweather lookup. (Retractable-roof parks are
# still queried — the page notes conditions; the model can weight them down.)
_MLB_PARK = {
    "ARI": "phoenix,az", "ATL": "atlanta,ga", "BAL": "baltimore,md", "BOS": "boston,ma",
    "CHC": "chicago,il", "CWS": "chicago,il", "CIN": "cincinnati,oh", "CLE": "cleveland,oh",
    "COL": "denver,co", "DET": "detroit,mi", "HOU": "houston,tx", "KC": "kansas city,mo",
    "LAA": "anaheim,ca", "LAD": "los angeles,ca", "MIA": "miami,fl", "MIL": "milwaukee,wi",
    "MIN": "minneapolis,mn", "NYM": "flushing,ny", "NYY": "bronx,ny", "OAK": "oakland,ca",
    "ATH": "sacramento,ca", "PHI": "philadelphia,pa", "PIT": "pittsburgh,pa", "SD": "san diego,ca",
    "SF": "san francisco,ca", "SEA": "seattle,wa", "STL": "st louis,mo", "TB": "st petersburg,fl",
    "TEX": "arlington,tx", "TOR": "toronto,canada", "WSH": "washington,dc", "WAS": "washington,dc",
}

_cache: dict[str, tuple[float, dict]] = {}
_TTL = 900.0  # 15 min


def _creds() -> Optional[tuple[str, str]]:
    cid, sec = os.getenv("XWEATHER_CLIENT_ID"), os.getenv("XWEATHER_CLIENT_SECRET")
    return (cid, sec) if cid and sec else None


def available() -> bool:
    return _creds() is not None


def _fetch(loc: str) -> Optional[dict]:
    creds = _creds()
    if not creds:
        return None
    now = time.time()
    hit = _cache.get(loc)
    if hit and now - hit[0] < _TTL:
        return hit[1]
    try:
        r = requests.get(_BASE.format(loc=loc),
                         params={"client_id": creds[0], "client_secret": creds[1],
                                 "format": "json"}, timeout=8)
        data = r.json()
    except Exception:
        return None
    if not data.get("success"):
        return None
    try:
        p = data["response"][0]["periods"][0]
        out = {
            "tempF": p.get("tempF"), "feelsF": p.get("feelslikeF"),
            "windMPH": p.get("windSpeedMPH"), "windDir": p.get("windDir"),
            "humidity": p.get("humidity"), "weather": p.get("weather"),
            "precipIN": p.get("precipIN"), "icon": p.get("icon"),
        }
    except Exception:
        return None
    _cache[loc] = (now, out)
    return out


def for_team(team: str) -> Optional[dict]:
    loc = _MLB_PARK.get((team or "").upper())
    return _fetch(loc) if loc else None


def status() -> dict:
    return {"available": available(),
            "reason": None if available() else "Set XWEATHER_CLIENT_ID + XWEATHER_CLIENT_SECRET."}


_SLATE_BUDGET = 30.0  # seconds — hard wall-clock cap so a slow/degraded provider can never


def slate(lines: list[dict[str, Any]], limit: int = 16) -> dict:
    """Weather for the home teams playing today (MLB only — outdoor). One card per team.
    Bounded by _SLATE_BUDGET total wall-clock time: each call already has its own timeout,
    but with ~30 teams to try that alone doesn't cap the WHOLE loop, so a degraded/slow
    provider could otherwise stall a build cycle for many minutes. Whatever games were
    fetched before the budget runs out are returned — a partial slate beats a stalled build."""
    if not available():
        return {"available": False, "reason": status()["reason"], "games": []}
    start = time.time()
    seen, games = set(), []
    for l in lines:
        if time.time() - start > _SLATE_BUDGET:
            break
        if l.get("sport") != "MLB":
            continue
        team = l.get("team")
        if not team or team in seen or team.upper() not in _MLB_PARK:
            continue
        seen.add(team)
        w = for_team(team)
        if w:
            games.append({"team": team, "startTime": l.get("start_time"),
                          "city": _MLB_PARK[team.upper()], **w})
        if len(games) >= limit:
            break
    return {"available": True, "games": games}
