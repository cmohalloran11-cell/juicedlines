"""
Live ESPN adapter — free, no key. Team crests + real player headshots for NFL, the same
public API `basketball/data/espn.py` already uses for WNBA (same host, same JSON shape,
just pointed at "football/nfl" instead of "basketball/wnba").

Why this exists separately from nflverse: nflverse publishes stats, not images. PrizePicks/
Underdog's own `image_url` for an NFL player is sometimes a team crest instead of a real
face (observed live 2026-08 on a recently-signed player) — ESPN's roster endpoint carries a
real per-athlete headshot URL, so it's used as the authoritative source for both, with the
book's own image kept only as a fallback when no ESPN match is found (never blanked for a
player we simply couldn't match).

Caching: goes through nfl.data.cache.fetch_text (disk-backed), NOT a bare in-process dict.
Found live 2026-08: an in-process-only cache here (the first version of this module) meant
every fresh `python build_static.py` process -- which is EVERY refresh cycle, since
refresh.yml spawns a new process each time (see that workflow's `run_cycle`) -- re-fetched
all 33 ESPN URLs (1 team list + 32 rosters) from scratch, with no cross-cycle memory at all.
At a ~100s fast-cycle cadence that's roughly 1,000+ ESPN requests/hour from one runner IP,
which coincided with WNBA's own ESPN-backed projections going silently to zero (0/2423 live
lines carried a model_proj, with no exception anywhere -- consistent with ESPN rate-limiting
the shared host and every affected call failing closed). Disk caching (12h for team crests,
6h for rosters -- these change rarely) cuts that to ~33 requests per cache window instead of
per cycle, matching the exact problem nfl/data/cache.py's own module docstring describes for
the nflverse pulls.
"""

from __future__ import annotations

import json
import time
import unicodedata

from . import cache

_SITE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

_cache: dict = {}

# Testing hook, mirrors nfl.data.set_source: forces team_assets()/all_headshots() to return
# fixed data with ZERO HTTP calls, so the offline test suite never depends on ESPN being
# reachable (or fast) from wherever tests happen to run. None (the default) means "make real
# calls" -- set_test_override(None, None) restores that.
_override: dict | None = None


def set_test_override(team_assets: dict | None, headshots: dict | None) -> None:
    global _override
    _override = ({"team_assets": team_assets or {}, "headshots": headshots or {}}
                if (team_assets is not None or headshots is not None) else None)


def _norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace(".", "").replace("'", "").split())


def _get(url: str, ttl: float = 1800) -> dict | None:
    """Disk-backed (nfl.data.cache.fetch_text) + a same-process JSON-parse cache on top --
    see the module docstring for why the disk layer is required, not optional, here."""
    now = time.time()
    hit = _cache.get(url)
    if hit and now - hit[0] < ttl:
        return hit[1]
    text = cache.fetch_text(url, ttl, timeout=20.0)
    d = None
    if text is None:
        print(f"[nfl.espn] {url} -> fetch_text returned None (network failure or no cached "
              f"copy — see nfl.data.cache.fetch_text)", flush=True)
    else:
        try:
            d = json.loads(text)
        except Exception as exc:
            print(f"[nfl.espn] {url} -> JSON parse failed: {type(exc).__name__}: {exc}", flush=True)
    _cache[url] = (now, d)
    return d


def team_assets() -> dict:
    """normalized team name/abbr → {id, abbr, logo, name} — same shape and cache pattern as
    basketball.data.espn.EspnBasketball.team_assets. 12h cache (crests don't change)."""
    if _override is not None:
        return _override["team_assets"]
    key = "assets"
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < 12 * 3600:
        return hit[1]
    d = _get(f"{_SITE}/teams", 12 * 3600)
    out: dict = {}
    try:
        for t in d["sports"][0]["leagues"][0]["teams"]:
            tm = t["team"]
            logos = tm.get("logos") or []
            rec = {"id": str(tm.get("id")), "abbr": tm.get("abbreviation"),
                   "logo": (logos[0]["href"] if logos else None),
                   "name": tm.get("displayName")}
            out[_norm_name(tm.get("displayName", ""))] = rec
            if tm.get("abbreviation"):
                out[_norm_name(tm["abbreviation"])] = rec
            if tm.get("shortDisplayName"):
                out[_norm_name(tm["shortDisplayName"])] = rec
    except Exception as exc:
        print(f"[nfl.espn] team_assets parse failed on {type(d)}: {type(exc).__name__}: {exc}", flush=True)
    # nflverse (and this repo's own P.norm_team) normalizes the Rams' real abbreviation
    # "LAR" down to the legacy code "LA" -- see nfl.projections._TEAM_ALIASES's own comment
    # ("LA is the Rams in nflverse"). ESPN has no team keyed "LA" (it uses the real "LAR"
    # everywhere, unlike nflverse), so a lookup by the aliased code silently missed the Rams'
    # crest -- found live 2026-08 (every NFL team's logo populated except the Rams).
    if "lar" in out and "la" not in out:
        out["la"] = out["lar"]
    _cache[key] = (time.time(), out)
    return out


def _team_ids() -> list[str]:
    d = _get(f"{_SITE}/teams", 12 * 3600)
    try:
        return [str(t["team"]["id"]) for t in d["sports"][0]["leagues"][0]["teams"]]
    except Exception:
        return []


def roster_headshots(team_id: str) -> dict:
    """normalized player name → real ESPN headshot URL, for one team's active roster.
    6h cache per team (a real roster changes far less often than that)."""
    key = f"roster::{team_id}"
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < 6 * 3600:
        return hit[1]
    d = _get(f"{_SITE}/teams/{team_id}/roster", 6 * 3600)
    out: dict = {}
    try:
        for group in (d or {}).get("athletes", []):
            for a in group.get("items", []):
                nm = a.get("displayName")
                href = (a.get("headshot") or {}).get("href")
                if nm and href:
                    out[_norm_name(nm)] = href
    except Exception as exc:
        print(f"[nfl.espn] roster_headshots({team_id}) parse failed: {type(exc).__name__}: {exc}", flush=True)
    _cache[key] = (time.time(), out)
    return out


def all_headshots() -> dict:
    """normalized player name → real ESPN headshot URL, across every NFL team. One call per
    team (32), each independently cached — a full league pull only re-hits teams whose
    per-team cache actually expired."""
    if _override is not None:
        return _override["headshots"]
    out: dict = {}
    for tid in _team_ids():
        out.update(roster_headshots(tid))
    return out
