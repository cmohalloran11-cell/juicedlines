"""
fantasy.sleeper_client — server-side-only client for the Sleeper fantasy-football API
(https://docs.sleeper.app). Every call in this module must be made from a route handler or a
background job, NEVER proxied straight through to the browser (task constraint).

Rate/volume constraints this module exists to enforce:
  • GET /v1/players/nfl is ~5MB and Sleeper asks it be called at most once per day -- only
    fantasy.players_sync (a scheduled job) calls fetch_all_players(); nothing on a request
    path may call it. See fantasy.repositories.SyncLogRepository for the once-a-day guard.
  • Everything else here is small and safe to call per-request; callers apply their own TTL
    caching (5 min for league reads, 10s for draft picks) at the route-handler layer -- this
    module makes no caching decisions itself, it's a plain HTTP client.
"""
from __future__ import annotations

from typing import Optional

import requests

BASE_URL = "https://api.sleeper.app/v1"
_UA = "juicedlines-fantasy/1.0"
_TIMEOUT = 15.0


class SleeperError(RuntimeError):
    pass


def _get(path: str, timeout: float = _TIMEOUT):
    try:
        resp = requests.get(f"{BASE_URL}{path}", headers={"User-Agent": _UA}, timeout=timeout)
    except requests.RequestException as exc:
        raise SleeperError(f"Sleeper request failed: {exc}") from exc
    if resp.status_code == 404:
        return None
    if resp.status_code != 200:
        raise SleeperError(f"Sleeper HTTP {resp.status_code} for {path}")
    return resp.json()


def get_user(username: str) -> Optional[dict]:
    """Resolve a Sleeper username to their user object (id, avatar, display_name)."""
    return _get(f"/user/{username}")


def get_user_leagues(user_id: str, season: str, sport: str = "nfl") -> list[dict]:
    return _get(f"/user/{user_id}/leagues/{sport}/{season}") or []


def get_league(league_id: str) -> Optional[dict]:
    """League detail: scoring_settings, roster_positions, total_rosters (league size)."""
    return _get(f"/league/{league_id}")


def get_league_rosters(league_id: str) -> list[dict]:
    return _get(f"/league/{league_id}/rosters") or []


def get_league_users(league_id: str) -> list[dict]:
    return _get(f"/league/{league_id}/users") or []


def get_league_drafts(league_id: str) -> list[dict]:
    return _get(f"/league/{league_id}/drafts") or []


def get_draft(draft_id: str) -> Optional[dict]:
    return _get(f"/draft/{draft_id}")


def get_draft_picks(draft_id: str) -> list[dict]:
    """Picks made so far, in draft order. Polled every 10s during a live draft by the caller
    -- this function itself is a plain uncached GET."""
    return _get(f"/draft/{draft_id}/picks") or []


def get_nfl_state() -> Optional[dict]:
    """Sleeper's own clock: current season/week. Used to default `season`/`week` params
    instead of hardcoding a year -- see routes_fantasy.py's lineup route."""
    return _get("/state/nfl")


def fetch_all_players(sport: str = "nfl") -> dict[str, dict]:
    """The ~5MB player dump (Sleeper id -> player object). ONLY call from
    fantasy.players_sync's scheduled job -- see module docstring."""
    return _get(f"/players/{sport}", timeout=60.0) or {}
