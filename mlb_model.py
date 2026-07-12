"""
mlb_model.py — real MLB game-log data for the projection engine.

Feeds sports_projector with actual recent game logs pulled from the MLB Stats API
(statsapi.mlb.com — free, no key, no bot wall). Used to project hitter props and
compare the model against the live DFS lines.

Two cheap, cacheable calls do the work:
  1. /sports/1/players?season=YYYY        -> name -> playerId map (cached ~12h)
  2. /people?personIds=...&hydrate=stats(group=[hitting],type=[gameLog],season=YYYY)
     -> batched per-game logs for many players in one request (cached ~3h)

Only unambiguous *hitting* stats are mapped; anything we can't map (pitcher props,
fantasy points) is skipped so the model never invents a wrong edge.
"""

from __future__ import annotations

import re
import time
import threading
import unicodedata
from datetime import date
from typing import Callable, Dict, List, Optional

import requests

BASE = "https://statsapi.mlb.com/api/v1"
_HEADERS = {"User-Agent": "edge-terminal/1.0", "Accept": "application/json"}
_RECENT_GAMES = 40          # how many recent games feed the projection
_BATCH = 40                 # player ids per hydrate request

_session = requests.Session()
_session.headers.update(_HEADERS)

# ------------------------------------------------------------------ tiny cache
_CACHE: Dict[str, tuple] = {}
_LOCK = threading.Lock()


def _cached(key: str, ttl: float, producer):
    now = time.time()
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    val = producer()
    with _LOCK:
        _CACHE[key] = (time.time(), val)
    return val


def season() -> int:
    return date.today().year


# ----------------------------------------------------------- stat label -> field
# Each entry maps a normalized Underdog stat label to a function over one
# game-log `stat` dict (hitting group) returning the per-game value.
def _g(d: dict, k: str) -> float:
    try:
        return float(d.get(k) or 0)
    except (TypeError, ValueError):
        return 0.0


STAT_MAP: Dict[str, Callable[[dict], float]] = {
    "hits": lambda s: _g(s, "hits"),
    "total bases": lambda s: _g(s, "totalBases"),
    "home runs": lambda s: _g(s, "homeRuns"),
    "runs": lambda s: _g(s, "runs"),
    "rbis": lambda s: _g(s, "rbi"),
    "runs batted in": lambda s: _g(s, "rbi"),
    "hits + runs + rbis": lambda s: _g(s, "hits") + _g(s, "runs") + _g(s, "rbi"),
    "stolen bases": lambda s: _g(s, "stolenBases"),
    "doubles": lambda s: _g(s, "doubles"),
    "triples": lambda s: _g(s, "triples"),
    "singles": lambda s: _g(s, "hits") - _g(s, "doubles") - _g(s, "triples") - _g(s, "homeRuns"),
    "walks": lambda s: _g(s, "baseOnBalls"),
    "batter strikeouts": lambda s: _g(s, "strikeOuts"),
    "hitter strikeouts": lambda s: _g(s, "strikeOuts"),
    "hits + runs": lambda s: _g(s, "hits") + _g(s, "runs"),
    "runs + rbis": lambda s: _g(s, "runs") + _g(s, "rbi"),
}


def map_stat(label: Optional[str]) -> Optional[Callable[[dict], float]]:
    return STAT_MAP.get(_norm_stat(label)) if label else None


def _norm_stat(label: str) -> str:
    return re.sub(r"\s+", " ", str(label).strip().lower())


# ------------------------------------------------------------------ name index
def _norm_name(name: str) -> str:
    """Lowercase, strip accents and punctuation, drop Jr/Sr/II suffixes."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", "")
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def player_index() -> Dict[str, int]:
    """{normalized full name -> playerId} for the current season."""
    def produce():
        r = _session.get(f"{BASE}/sports/1/players", params={"season": season()}, timeout=25)
        r.raise_for_status()
        idx: Dict[str, int] = {}
        for p in r.json().get("people", []):
            idx[_norm_name(p.get("fullName", ""))] = p.get("id")
        return idx
    return _cached(f"mlb_players_{season()}", 12 * 3600, produce)


def resolve_ids(names: List[str]) -> Dict[str, int]:
    """Map display names -> playerId where we can find a match."""
    idx = player_index()
    out: Dict[str, int] = {}
    for nm in names:
        pid = idx.get(_norm_name(nm))
        if pid:
            out[nm] = pid
    return out


# ------------------------------------------------------------------ game logs
def _fetch_logs_batch(ids: List[int]) -> Dict[int, List[dict]]:
    """Batched hitting game logs: {playerId: [per-game stat dicts, oldest->newest]}."""
    hyd = f"stats(group=[hitting],type=[gameLog],season=[{season()}])"
    r = _session.get(f"{BASE}/people",
                     params={"personIds": ",".join(str(i) for i in ids), "hydrate": hyd},
                     timeout=25)
    r.raise_for_status()
    out: Dict[int, List[dict]] = {}
    for person in r.json().get("people", []):
        splits = []
        for block in person.get("stats", []) or []:
            splits = block.get("splits", []) or splits
        out[person.get("id")] = [sp.get("stat", {}) for sp in splits]
    return out


def game_logs(ids: List[int]) -> Dict[int, List[dict]]:
    """Game logs for many players, cached per id, fetched in batches for misses."""
    result: Dict[int, List[dict]] = {}
    misses: List[int] = []
    now = time.time()
    with _LOCK:
        for i in ids:
            hit = _CACHE.get(f"mlb_log_{i}")
            if hit and now - hit[0] < 3 * 3600:
                result[i] = hit[1]
            else:
                misses.append(i)
    for start in range(0, len(misses), _BATCH):
        batch = misses[start:start + _BATCH]
        try:
            fetched = _fetch_logs_batch(batch)
        except requests.RequestException:
            fetched = {}
        with _LOCK:
            for i in batch:
                logs = fetched.get(i, [])
                _CACHE[f"mlb_log_{i}"] = (time.time(), logs)
                result[i] = logs
    return result


# ------------------------------------------------------------ empirical model
# Baseball counting stats are overdispersed and the props market is sharp, so a
# parametric (Normal/Poisson) projection invents huge false edges. Instead we use
# the player's own EMPIRICAL frequency of clearing the line — no distributional
# assumption — with a small pseudo-count shrink toward 0.5 to tame thin samples.
def empirical_prob_over(values: List[float], line: float, prior_k: float = 4.0) -> Optional[float]:
    """P(value > line) from how often the player actually cleared it."""
    vals = [v for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return None
    overs = sum(1 for v in vals if v > line)
    return (overs + prior_k * 0.5) / (n + prior_k)


def values_for(stat_logs: List[dict], value_fn: Callable[[dict], float]) -> List[float]:
    """Per-game values for one stat over the recent window."""
    return [value_fn(s) for s in stat_logs[-_RECENT_GAMES:]]
