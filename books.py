"""
books.py — sportsbook line connectors (spec Vol IV — DataOS / market data pipeline).

A pluggable connector per book. Each `fetch_*` returns (lines, error) in the same Line-dict
shape the board already uses, so new books drop straight into the merge in main.refresh_lines.

Status of the five books the UI lists (+ DraftKings Pick6):
  • prizepicks  — LIVE  (partner-api, in pullers.py)
  • underdog    — LIVE  (partner-api client, in pullers.py)
  • sleeper     — LIVE  (public /lines/available + /players/{sport} — implemented here)
  • DraftKings Pick6 / ParlayPlay / Chalkboard — BOT-WALLED server-side (Akamai / Cloudflare):
    a plain server request gets 403/blocked, so they are intentionally NOT listed. Re-add an
    entry (with a `fetch`) if a licensed odds-provider feed or a working session is available.

REGISTRY drives /api/books so the frontend renders exactly the books we can pull.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import requests

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# ── Sleeper ───────────────────────────────────────────────────────────────────
_SLEEPER_LINES = "https://api.sleeper.app/lines/available?include_preseason=true"
_SLEEPER_PLAYERS = "https://api.sleeper.app/players/{sport}"
# Sleeper sport code → our sport label. Only sports the engine can project are enabled;
# others (nba/nfl/nhl/cs) can be added once their models exist — the data already flows.
_SLEEPER_SPORTS = {"mlb": "MLB", "wnba": "WNBA", "tennis": "Tennis"}

# Sleeper wager_type → our stat label (matched to what analytics._resolve_stat expects).
_SLEEPER_STAT = {
    # MLB
    "hits": "Hits", "total_bases": "Total Bases", "home_runs": "Home Runs", "rbis": "RBIs",
    "runs": "Runs", "singles": "Singles", "doubles": "Doubles", "stolen_bases": "Stolen Bases",
    "bat_walks": "Walks", "hits_runs_rbis": "Hits+Runs+RBIs", "strike_outs": "Strikeouts",
    "outs": "Pitching Outs", "hits_allowed": "Hits Allowed", "earned_runs": "Earned Runs",
    "walks": "Walks Allowed", "first_inning_runs": "1st Inning Runs Allowed",
    # WNBA
    "points": "Points", "rebounds": "Rebounds", "assists": "Assists",
    "threes_made": "3-PT Made", "pts_reb_ast": "Pts+Rebs+Asts",
    # Tennis
    "aces": "Aces", "games_won": "Games Won", "games_played": "Total Games",
    "breakpts_won": "Break Points Won",
}

_players_cache: dict[str, tuple[float, dict]] = {}   # sport -> (ts, {id: "First Last"})
_PLAYERS_TTL = 86_400.0
_lines_cache: tuple[float, list] | None = None
_LINES_TTL = 90.0


def _sleeper_players(sport: str) -> dict:
    hit = _players_cache.get(sport)
    if hit and time.time() - hit[0] < _PLAYERS_TTL:
        return hit[1]
    try:
        r = requests.get(_SLEEPER_PLAYERS.format(sport=sport),
                         headers={"User-Agent": _UA}, timeout=30)
        raw = r.json() if r.status_code == 200 else {}
    except Exception:
        raw = {}
    names = {}
    for pid, p in (raw or {}).items():
        if not isinstance(p, dict):
            continue
        fn, ln = p.get("first_name") or "", p.get("last_name") or ""
        nm = (fn + " " + ln).strip()
        if nm:
            names[str(pid)] = nm
    _players_cache[sport] = (time.time(), names)
    return names


def fetch_sleeper(sport_filter: Optional[str] = None) -> tuple[list[dict], Optional[str]]:
    """Live Sleeper pick'em lines → board Line dicts. Cached 90s."""
    global _lines_cache
    now = time.time()
    if _lines_cache and now - _lines_cache[0] < _LINES_TTL:
        entries = _lines_cache[1]
    else:
        try:
            r = requests.get(_SLEEPER_LINES, headers={"User-Agent": _UA}, timeout=30)
            if r.status_code != 200:
                return [], f"sleeper HTTP {r.status_code}"
            entries = r.json()
        except Exception as exc:
            return [], f"sleeper: {exc}"
        _lines_cache = (now, entries)

    want = {s.lower() for s in ([sport_filter] if sport_filter else _SLEEPER_SPORTS.values())}
    players_by_sport: dict[str, dict] = {}
    out: list[dict] = []

    for e in entries:
        opts = e.get("options") or []
        if not opts:
            continue
        base = opts[0]
        sp_code = (base.get("sport") or "").lower()
        sport = _SLEEPER_SPORTS.get(sp_code)
        if not sport or sport.lower() not in want:
            continue
        wager = base.get("wager_type") or ""
        stat = _SLEEPER_STAT.get(wager)
        if not stat:
            continue
        if players_by_sport.get(sp_code) is None:
            players_by_sport[sp_code] = _sleeper_players(sp_code)
        pid = str(base.get("subject_id") or "")
        player = players_by_sport[sp_code].get(pid)
        if not player:
            continue
        try:
            line_val = float(base.get("outcome_value"))
        except (TypeError, ValueError):
            continue
        lt = (base.get("line_type") or "normal").lower()
        row = {
            "id": f"sl_{base.get('line_id') or base.get('market_type')}",
            "source": "sleeper", "sport": sport, "player": player,
            "team": base.get("subject_team"), "position": base.get("subject_position"),
            "stat_type": stat, "line": line_val,
            "odds_type": "standard" if lt == "normal" else lt,
            "matchup": None, "start_time": None, "status": base.get("status"),
            "over_implied": None, "under_implied": None,
            "over_price": None, "under_price": None, "headshot": None,
            "meta": {"line_id": base.get("line_id"), "wager_type": wager,
                     "game_id": base.get("game_id")},
        }
        for o in opts:
            try:
                mult = float(o.get("payout_multiplier"))
            except (TypeError, ValueError):
                mult = None
            imp = round(1.0 / mult, 4) if mult and mult > 0 else None
            if (o.get("outcome") or "").lower() == "over":
                row["over_implied"] = imp
            else:
                row["under_implied"] = imp
        out.append(row)

    return out, None


# ── registry ──────────────────────────────────────────────────────────────────
# Only books we can actually pull are listed. DraftKings Pick6 / ParlayPlay / Chalkboard
# are bot-walled (Akamai/Cloudflare) with no server-reachable feed, so they're omitted
# entirely rather than shown as dead options — re-add here with a `fetch` if a licensed
# odds-provider feed or a working session becomes available.
# 'core' books live in pullers.py; 'extra' books are fetched here and merged in.
REGISTRY = [
    {"key": "prizepicks", "name": "PrizePicks", "status": "live", "where": "core"},
    {"key": "underdog", "name": "Underdog", "status": "live", "where": "core"},
    {"key": "sleeper", "name": "Sleeper", "status": "live", "where": "extra", "fetch": fetch_sleeper},
]


def fetch_extra_books(sport_filter: Optional[str] = None) -> tuple[list[dict], dict]:
    """Fetch all 'extra' books (currently: Sleeper live; others blocked). Returns
    (merged_lines, {book_key: error_or_None})."""
    lines: list[dict] = []
    errors: dict[str, Optional[str]] = {}
    for b in REGISTRY:
        if b["where"] != "extra":
            continue
        try:
            got, err = b["fetch"](sport_filter)
        except Exception as exc:
            got, err = [], str(exc)
        if got:
            lines.extend(got)
        if err:
            errors[b["key"]] = err
    return lines, errors


def status() -> list[dict]:
    """Public book status for /api/books (no fetch functions leaked)."""
    return [{"key": b["key"], "name": b["name"], "status": b["status"]} for b in REGISTRY]
