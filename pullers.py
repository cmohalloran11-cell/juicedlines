"""
pullers.py — Adapters for PrizePicks and Underdog that normalize
output into the unified Line schema served by the API.

Unified Line schema:
{
  "id":            str,          # unique key: "ud_{over_under_id}", "pp_{id}"
  "source":        str,          # "underdog" | "prizepicks"
  "sport":         str,          # "MLB" | "World Cup" | "other"
  "player":        str | None,
  "team":          str | None,
  "position":      str | None,
  "stat_type":     str | None,   # "Strikeouts", "Goals", etc.
  "line":          float | None, # the O/U number
  "odds_type":     str | None,   # "standard"|"demon"|"goblin"
  "matchup":       str | None,
  "start_time":    str | None,   # ISO8601
  "status":        str | None,
  "over_implied":  float | None, # 0–1 implied prob of OVER / YES
  "under_implied": float | None,
  "over_price":    str | None,   # american odds string e.g. "-139"
  "under_price":   str | None,
  "meta":          dict,         # source-specific extras
}
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any

import json as _json

import requests

_BD = Path(__file__).parent.parent / "betting_dashboard"
_CONFIG_PATH = Path(__file__).parent / "config.json"


def _load_config() -> dict:
    try:
        return _json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
# Local dev pulls the clients from the sibling betting_dashboard; a standalone deploy
# has vendored copies (underdog.py/mlb_model.py) alongside this file instead.
if _BD.exists() and str(_BD) not in sys.path:
    sys.path.insert(0, str(_BD))

# PrizePicks now reads straight from the partner API (see fetch_prizepicks) — no
# library, no cookie. The old `prizepicks` client is intentionally not imported.

try:
    from underdog import Underdog, UnderdogProp
    _UD_OK = True
except ImportError:
    _UD_OK = False


# ─────────────────────────────────── sport detection helpers ─────────────────

_UD_SPORT_MAP: dict[str, str] = {
    "MLB": "MLB",
    "FIFA": "World Cup",
    "KBO": "other",
    "WNBA": "WNBA",
    "PGA": "other",
    "NFL": "other",
    "CFL": "other",
    "TENNIS": "Tennis",
    "MMA": "other",
    "BOXING": "other",
    "ESPORTS": "other",
    "RACING": "other",
    "BASKETBALL": "other",
    "NPB": "other",
}

_PP_LEAGUE_MLB = {"mlb", "baseball"}
_PP_LEAGUE_WC  = {"fifa", "world cup", "soccer", "copa", "euro 2024", "euro"}
_PP_LEAGUE_TENNIS = {"tennis", "atp", "wta"}


def _tennis_opp(p) -> str | None:
    """Opponent name from a tennis prop title, e.g. 'Sinner Games O/U (vs Zverev)'."""
    title = ((getattr(p, "raw", {}) or {}).get("over_under") or {}).get("title", "") or getattr(p, "player_name", "") or ""
    m = re.search(r"\(vs\.?\s+(.+?)\)\s*$", title)
    return m.group(1).strip() if m else None


def _sport_from_pp_league(league: str | None) -> str:
    if not league:
        return "other"
    l = league.strip().lower()
    # Basketball — must precede the nba/basket guard below (WNBA & NBASL contain "nba").
    # Exclude period/half/quarter/season sub-leagues (WNBA1H, WNBA1Q, NBASL2H, WNBASZN…):
    # their stat_type is plain "Points"/"Rebounds" with a small line, which the full-game
    # model would wildly over-project. Only the full-game league maps to the sport.
    _period = ("1h", "2h", "1q", "2q", "3q", "4q", "szn")
    if "wnba" in l:
        return "other" if any(t in l for t in _period) else "WNBA"
    if "nbasl" in l or "summer league" in l:
        return "other" if any(t in l for t in _period) else "NBA Summer League"
    if any(x in l for x in _PP_LEAGUE_TENNIS):
        return "Tennis"
    # Guard against leagues that share a token (e.g. "EUROGOLF", regular-season NBA).
    if any(x in l for x in ("golf", "basket", "hockey", "nascar",
                            "cricket", "rugby", "nba", "nfl")):
        return "other"
    if l in _PP_LEAGUE_MLB or "mlb" in l or "baseball" in l:
        return "MLB"
    if any(k in l for k in ("fifa", "world cup", "soccer", "copa", "euro")):
        return "World Cup"
    return "other"


def _american_to_implied(price: str | None) -> float | None:
    """Convert american odds string to implied probability (0–1)."""
    if not price:
        return None
    try:
        p = float(price)
        if p > 0:
            return round(100 / (p + 100), 4)
        else:
            return round(abs(p) / (abs(p) + 100), 4)
    except (TypeError, ValueError):
        return None


def _clean_stat(raw: str) -> str:
    """'pitch_outs' → 'Pitch Outs', 'period_1_2_shots_on_target' → 'Shots On Target'"""
    # Strip leading period specifiers like "period_1_2_"
    raw = re.sub(r'^period_\d+_\d+_', '', raw)
    return raw.replace('_', ' ').title()


# ────────────────────────────────────────────── Underdog adapter ─────────────

def _ud_dedup(props: list["UnderdogProp"]) -> list[dict[str, Any]]:
    """
    Underdog emits one UnderdogProp per side (over + under share the same
    over_under_id but have opposite choice values). Deduplicate into one row
    per line, capturing both prices.
    """
    groups: dict[str, dict[str, Any]] = {}

    for p in props:
        # Boosted picks (payout multiplier != 1.0) are Underdog's alt-line / promo
        # variants — excluded entirely (the app tracks standard lines only). This also
        # fixes an old grouping artifact where a boosted choice seen first would shadow
        # the real standard line for that prop.
        if p.is_boosted:
            continue
        # Clean player name: strip trailing " O/U"
        raw_name = p.player_name or ""
        player = re.sub(r'\s+O/U\s*$', '', raw_name).strip()

        # Try to get clean name from raw options
        opts = p.raw.get("options", [])
        if opts:
            player = opts[0].get("selection_header") or player

        sport = _UD_SPORT_MAP.get(p.sport or "", "other")
        stat  = _clean_stat(p.stat or "")
        uid   = f"ud_{p.raw.get('over_under_id') or p.line_id}"

        if uid not in groups:
            groups[uid] = {
                "id": uid,
                "source": "underdog",
                "sport": sport,
                "player": player,
                # Underdog gives no team name (team_id is a UUID); for soccer the
                # country is the meaningful "team".
                "team": p.country if sport == "World Cup" else None,
                "position": p.position,
                "stat_type": stat,
                "line": p.line,
                "odds_type": "standard",   # boosted picks are filtered out above
                "matchup": _tennis_opp(p) if sport == "Tennis" else None,
                "start_time": None,
                "status": p.status,
                "over_implied": None,
                "under_implied": None,
                "over_price": None,
                "under_price": None,
                # Underdog ships a real player headshot — used by the board + drawer.
                "headshot": p.image_url,
                "country": p.country,
                "meta": {
                    "line_id": p.line_id,
                    "match_id": p.match_id,
                    "raw_stat": p.stat,
                    "is_boosted": p.is_boosted,
                },
            }

        row = groups[uid]
        choice = (p.choice or "").lower()
        implied = _american_to_implied(p.american_price)
        if choice in ("over", "higher"):
            row["over_price"]   = p.american_price
            row["over_implied"] = implied
        else:
            row["under_price"]   = p.american_price
            row["under_implied"] = implied

    return list(groups.values())


def fetch_underdog(sport_filter: str | None = None) -> tuple[list[dict], str | None]:
    if not _UD_OK:
        return [], "underdog module not available"
    try:
        ud = Underdog()
        props = ud.get_props()

        # Filter to wanted sports
        wanted = ({"MLB", "World Cup", "Tennis", "WNBA", "NBA Summer League"}
                  if not sport_filter or sport_filter == "all" else {sport_filter})
        filtered = [p for p in props
                    if _UD_SPORT_MAP.get(p.sport or "", "other") in wanted]

        lines = _ud_dedup(filtered)
        return lines, None
    except Exception as exc:
        return [], str(exc)


# ──────────────────────────────────────────── PrizePicks adapter ──────────────
# PrizePicks' public app API (api.prizepicks.com) is behind DataDome bot protection
# (403 → geo.captcha-delivery.com), which no static cookie survives — it rotates and
# fingerprints the browser. The PARTNER API host serves the identical JSON:API feed
# with NO bot wall and NO auth, so we read straight from it. No cookie, no library.
_PP_PARTNER = "https://partner-api.prizepicks.com"
_PP_WANTED = ("MLB", "World Cup", "Tennis", "WNBA", "NBA Summer League")

# PrizePicks is a flat pick'em — no per-pick moneyline. A STANDARD leg's implied
# price is the break-even of a 2-pick Power play (pays 3x → each leg needs a
# 3^(1/2)≈1.732 decimal payout, i.e. ~57.7% win prob / American ≈ -137). We attach
# that as `pickem_price` so the EV engine can price standard PrizePicks legs.
# Demon/goblin legs carry their own multiplier, which this feed does NOT expose,
# so we leave those unpriced rather than guess.
_PP_PICKEM_DECIMAL = 3.0 ** 0.5


def _decimal_to_american(d: float) -> int:
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))


_PP_PICKEM_AMERICAN = _decimal_to_american(_PP_PICKEM_DECIMAL)   # -137

# The partner host throttles hard — only ~3 requests before a 429 that escalates
# with each retry. So we make ONE all-sports call per fetch (no per-league fan-out;
# ~12k projections come back in a single response), back off on 429, and cache the
# result so board refreshes never re-hammer it.
_PP_TTL = 90.0            # seconds to reuse a full successful pull
_pp_result_cache: dict = {}                      # sport_filter -> (ts, lines)


def _pp_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
        "Accept": "application/json",
    })
    return s


def _pp_get(s: requests.Session, url: str, params: dict | None = None,
            retries: int = 3) -> requests.Response:
    """GET with 429 backoff (honours Retry-After, else exponential)."""
    r = None
    for i in range(retries):
        r = s.get(url, params=params, timeout=30)
        if r.status_code != 429:
            return r
        wait = 0.0
        try:
            wait = float(r.headers.get("Retry-After", "") or 0)
        except ValueError:
            wait = 0.0
        time.sleep(min(wait or 1.5 * (2 ** i), 8.0))
    return r


# PrizePicks posts partial-game props (1st half / quarter) with the SAME full-game
# stat_type (e.g. "Shots") but event_type "*_with_duration" and a period in the
# description ("Spain 1H", "LAC 2nd Half"). Left untouched they collide with the
# full-game line — a "Shots" line at ~half value — and wrongly inherit the full-game
# projection, creating phantom edges (a 4.5-shots player showing a "94% over" on a 2
# line). Tag the period into the stat_type so the line is DISTINCT from the full-game
# prop AND the projection engines' period guards ("1h"/"2h"/"1q"/"half") skip it.
_PP_PERIOD_RE = re.compile(r"\b(1st half|2nd half|first half|second half|[1-4][HQP]|OT)\b", re.I)
_PP_PERIOD_NORM = {"1STHALF": "1H", "2NDHALF": "2H", "FIRSTHALF": "1H", "SECONDHALF": "2H"}


def _pp_period_tag(description: str | None) -> str:
    """Compact period label from a PrizePicks description, e.g. 'Spain 1H' → '1H'.
    Falls back to 'Half' (a guard-recognized token) when the period can't be parsed."""
    m = _PP_PERIOD_RE.search(description or "")
    if not m:
        return "Half"
    t = m.group(1).upper().replace(" ", "")
    return _PP_PERIOD_NORM.get(t, t)


def _pp_line(proj: dict, idx: dict) -> dict[str, Any]:
    """Map one JSON:API projection (+ the payload's `included` index) to a Line."""
    attr = proj.get("attributes", {}) or {}
    rel = proj.get("relationships", {}) or {}

    def resolve(name: str) -> dict:
        ref = ((rel.get(name) or {}).get("data")) or {}
        return idx.get((ref.get("type"), ref.get("id")), {}) or {}

    pl = resolve("new_player")
    pa = pl.get("attributes", {}) or {}
    lg = resolve("league")
    league_name = (lg.get("attributes", {}) or {}).get("name") or pa.get("league")
    sport = _sport_from_pp_league(league_name)

    # tag partial-game props so they don't masquerade as (or collide with) full-game lines
    stat = attr.get("stat_type")
    if "with_duration" in (attr.get("event_type") or "") and stat:
        stat = f"{stat} ({_pp_period_tag(attr.get('description'))})"

    return {
        "id": f"pp_{proj.get('id')}",
        "source": "prizepicks",
        "sport": sport,
        "player": pa.get("display_name") or pa.get("name"),
        "team": pa.get("team"),
        "position": pa.get("position"),
        "stat_type": stat,
        "line": attr.get("line_score"),
        "odds_type": attr.get("odds_type") or "standard",
        "matchup": attr.get("description"),   # for tennis this is the opponent name
        "start_time": attr.get("start_time"),
        "status": attr.get("status"),
        "over_implied": None,
        "under_implied": None,
        "over_price": None,
        "under_price": None,
        # standard pick'em legs get the 2-pick Power break-even price for EV; the
        # feed doesn't expose demon/goblin multipliers, so those stay unpriced.
        "pickem_price": _PP_PICKEM_AMERICAN if (attr.get("odds_type") or "standard") == "standard" else None,
        "headshot": pa.get("image_url"),      # PrizePicks ships a player headshot
        "country": pa.get("team") if sport == "World Cup" else None,
        "meta": {
            "player_id": pl.get("id"),
            "league": league_name,
            "league_id": lg.get("id"),
            "is_promo": attr.get("is_promo"),
            "rank": attr.get("rank"),
        },
    }


def fetch_prizepicks(sport_filter: str | None = None) -> tuple[list[dict], str | None]:
    key = sport_filter or "all"
    now = time.time()
    hit = _pp_result_cache.get(key)
    if hit and now - hit[0] < _PP_TTL:
        return hit[1], None
    try:
        s = _pp_session()
        # ONE request returns every projection across all sports (~12k). This is far
        # more reliable than per-league fan-out, which 429s after ~3 calls. Filter to
        # the sports the board uses client-side.
        r = _pp_get(s, f"{_PP_PARTNER}/projections", params={"per_page": 5000})
        if r.status_code != 200:
            return [], f"PrizePicks partner API {r.status_code} (rate-limited); retrying next refresh"
        payload = r.json()
        idx = {(i.get("type"), i.get("id")): i for i in payload.get("included", [])}
        want = None if (not sport_filter or sport_filter == "all") else sport_filter.lower()
        lines: list[dict] = []
        for p in payload.get("data", []):
            row = _pp_line(p, idx)
            # models are per-player → drop multi-player / labelled combo props
            if (row["player"] and " + " in row["player"]) or \
               (row["stat_type"] and "(Combo)" in row["stat_type"]):
                continue
            if want is None:
                if row["sport"] not in _PP_WANTED:
                    continue
            elif row["sport"].lower() != want:
                continue
            lines.append(row)
        _pp_result_cache[key] = (now, lines)
        return lines, None
    except Exception as exc:
        return [], str(exc)


# ──────────────────────────────────────────────────── mock data ───────────────

def mock_lines() -> list[dict]:
    """Small dataset for offline/dev use. Active when real sources return 0 lines."""
    _base = {"position": None, "matchup": None, "start_time": None, "status": "pre_game",
             "over_price": None, "under_price": None, "over_implied": None, "under_implied": None}
    return [
        {**_base, "id":"pp_mock_1","source":"prizepicks","sport":"MLB","player":"Aaron Judge","team":"NYY","stat_type":"Home Runs","line":0.5,"odds_type":"standard","meta":{}},
        {**_base, "id":"pp_mock_2","source":"prizepicks","sport":"MLB","player":"Shohei Ohtani","team":"LAD","stat_type":"Hits+Runs+RBI","line":2.5,"odds_type":"standard","meta":{}},
        {**_base, "id":"pp_mock_3","source":"prizepicks","sport":"MLB","player":"Corbin Carroll","team":"ARI","stat_type":"Stolen Bases","line":0.5,"odds_type":"demon","meta":{}},
        {**_base, "id":"ud_mock_1","source":"underdog","sport":"MLB","player":"Paul Skenes","team":"PIT","stat_type":"Strikeouts","line":7.5,"odds_type":"standard","over_price":"-139","under_price":"+105","over_implied":0.582,"under_implied":0.488,"meta":{}},
        {**_base, "id":"ud_mock_2","source":"underdog","sport":"World Cup","player":"Kylian Mbappé","team":"FRA","stat_type":"Shots On Target","line":1.5,"odds_type":"standard","over_price":"-110","under_price":"-110","over_implied":0.524,"under_implied":0.524,"meta":{}},
    ]
