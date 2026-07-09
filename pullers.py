"""
pullers.py — Adapters for Kalshi, PrizePicks, and Underdog that normalize
output into the unified Line schema served by the API.

Unified Line schema:
{
  "id":            str,          # unique key: "ud_{over_under_id}", "pp_{id}", "kalshi_{ticker}"
  "source":        str,          # "underdog" | "prizepicks" | "kalshi"
  "sport":         str,          # "MLB" | "World Cup" | "other"
  "player":        str | None,
  "team":          str | None,
  "position":      str | None,
  "stat_type":     str | None,   # "Strikeouts", "Goals", etc.
  "line":          float | None, # the O/U number
  "odds_type":     str | None,   # "standard"|"demon"|"goblin"|"prediction"|"boosted"
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
from pathlib import Path
from typing import Any

import json as _json

_BD = Path(__file__).parent.parent / "betting_dashboard"
_CONFIG_PATH = Path(__file__).parent / "config.json"


def _load_config() -> dict:
    try:
        return _json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
if str(_BD) not in sys.path:
    sys.path.insert(0, str(_BD))

try:
    from kalshi import Kalshi, KalshiMarket
    _KALSHI_OK = True
except ImportError:
    _KALSHI_OK = False

try:
    from prizepicks import PrizePicks, Projection
    _PP_OK = True
except ImportError:
    _PP_OK = False

try:
    from underdog import Underdog, UnderdogProp
    _UD_OK = True
except ImportError:
    _UD_OK = False


# ─────────────────────────────────── sport detection helpers ─────────────────

_MLB_RE = re.compile(
    r"\bMLB\b|baseball|home\s+run|strikeout|earned\s+run|batting|pitcher|hitter|runs?\s+scored",
    re.I,
)
_WC_RE = re.compile(
    r"\bsoccer\b|FIFA|World\s+Cup|WCUP|Copa|EURO|\bgoal\b|\bshots?\s+on\s+target\b",
    re.I,
)

_UD_SPORT_MAP: dict[str, str] = {
    "MLB": "MLB",
    "FIFA": "World Cup",
    "KBO": "other",
    "WNBA": "other",
    "PGA": "other",
    "NFL": "other",
    "CFL": "other",
    "TENNIS": "other",
    "MMA": "other",
    "BOXING": "other",
    "ESPORTS": "other",
    "RACING": "other",
    "BASKETBALL": "other",
    "NPB": "other",
}

_PP_LEAGUE_MLB = {"mlb", "baseball"}
_PP_LEAGUE_WC  = {"fifa", "world cup", "soccer", "copa", "euro 2024", "euro"}


def _sport_from_text(*texts: str | None) -> str:
    combined = " ".join(t for t in texts if t)
    if _MLB_RE.search(combined):
        return "MLB"
    if _WC_RE.search(combined):
        return "World Cup"
    return "other"


def _sport_from_pp_league(league: str | None) -> str:
    if not league:
        return "other"
    l = league.strip().lower()
    # Guard against non-soccer leagues that share a token (e.g. "EUROGOLF").
    if any(x in l for x in ("golf", "tennis", "basket", "hockey", "nascar",
                            "cricket", "rugby", "nba", "nfl", "wnba")):
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
                "odds_type": "boosted" if p.is_boosted else "standard",
                "matchup": None,
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
        wanted = {"MLB", "World Cup"} if not sport_filter or sport_filter == "all" else {sport_filter}
        filtered = [p for p in props
                    if _UD_SPORT_MAP.get(p.sport or "", "other") in wanted]

        lines = _ud_dedup(filtered)
        return lines, None
    except Exception as exc:
        return [], str(exc)


# ───────────────────────────────────────────────── Kalshi adapter ─────────────

def _kalshi_line(m: "KalshiMarket") -> dict[str, Any]:
    yes_prob = m.implied_prob
    no_prob  = round(1 - yes_prob, 4) if yes_prob is not None else None
    sport    = _sport_from_text(m.ticker, m.title, m.subtitle)
    return {
        "id": f"kalshi_{m.ticker}",
        "source": "kalshi",
        "sport": sport,
        "player": None,
        "team": None,
        "position": None,
        "stat_type": m.subtitle or m.title,
        "line": None,
        "odds_type": "prediction",
        "matchup": m.title,
        "start_time": m.close_time,
        "status": m.status,
        "over_implied": yes_prob,
        "under_implied": no_prob,
        "over_price": None,
        "under_price": None,
        "meta": {
            "ticker": m.ticker,
            "event_ticker": m.event_ticker,
            "yes_bid": m.yes_bid,
            "yes_ask": m.yes_ask,
            "volume": m.volume,
            "volume_24h": m.volume_24h,
            "open_interest": m.open_interest,
        },
    }


def fetch_kalshi(sport_filter: str | None = None) -> tuple[list[dict], str | None]:
    if not _KALSHI_OK:
        return [], "kalshi module not available"
    try:
        k = Kalshi()
        markets = k.get_markets(status="open", limit=2000)
        lines = [_kalshi_line(m) for m in markets]
        if sport_filter and sport_filter != "all":
            lines = [l for l in lines if l["sport"].lower() == sport_filter.lower()]
        else:
            lines = [l for l in lines if l["sport"] in ("MLB", "World Cup")]
        return lines, None
    except Exception as exc:
        return [], str(exc)


# ──────────────────────────────────────────── PrizePicks adapter ──────────────

def _pp_line(p: "Projection") -> dict[str, Any]:
    sport = _sport_from_pp_league(p.league)
    return {
        "id": f"pp_{p.id}",
        "source": "prizepicks",
        "sport": sport,
        "player": p.player_name,
        "team": p.team,
        "position": p.position,
        "stat_type": p.stat_type,
        "line": p.line_score,
        "odds_type": p.odds_type or "standard",
        "matchup": p.description,
        "start_time": p.start_time,
        "status": p.status,
        "over_implied": None,
        "under_implied": None,
        "over_price": None,
        "under_price": None,
        "headshot": p.image_url,           # PrizePicks ships a player headshot
        "country": p.team if sport == "World Cup" else None,
        "meta": {
            "player_id": p.player_id,
            "league": p.league,
            "league_id": p.league_id,
            "is_promo": p.is_promo,
            "rank": p.rank,
        },
    }


def fetch_prizepicks(sport_filter: str | None = None) -> tuple[list[dict], str | None]:
    if not _PP_OK:
        return [], "prizepicks module not available"
    try:
        cfg = _load_config()
        extra_headers: dict[str, str] = {}
        if cfg.get("prizepicks_cookie"):
            extra_headers["Cookie"] = cfg["prizepicks_cookie"]
        pp = PrizePicks(request_delay=0.5, headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Accept": "application/json",
            "Origin": "https://app.prizepicks.com",
            "Referer": "https://app.prizepicks.com/",
            **extra_headers,
        })
        leagues = pp.get_leagues()
        wanted: list[str] = []
        for lg in leagues:
            s = _sport_from_pp_league(lg.name)
            if sport_filter and sport_filter != "all":
                if s.lower() != sport_filter.lower():
                    continue
            else:
                if s not in ("MLB", "World Cup"):
                    continue
            if (lg.projections_count or 0) > 0:
                wanted.append(lg.id)
        lines: list[dict] = []
        for lid in wanted:
            projs = pp.get_projections(league_id=lid)
            lines.extend(_pp_line(p) for p in projs)
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
        {**_base, "id":"kalshi_mock_1","source":"kalshi","sport":"MLB","player":None,"team":None,"stat_type":"NYY to win today","line":None,"odds_type":"prediction","over_implied":0.58,"under_implied":0.42,"meta":{"ticker":"KXMLB-25JUN-NYY","yes_bid":57,"yes_ask":59}},
        {**_base, "id":"kalshi_mock_2","source":"kalshi","sport":"World Cup","player":None,"team":None,"stat_type":"France to win","line":None,"odds_type":"prediction","over_implied":0.49,"under_implied":0.51,"meta":{"ticker":"KXWC-FRA-ARG","yes_bid":48,"yes_ask":50}},
    ]
