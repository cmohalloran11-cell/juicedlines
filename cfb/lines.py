"""
cfb.lines — turns The Odds API's player-prop markets into board Line-dicts (pullers.py's
unified Line schema, sport="CFB") and resolves each player to our canonical id.

`fetch_cfb_props` is registered in books.REGISTRY (see main.py/build_static.py's existing
`books.fetch_extra_books()` call, unchanged by this module) so both deploy paths pick CFB up
with no further wiring -- the exact "pluggable connector per book... drops straight into the
merge in main.refresh_lines" contract books.py's own docstring describes.

GATING (task requirement): a prop is only ever emitted here for a market The Odds API
actually returned odds for. We project all 134 FBS teams internally (see cfb/players_sync.py)
independently of this -- this module never walks the internal roster to synthesize a line for
a player/market nobody posted; it only ever transforms what player_props() actually returned.
"""
from __future__ import annotations

from typing import Optional

import store
from pullers import _american_to_implied
from .config import ODDS_MARKET_TO_STAT, odds_api_key
from .data.odds_provider import TheOddsApiAdapter
from .player_matching import normalize_name, resolve_or_log
from .repositories import PlayerRepository


def _resolve_display(db, player_id: Optional[str], raw_name: str) -> tuple[Optional[str], Optional[str]]:
    """(team, position) from the canonical player row if resolved, else (None, None) --
    honestly unknown rather than guessed, same as every other sport's unresolved-player path."""
    if not player_id:
        return None, None
    row = PlayerRepository(db).get(player_id)
    if not row:
        return None, None
    return row.get("team"), row.get("position")


def fetch_cfb_props(sport_filter: Optional[str] = None) -> tuple[list[dict], Optional[str]]:
    """Live CFB player props -> board Line dicts. Mirrors books.fetch_sleeper's (lines, error)
    contract. No ODDS_API_KEY configured -> the adapter returns [] for every call, so this is
    a clean, silent no-op ([], None) -- not an error, matching the README's "auth/AI features
    simply stay disabled until their env vars are set" philosophy for every optional
    integration in this repo."""
    if sport_filter and sport_filter.upper() != "CFB":
        return [], None
    if not odds_api_key():
        return [], None

    db = store.get_database()
    adapter = TheOddsApiAdapter()
    markets = list(ODDS_MARKET_TO_STAT)
    out: list[dict] = []
    try:
        events = adapter.events()
    except Exception as exc:
        return [], f"cfb odds events: {exc}"

    for ev in events:
        eid = ev.get("id")
        if not eid:
            continue
        try:
            props = adapter.player_props(eid, markets)
        except Exception:
            continue
        for p in props:
            stat = ODDS_MARKET_TO_STAT.get(p.market)
            if not stat:
                continue
            source_id = normalize_name(p.player)
            player_id = resolve_or_log(db, source="the_odds_api", source_id=source_id,
                                       raw_name=p.player)
            team, position = _resolve_display(db, player_id, p.player)
            row = {
                "id": f"cfb_odds_{eid}_{p.book}_{p.market}_{source_id}",
                "source": p.book, "sport": "CFB", "player": p.player,
                "team": team, "position": position, "stat_type": stat, "line": p.line,
                "odds_type": "standard",
                "matchup": f"{p.away_team} @ {p.home_team}" if p.away_team and p.home_team else None,
                "start_time": p.commence_time, "status": None,
                "over_implied": _american_to_implied(p.over_price),
                "under_implied": _american_to_implied(p.under_price),
                "over_price": p.over_price, "under_price": p.under_price, "headshot": None,
                "meta": {"event_id": eid, "cfb_player_id": player_id},
                "game_id": eid,
            }
            out.append(row)
    return out, None
