"""
cfb.data.odds_provider — swappable player-prop odds source + a The Odds API adapter.

CFBD carries no player props (task constraint) -- this is the ONLY source of CFB prop lines,
which is why it's a separate interface from CfbDataSource (base.py) rather than another
method bolted onto it. Same spirit as basketball/data/base.py's GameLogSource: the board only
ever sees `PlayerProp`, so a different odds vendor later just implements `OddsProvider`.

Requires an API key: set the ODDS_API_KEY env var (server-side only). Absent, every method
returns an honestly-empty result -- see `_key()`/`_get` below, same pattern as
cfb/data/cfbd_client.py and basketball/data/balldontlie.py.

ENDPOINT SHAPES USED BELOW ARE UNVERIFIED AGAINST A LIVE RESPONSE (this sandbox has no route
to api.the-odds-api.com and no ODDS_API_KEY is available here either way). Built against
the-odds-api's own published, documented `event-odds` endpoint
(https://the-odds-api.com/liveapi/guides/v4/#get-event-odds), which is a stable public
contract, not a scraped/reverse-engineered one:
  GET /v4/sports/{sport}/events?apiKey=
      -> [{id, sport_key, commence_time, home_team, away_team}, ...]
  GET /v4/sports/{sport}/events/{eventId}/odds?apiKey=&regions=us&markets=<csv>&oddsFormat=american
      -> {id, commence_time, home_team, away_team,
          bookmakers: [{key, title, markets: [
              {key, outcomes: [{name ("Over"/"Under"), description (player name),
                                price (American odds int), point (the line)}, ...]}
          ]}]}
If the live shape differs, every parse failure prints a `[cfb.odds_provider]` diagnostic
(flush=True) rather than silently returning nothing indistinguishable from "no key set" --
same pattern already proven this session by balldontlie.py / cfbd_client.py. Cross-check
https://the-odds-api.com/liveapi/guides/v4/ first if a live response turns out to differ.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import requests

from ..config import ODDS_API_BASE_URL, ODDS_API_SPORT_KEY, HTTP_TIMEOUT, CACHE_TTL, odds_api_key

_memory: dict = {}
_warned_no_key = False


@dataclass
class PlayerProp:
    """One player-prop market a real book actually posted for one event. `market` is the
    Odds API market key (e.g. "player_pass_yds") -- the caller maps it to a canonical stat
    label via cfb.config.ODDS_MARKET_TO_STAT, this dataclass doesn't presume that mapping."""
    event_id: str
    commence_time: str
    home_team: str
    away_team: str
    player: str                       # raw display name from the book -- no player id
    market: str
    book: str                         # bookmaker key (e.g. "draftkings")
    line: float
    over_price: Optional[str] = None  # American odds, as a string ("-110", "+120")
    under_price: Optional[str] = None


class OddsProvider(ABC):
    """Player-prop odds for one sport. Every method returns [] rather than raising on a
    missing key, a network failure, or an unparseable response -- treat [] as "nothing
    posted right now", never as an error to surface to the board."""

    @abstractmethod
    def events(self) -> list[dict]:
        """Upcoming/live events: [{id, commence_time, home_team, away_team}, ...]."""

    @abstractmethod
    def player_props(self, event_id: str, markets: Sequence[str]) -> list[PlayerProp]:
        """Player-prop markets a real book actually posted for one event. Never synthesizes
        a market/player nobody priced -- an empty list here means "no book posted this",
        which the caller (cfb/lines.py) must treat as "don't show a prop", not as an error."""


def _key() -> Optional[str]:
    return odds_api_key()


def _get(path: str, params: Optional[dict] = None, ttl: float = 60.0) -> Optional[list | dict]:
    global _warned_no_key
    key = _key()
    if not key:
        if not _warned_no_key:
            print("[cfb.odds_provider] ODDS_API_KEY is not set -- CFB player props return "
                 "empty until it is. See DEPLOY.md's env var table.", flush=True)
            _warned_no_key = True
        return None
    p = dict(params or {})
    p["apiKey"] = key
    ck = (path, tuple(sorted((k, v) for k, v in p.items() if k != "apiKey")))
    now = time.time()
    hit = _memory.get(ck)
    if hit and now - hit[0] < ttl:
        return hit[1]
    url = f"{ODDS_API_BASE_URL}{path}"
    val = None
    try:
        resp = requests.get(url, params=p, timeout=HTTP_TIMEOUT)
        if resp.status_code != 200:
            print(f"[cfb.odds_provider] {path} -> HTTP {resp.status_code}: {resp.text[:200]}",
                 flush=True)
        else:
            val = resp.json()
    except Exception as exc:
        print(f"[cfb.odds_provider] {path} -> {exc}", flush=True)
    _memory[ck] = (now, val)
    return val


class TheOddsApiAdapter(OddsProvider):
    """Live The Odds API adapter for CFB (`americanfootball_ncaaf`) player props."""

    def __init__(self, sport_key: str = ODDS_API_SPORT_KEY, regions: str = "us"):
        self.sport_key = sport_key
        self.regions = regions

    def events(self) -> list[dict]:
        raw = _get(f"/sports/{self.sport_key}/events", ttl=CACHE_TTL)
        if not isinstance(raw, list):
            return []
        out = []
        for e in raw:
            if not isinstance(e, dict):
                continue
            eid = e.get("id")
            if not eid:
                continue
            out.append({"id": eid, "commence_time": e.get("commence_time"),
                       "home_team": e.get("home_team"), "away_team": e.get("away_team")})
        return out

    def player_props(self, event_id: str, markets: Sequence[str]) -> list[PlayerProp]:
        raw = _get(f"/sports/{self.sport_key}/events/{event_id}/odds",
                   {"regions": self.regions, "markets": ",".join(markets),
                    "oddsFormat": "american"}, ttl=90.0)
        if not isinstance(raw, dict):
            return []
        home, away = raw.get("home_team"), raw.get("away_team")
        commence = raw.get("commence_time")
        out: list[PlayerProp] = []
        for bm in raw.get("bookmakers") or []:
            if not isinstance(bm, dict):
                continue
            book = bm.get("key") or bm.get("title") or "unknown"
            for mk in bm.get("markets") or []:
                if not isinstance(mk, dict) or mk.get("key") not in markets:
                    continue
                by_player: dict[str, PlayerProp] = {}
                for oc in mk.get("outcomes") or []:
                    if not isinstance(oc, dict):
                        continue
                    player = oc.get("description")
                    point = oc.get("point")
                    side = (oc.get("name") or "").lower()
                    price = oc.get("price")
                    if not player or point is None or side not in ("over", "under"):
                        continue
                    prop = by_player.get(player)
                    if prop is None:
                        prop = PlayerProp(
                            event_id=event_id, commence_time=commence or "",
                            home_team=home or "", away_team=away or "", player=player,
                            market=mk["key"], book=book, line=float(point))
                        by_player[player] = prop
                    price_str = f"{price:+d}" if isinstance(price, int) else str(price)
                    if side == "over":
                        prop.over_price = price_str
                    else:
                        prop.under_price = price_str
                out.extend(by_player.values())
        return out
