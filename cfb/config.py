"""
cfb.config — environment + constants for the CFB data layer.

CFBD_API_KEY / ODDS_API_KEY are read here once so every module in cfb/ shares one source of
truth. Unset means "that data source is disabled" -- an honest empty/unavailable result, never
a crash and never fabricated data (see cfb/data/cfbd_client.py and cfb/data/odds_provider.py's
own module docstrings for the exact degrade-to-empty contract, matching
basketball/data/balldontlie.py's established BALLDONTLIE_API_KEY pattern).

LICENSE CONSTRAINT: CFBD_API_KEY is server-side only. Nothing in this module, or anything that
imports it, may place the key (or a raw CFBD response) somewhere the browser can read it --
not in a static/dashboard.html response, not in build_static.py's output JSON. See
cfb/README.md's "License constraints" section.
"""
from __future__ import annotations

import os

CFBD_BASE_URL = os.getenv("CFBD_BASE_URL", "https://api.collegefootballdata.com")
ODDS_API_BASE_URL = os.getenv("ODDS_API_BASE_URL", "https://api.the-odds-api.com/v4")

# The Odds API's own sport key for FBS college football (documented at
# https://the-odds-api.com/sports-odds-data/sports-apis.html -- unverified against a live
# response in this session, see odds_provider.py's module docstring).
ODDS_API_SPORT_KEY = "americanfootball_ncaaf"

FBS_CLASSIFICATION = "fbs"     # CFBD's own team classification value, used to filter teams()
# 2025 realignment count. Informational only (logged, never enforced as a hard cap) --
# conference realignment moves this most offseasons, and teams() always reflects whatever
# CFBD actually returns for `classification`, not this constant.
FBS_TEAM_COUNT = 134

HTTP_TIMEOUT = 20
CACHE_TTL = 900.0   # 15 min in-process memo TTL, matches balldontlie.py's _memory pattern


def cfbd_api_key() -> str | None:
    return os.getenv("CFBD_API_KEY") or None


def odds_api_key() -> str | None:
    return os.getenv("ODDS_API_KEY") or None


# Player-prop markets pulled from the Odds API (task spec; also the only CFB player-prop
# markets the-odds-api documents as of 2026-08). Maps each market key to the canonical
# stat_type label the rest of the board already understands the SHAPE of (one float line +
# a two-sided price) -- what to DO with that shape (project it) is the modeling agent's job,
# not this data layer's; see cfb/board.py's attach_cfb extension point.
ODDS_MARKET_TO_STAT: dict[str, str] = {
    "player_pass_yds": "Passing Yards",
    "player_rush_yds": "Rushing Yards",
    "player_reception_yds": "Receiving Yards",
    "player_receptions": "Receptions",
    "player_anytime_td": "Anytime TD",
}
