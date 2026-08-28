"""
cfb — College Football (FBS, all 134 teams) data + plumbing layer inside JUICED.

This package is DATA/PLUMBING ONLY (Phase 4, part A). It does not simulate anything -- there
is no cfb/model/ or cfb/sim/ yet, unlike basketball/nfl/tennis. What it provides for the
modeling agent to build on top of:

  data/cfbd_client.py   CFBD REST adapter (teams, rosters, schedule + market spread/total,
                         per-player box scores, per-team advanced efficiency). Server-side
                         only -- CFBD_API_KEY never reaches a browser or a static-build output.
  data/odds_provider.py Swappable OddsProvider interface + a The Odds API adapter for player
                         props (CFBD does not carry player props). A prop is only ever
                         surfaced if a real book actually posted that market -- see
                         cfb/lines.py.
  schema.py/repositories.py  Canonical player table (id, name, team, position) + a
                         source-id mapping layer (CFBD athlete id, Odds API player name) with
                         a fuzzy-match fallback and a human review log for anything unresolved
                         -- same shape as fantasy/'s player_matching, reused rather than
                         reinvented (see player_matching.py).
  player_status.py/PlayerStatusRepository  Manual, admin-editable status override table with
                         timestamps -- no CFBD/Odds-API injury report is mandated to exist.
  lines.py               Turns Odds-API player props into board Line-dicts, gated strictly on
                         a market actually being posted, and registers into books.REGISTRY so
                         both deploy paths (main.py's live loop, build_static.py) pick it up
                         with zero further wiring.
  board.py                attach_cfb(lines) -- the extension point the modeling agent's real
                         projection math (garbage-time model, pace model, 3-tier prior
                         fallback, opponent adjustment) plugs into. Registered in
                         analytics.attach_projections exactly like tennis/basketball/nfl.

Every CFB row logs the same prop_clv ledger schema every other sport does (sport='CFB'), pre-
anchor model_raw/model_raw_prob/trust_weight/model_version included from day one -- see
cfb/tests/test_ledger.py.

New sport, no math shipped yet -> CFB starts UNMEASURABLE by definition. model_health /
backtest already report "insufficient_data" honestly for any sport with zero graded rows
(the same path MLB/WNBA/Tennis validated on day one); CFB needs no special-casing there.
"""
from __future__ import annotations

LEAGUES = ("CFB",)

POSITIONS = ("QB", "RB", "WR", "TE")

# Canonical market keys this data layer hands the modeling agent a real, book-posted line
# for (see config.ODDS_MARKET_TO_STAT). No combo markets yet -- rush_rec_yards etc. are the
# modeling agent's call once the base markets are validated individually.
from .config import ODDS_MARKET_TO_STAT  # noqa: E402

MARKETS = tuple(ODDS_MARKET_TO_STAT.values())

SEASON_TYPES = ("regular", "postseason")

from .schema import init_schema  # noqa: E402

__all__ = ["LEAGUES", "POSITIONS", "MARKETS", "SEASON_TYPES", "init_schema"]
