"""
cfb — College Football (FBS, all 134 teams) inside JUICED: data layer + Monte Carlo engine.

  model/                 The fitting layer -- league priors and their empirical-Bayes
                         shrinkage strengths, team pace, opponent strength, the garbage-time
                         blowout discount, and the three-tier prior (returning / transfer /
                         true freshman) that blends them. Pure functions of already-fetched
                         dataclasses; no I/O, no database.
  sim/engine.py          The Monte Carlo run: projected team plays x usage share x efficiency,
                         with the two-stage parameter+outcome uncertainty layer every other
                         engine in this repo uses.
  projections.py         The engine's only I/O: fetch, cache, fit, project one player-game.
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
  board.py                attach_cfb(lines) -- writes model_proj/model_prob/... onto every CFB
                         line the engine can price, market-anchored on the MEDIAN. Registered
                         in analytics.attach_projections exactly like tennis/basketball/nfl.

Every CFB row logs the same prop_clv ledger schema every other sport does (sport='CFB'), pre-
anchor model_raw/model_raw_prob/trust_weight/model_version included from day one -- see
cfb/tests/test_ledger.py.

CFB IS STILL UNMEASURABLE, and shipping an engine did not change that: no CFBD_API_KEY has
existed in any environment this code has run in, so none of its fits has seen a live response,
and the ledger holds zero graded CFB rows. model_health / backtest report "insufficient_data"
honestly for any sport with zero graded rows (the same path MLB/WNBA/Tennis validated on day
one); CFB needs no special-casing there, and will keep reporting it until real games grade
under cfb-1.0.0. See cfb/README.md's "What's genuinely NOT verified".
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
