"""
cfb.board — the extension point the modeling agent's real CFB projection math plugs into.

Same contract shape as tennis/board.py::attach_tennis, basketball/board.py::attach_basketball,
nfl/board.py::attach_nfl: take the board's live line dicts, filter to this sport, project
each player, and write model_proj / model_prob / model_edge / model_floor / model_ceiling /
model_n / proj_kind (+ model_raw / model_raw_prob / trust_weight / model_version once an
anchor exists) onto every line this sport can price. Called from analytics.attach_projections
exactly like the other three sports, so both deploy paths (main.py's live loop,
build_static.py) pick it up automatically -- no further wiring needed once the modeling math
below exists.

THIS FUNCTION IS CURRENTLY A NO-OP STUB (Phase 4, part A -- data/plumbing only; see this
package's own docstring). It does not write model_proj, so CFB lines currently reach the
board unprojected (same visible state as any sport whose engine failed for a day -- the
board still shows the line, just with no pick/EV/Juice Score attached; see
valuation.py/dashboard.py's existing handling of a None model_proj). It is deliberately wired
into analytics.py now rather than left unregistered, so the modeling agent's real
implementation is a body-only change with zero additional dispatch/registration work --
exactly the mistake CLAUDE.md's testing-requirements section calls out (WNBA/tennis grading
existed fully implemented and simply was never invoked for a long time; a stub that's
correctly wired in advance can't repeat that).

EXTENSION POINTS for the modeling agent (garbage-time blowout-probability layer, plays-per-
game x usage-share x efficiency pace model, 3-tier prior fallback exposing which tier was
used, opponent-adjusted down-weighting vs FCS/bottom-quartile opponents):
  * cfb/data/cfbd_client.py::CFBDClient.team_efficiency -- per-game offense/defense PPA +
    success rate, the opponent-adjustment input.
  * cfb/data/cfbd_client.py::CFBDClient.schedule -- market spread/over_under per game (CFBD's
    /lines, aggregated real books), the game-environment/pace input -- same role
    nfl/model/environment.py's spread_line/total_line play for NFL.
    (Odds API events don't carry spread/total -- CFBD's /lines is the source for those, not
    the-odds-api adapter here, which only carries player-prop markets.)
  * cfb/player_status.py::is_stale -- flag a projection when the manual status override is
    missing or stale near kickoff (garbage-time/injury signal proxy until a real feed exists).
  * cfb/config.py::ODDS_MARKET_TO_STAT -- the only markets a line can exist for; adding a
    combo market (e.g. rush_rec_yards, mirroring nfl/__init__.py's COMBOS) is a config change
    here plus this file's simulate step, not a new data-layer module.
  A proj_kind convention analogous to NFL's "nfl_regular"/"nfl_preseason" (e.g.
  "cfb_prior_a"/"cfb_prior_b"/"cfb_prior_c" for the 3-tier fallback) is recommended so a
  calibration query can distinguish which tier actually priced a given graded row -- see
  db.stat_gammas's proj_kind scoping and provenance.MODEL_VERSIONS's "deliberate, hand-bumped"
  rule before shipping the first real cfb-*.*.* version.
"""
from __future__ import annotations

_SPORTS = ("CFB",)


def attach_cfb(lines: list[dict]) -> None:
    """No-op today (see module docstring) -- iterates CFB lines only so the shape of the
    real implementation (filter -> resolve -> project -> write fields) is already in place."""
    for l in lines:
        if l.get("sport") not in _SPORTS:
            continue
        # Modeling agent: compute model_proj/model_prob/... here and assign onto `l`,
        # following the market-anchoring pattern documented in nfl/board.py's module
        # docstring (blend/shift on the MEDIAN, never the mean -- see
        # provenance.MODEL_VERSIONS's NFL/Tennis/WNBA changelog entries for why).
        continue
