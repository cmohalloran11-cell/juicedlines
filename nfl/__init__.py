"""
NFL projection system — regular season AND preseason.

Two DISTINCT models sharing one data layer and one simulation engine:

  regular season  usage.py fits per-snap opportunity + per-attempt efficiency from the
                  player's own recent games (recency-weighted, empirical-Bayes shrunk toward
                  a positional prior measured from the whole league), playing_time.py
                  projects the snap share, environment.py sizes the game (plays, pass rate)
                  off the market's own spread/total, matchup.py scales efficiency by the
                  opponent's measured defense.

  preseason       rotation.py replaces playing_time.py entirely. A preseason snap share is
                  NOT a regular-season snap share with a knob turned down — it is a coaching
                  decision about rotation tiers, so players are classified into tiers
                  (confirmed starter … fringe/unknown) and each tier carries its own
                  playing-time distribution SHAPE. No free data source publishes preseason
                  snap counts (see config.PRESEASON_TIER_SNAP_SHARE), so those shapes are
                  labelled assumptions and preseason confidence is structurally low, which
                  makes board.py lean much harder on the market line for the LEVEL while the
                  model supplies the SPREAD.

Layout mirrors basketball/:
  data/         swappable source adapters (nflverse releases) behind an ABC
  model/        usage, playing time, preseason rotation, matchup, environment, combo corr
  sim/          Monte-Carlo engine -> per-market distributions
  projections   public API (fit/cache per player, project a player-game, read markets)
  board         attach_nfl(lines) — writes model_proj/prob/edge/floor/ceiling onto live lines
  analytics     analyze(line) — the real drivers behind one projection
  confidence    the supplementary nfl_confidence blend (display only)

Every constant in config.py that could have been guessed was instead MEASURED from real
nflverse data; each one records its own sample size and window. The handful that genuinely
cannot be measured from any free source are marked "basis": "assumed" and say why.
"""

from __future__ import annotations

LEAGUES = ("NFL",)

POSITIONS = ("QB", "RB", "WR", "TE")

# Canonical market keys the engine simulates. Rushing/receiving TOUCHDOWN props are
# deliberately absent: the TD probability model isn't proven reliable yet (product decision),
# and projections.py's market resolver returns None for them so they're skipped rather than
# mis-mapped onto a modelled stat.
BASE_MARKETS = (
    "pass_yards", "pass_attempts", "completions", "pass_tds",
    "rush_yards", "rush_attempts",
    "receptions", "rec_yards", "targets",
)

# Combo markets = sums of base markets, simulated jointly so they stay correlated
# (see model/combo_corr.py — the correlation is measured, not assumed).
COMBOS = {
    "rush_rec_yards": ("rush_yards", "rec_yards"),
}

MARKETS = BASE_MARKETS + tuple(COMBOS)

SEASON_TYPES = ("regular", "preseason")
