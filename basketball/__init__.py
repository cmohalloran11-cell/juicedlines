"""
Basketball projection system — WNBA + NBA Summer League.

One shared projection CORE (per-possession rates × projected minutes × projected
pace, opponent-adjusted → a simulated distribution per stat), plugged in as two
separate league CONFIGS. WNBA is a re-baselining job on stable, well-sampled data
(tight intervals). Summer League leans on translated pre-NBA priors + explicit
minutes modelling because players have little/no usable history (wide intervals,
low confidence by design).

Layout mirrors the tennis package:
  data/        swappable source adapters (ESPN live) behind ABC interfaces
  model/       rates+shrinkage, pace, minutes, opponent, priors (the league hooks)
  sim/         Monte-Carlo engine → per-stat distributions, O/U prob, variance
  projections  public API (fit/cache per league, project a player-game, read markets)
  board        attach model_proj/prob/confidence to live PrizePicks/Underdog lines
  backtest/    date-strict calibration, per league + SL translation-bias check
  value/       odds → implied prob, edge/EV
"""

from __future__ import annotations

LEAGUES = ("WNBA", "NBA Summer League")

# Canonical base box stats the core projects (everything else is a combo of these).
BASE_STATS = ("pts", "reb", "ast", "stl", "blk", "3pm", "to")

# Combo markets = sums of base stats (simulated jointly so they stay correlated).
COMBOS = {
    "pra": ("pts", "reb", "ast"),
    "pr": ("pts", "reb"),
    "pa": ("pts", "ast"),
    "ra": ("reb", "ast"),
    "stocks": ("stl", "blk"),
}
