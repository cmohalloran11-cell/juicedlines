"""
Basketball projection system — WNBA.

One shared projection CORE (per-possession rates × projected minutes × projected
pace, opponent-adjusted → a simulated distribution per stat), driven by a league
CONFIG. WNBA is a re-baselining job on stable, well-sampled data (tight intervals).

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

LEAGUES = ("WNBA",)

# Canonical base box stats the core projects (everything else is a combo of these).
BASE_STATS = ("pts", "reb", "ast", "stl", "blk", "3pm", "to")

# Offensive/Defensive rebounds are DERIVED from `reb`, not fitted as their own rates.
# Why: ESPN's athlete gamelog (the WNBA source) only exposes totalRebounds — the OREB/DREB
# split lives in box scores, which cover far fewer games (Reese: 25 gamelog vs 8 box). Fitting
# separate orb/drb rates off 8 games would be noise, and switching the whole league to box
# scores would gut the sample for every other stat.
# Instead: total rebounds are well-estimated (25 games) and a player's OFFENSIVE SHARE is a
# stable skill that 8 games estimates fine. So the sim splits each simulated rebound
# binomially at that share — which keeps orb+drb == reb exactly (as reality does), and gives
# the right variance and the right correlation with reb for free.
DERIVED_STATS = ("orb", "drb")

# Combo markets = sums of base stats (simulated jointly so they stay correlated).
COMBOS = {
    "pra": ("pts", "reb", "ast"),
    "pr": ("pts", "reb"),
    "pa": ("pts", "ast"),
    "ra": ("reb", "ast"),
    "stocks": ("stl", "blk"),
}
