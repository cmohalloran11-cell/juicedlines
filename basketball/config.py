"""
Basketball config — the two league configs plug into one shared core here.

Everything the core needs to differ between WNBA and Summer League lives in
`leagues`: the prior each rate regresses to, how minutes are projected, the
pace baseline, and the variance width. Tune here without touching the model.
"""

from __future__ import annotations

CONFIG = {
    # ── shared core knobs ─────────────────────────────────────────────────────
    "model": {
        "n_sims": 10000,
        "recency_halflife": 6,        # games; exponential recency weight on rates
    },
    "confidence": {                   # effective (recency-weighted) games → flag
        "high": 8,
        "medium": 3,
    },
    # Opponent-DEFENSE adjustment (model/opponent.py). Wired end-to-end but DEFAULT OFF:
    # measured on a live slate it made calibration WORSE, not better —
    #   WNBA  baseline −0.475 → pace-only −0.294 → pace+def −0.323
    #   SL    baseline −0.008 → pace-only −0.003 → pace+def −0.353  (much worse)
    # A team def_rtg off ~6 games (2-3 in Summer League) is mostly noise, and ESPN gives no
    # cheap POSITIONAL defense — so it injects error instead of signal. Flip a league to true
    # only once a date-strict backtest shows it improves MAE (needs as-of-date def ratings,
    # which team_pace doesn't do yet — it reads CURRENT form). Matchup PACE stays on: it's
    # relative, mean-neutral, and measurably helped.
    "opp_def_adjust": False,

    # ── per-league configs (the override hooks) ───────────────────────────────
    "leagues": {
        "WNBA": {
            "espn_path": "basketball/wnba",
            "gamelog_mode": "athlete",    # ESPN populates WNBA season gamelogs
            "game_minutes": 40,       # WNBA plays 4×10
            "league_pace": 96.0,      # possessions/game — refined live from box scores
            "roster_depth": 10,       # tight rotations → starters play more
            # rate shrinkage: pseudo-possessions of the positional prior. Low, so a
            # healthy WNBA sample dominates the rough positional prior → tight rates.
            "shrink_poss": 120,
            # minutes: the player's own recency-weighted minutes (short half-life →
            # tracks current role). No pull to a global midpoint for players with a
            # sample; the midpoint only applies to zero-game players (via the fallback).
            "minutes_shrink_games": 0,
            "prior": "positional",    # regress rates toward WNBA positional averages
            # variance widths (fraction of the projected mean)
            "min_sd_frac": 0.13,      # minutes are fairly stable in WNBA
            "pace_sd_frac": 0.05,
            "disp": 0.10,             # per-stat overdispersion (NegBin); small = tight
        },
        "NBA Summer League": {
            "espn_path": "basketball/nba-summer-las-vegas",
            "gamelog_mode": "boxscore",   # ESPN has no SL gamelog → derive from box scores
            "game_minutes": 40,
            "league_pace": 102.0,     # SL runs faster and much more variable
            "roster_depth": 12,
            # LIGHT shrinkage — the actual SL games are the best (only) signal we have, so
            # lean on them: a player's real SL production drives the projection. Small
            # samples stay noisy (kept low confidence), but the projection is real, not a
            # line mirror. (Without a draft/college prior feed, heavy shrinkage just erased
            # the only usable data.) At 70 the slate skewed systematically UNDER — projections
            # sat below both the market and the players' own SL averages (over-regression to a
            # modest prior). 30 centers it on actual production (slate mean edge −0.65 → −0.3,
            # under/over 59/30 → ~50/37) while keeping healthy regression on 2-game samples.
            # 30→25 once the actual SL games proved predictive (well-calibrated night): lean a
            # bit harder on real production now that most players have 2-3 games logged.
            "shrink_poss": 25,
            # minutes: use the player's actual SL minutes (like WNBA); the draft-slot
            # baseline only applies to players with zero games so far.
            "minutes_shrink_games": 0,
            "prior": "translated",    # draft slot + archetype + translated pre-NBA rates
            # wide everything — post fewer markets, later, gated by confidence
            "min_sd_frac": 0.32,      # coaches distribute minutes almost arbitrarily
            "pace_sd_frac": 0.13,
            "disp": 0.22,
        },
    },

    "keys": {},
}


def cfg(*path, default=None):
    d = CONFIG
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def league_cfg(league: str) -> dict:
    return CONFIG["leagues"].get(league, {})
