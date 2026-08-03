"""
Basketball config — the two league configs plug into one shared core here.

Everything the core needs to know about a league lives in
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
    # A team def_rtg off ~6 games is mostly noise, and ESPN gives no
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
            # tracks current role), regressed toward the role baseline with pseudo-count
            # k=1.5. 2026-08 fix: this was 0 -- meaning a player's FIRST game (however
            # fluky -- extended overtime, a blowout benching) became their entire minutes
            # projection with zero smoothing, only a literal zero-game player ever touched
            # the baseline fallback. Since the shrink weight is k/(effective_games+k), k=1.5
            # gives real protection for a 1-3 game sample (e.g. ~60% pull toward baseline at
            # n=1) while barely touching a player with an established, stable role (the
            # recency half-life below caps effective sample size around ~4 games regardless
            # of career length, so even a full-season veteran keeps a modest, non-zero pull
            # toward the role baseline -- a real but smaller residual gap than a debut game
            # getting none at all).
            "minutes_shrink_games": 1.5,
            "prior": "positional",    # regress rates toward WNBA positional averages
            # variance widths (fraction of the projected mean)
            "min_sd_frac": 0.13,      # minutes are fairly stable in WNBA
            "pace_sd_frac": 0.05,
            "disp": 0.10,             # per-stat overdispersion (NegBin); small = tight
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
