"""
Tennis config — data-source selection + model weights (tune without touching code).

`sources` picks each adapter: `sackmann` (historical mirror), `espn` (live),
`sample`, or `licensed` (production feed you wire later). ATP and WTA weights /
baselines are kept SEPARATE downstream — this only holds shared knobs + defaults.
"""

from __future__ import annotations

import os

CONFIG = {
    "sources": {
        "history": "sackmann",     # sackmann | licensed
        "upcoming": "espn",        # espn | licensed | sample
        "rankings": "sackmann",    # sackmann | espn | licensed
        "odds": "sample",          # sample | oddsapi | licensed
    },
    # Sackmann-format history mirrors (CC non-commercial → build/validate only).
    "repos": {
        "ATP": "stakah/tennis_atp",
        "WTA": "stakah/tennis_wta",
    },
    "keys": {
        "odds_api": os.getenv("ODDS_API_KEY", ""),
        "licensed": os.getenv("TENNIS_FEED_KEY", ""),
    },
    "model": {
        # serve/return point model
        "p_serve_clamp": [0.50, 0.85],
        # cold-start shrinkage: regress rates toward the tour/tier prior using
        # pseudo-counts (higher = trust the sample less when thin).
        "shrink_serve_pts": 800,
        "shrink_return_pts": 800,
        "min_matches_full_confidence": 25,
        # surface fallback: blend a player's surface rate with overall+shift when
        # the surface sample is thin.
        "shrink_surface_matches": 12,
        # Elo backbone
        "elo_k": 32,
        "elo_surface_weight": 0.5,   # 0.5*overall + 0.5*surface
        "elo_start": 1500,
        "match_prob_blend": 0.5,     # blend point-model win prob with Elo win prob
        # simulation
        "n_sims": 10000,
        # default final-set tiebreak point (per-match override wins)
        "final_set_tb_at": 6,        # 6-6; None would mean advantage set
    },
    "confidence": {                  # effective-sample-size → flag
        "high": 25,                  # matches (on surface, roughly)
        "medium": 8,
    },
}


def cfg(*path, default=None):
    d = CONFIG
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d
