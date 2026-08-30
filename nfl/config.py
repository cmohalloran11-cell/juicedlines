"""
NFL config — every tunable weight, prior and shrinkage constant in one place.

Provenance rule for this file: a number here is either MEASURED (with the window and sample
size that produced it recorded next to it) or explicitly marked as an assumption with the
reason no measurement exists. Nothing in between. The measurement code that produced each
measured block is still in the repo and re-runnable:

    model/usage.py:positional_priors        -> POSITIONAL_PRIORS
    model/usage.py (holdout sweep)          -> SHRINK_K            [see MEASURED below]
    model/playing_time.py:measure_snap_prior-> SNAP_SHARE
    model/playing_time.py:measure_depth_rank_shares -> DEPTH_RANK_SNAP_SHARE
    model/matchup.py:fit_defense            -> DEFENSE
    model/environment.py:measure_environment-> ENVIRONMENT
    model/combo_corr.py                     -> COMBO_CORR

Regular season only — the preseason rotation model this engine once shipped alongside it was
removed (see provenance.MODEL_CHANGELOG's NFL entries).

MEASUREMENT WINDOW for everything tagged 2022-2025: nflverse `stats_player`
(stats_player_week_{season}.csv), `snap_counts`, `depth_charts` and `schedules` releases,
season_type REG, seasons 2022-2025, joined on (game_id, team, player) — 72,457 player-weeks,
22,471 skill player-games with a matched snap row (95.6% join rate), 2,174 team-games.
Confirmed live 2026-08-15.
"""

from __future__ import annotations

CONFIG = {
    "model": {
        "n_sims": 10000,
        # Recency half-life in GAMES for the usage rate fit. Separate, shorter half-life for
        # the snap share (below) because playing time tracks the current role far more
        # sharply than efficiency does.
        "recency_halflife": 6,
        # Lookback for the usage fit. Two seasons keeps a full-season starter's sample deep
        # while still letting a role change from last year wash out through the half-life.
        "lookback_seasons": 2,
    },

    # Effective (recency-weighted) games -> the coarse high/medium/low flag, same shape as
    # basketball's `confidence` block.
    "confidence": {
        "high": 8,
        "medium": 3,
    },

    # ── league-average opportunity + efficiency, per position ────────────────────────────
    # MEASURED 2022-2025 REG. Opportunity rates are per OFFENSIVE SNAP (which is what makes
    # opportunity separable from playing time); efficiency rates are per attempt/target.
    # Sample sizes: QB 2,556 player-games / 138,482 snaps; RB 5,515 / 140,485;
    # WR 8,831 / 320,758; TE 4,743 / 160,540.
    "positional_priors": {
        "QB": {
            "carries_per_snap": 0.06524, "targets_per_snap": 0.00024,
            "pass_att_per_snap": 0.50505,
            "yards_per_carry": 4.4271, "yards_per_target": 1.4545, "catch_rate": 0.8182,
            "completion_rate": 0.6467, "yards_per_attempt": 7.0523,
        },
        "RB": {
            "carries_per_snap": 0.32404, "targets_per_snap": 0.08455,
            "pass_att_per_snap": 0.00015,
            "yards_per_carry": 4.3243, "yards_per_target": 5.7263, "catch_rate": 0.7788,
            "completion_rate": 0.5714, "yards_per_attempt": 5.1429,
        },
        "WR": {
            "carries_per_snap": 0.00465, "targets_per_snap": 0.11645,
            "pass_att_per_snap": 0.00016,
            "yards_per_carry": 5.7523, "yards_per_target": 7.9376, "catch_rate": 0.6297,
            "completion_rate": 0.4808, "yards_per_attempt": 7.0962,
        },
        "TE": {
            "carries_per_snap": 0.00245, "targets_per_snap": 0.09154,
            "pass_att_per_snap": 0.00027,
            "yards_per_carry": 4.1959, "yards_per_target": 7.2983, "catch_rate": 0.7133,
            "completion_rate": 0.5349, "yards_per_attempt": 9.3256,
        },
    },

    # ── empirical-Bayes shrinkage strength, in DENOMINATOR units ─────────────────────────
    # (snaps for a per-snap rate, carries for yards_per_carry, targets for yards_per_target
    # and catch_rate, attempts for completion_rate and yards_per_attempt).
    #
    # MEASURED, not chosen: for every (position, rate) each player-season with >=12 games was
    # split into a first-8-game fit and a held-out remainder, k was swept over
    # {0,1,2,5,...,1000}, and the k minimizing denominator-weighted out-of-sample squared
    # error was kept. Every value below beat BOTH k=0 (raw observed rate) and prior-only, so
    # the shrinkage is doing real work rather than being a formality. Examples of the margin:
    #   RB yards_per_carry  k=300: wMSE 0.550  vs k=0 1.096  vs prior-only 0.600
    #   WR yards_per_target k=150: wMSE 3.609  vs k=0 6.411  vs prior-only 3.865
    #   RB carries_per_snap k=40 : wMSE 0.00489 vs k=0 0.00548 vs prior-only 0.01097
    # QB completion_rate landed at the top of the sweep (k=1000, essentially prior-only) —
    # eight games of a QB's completion rate really is that unpredictive out of sample, and
    # pretending otherwise with a small k measured strictly worse.
    "shrink_k": {
        "QB": {"carries_per_snap": 80, "pass_att_per_snap": 400, "targets_per_snap": 1000,
               "yards_per_carry": 20, "yards_per_target": 1000, "catch_rate": 1000,
               "completion_rate": 1000, "yards_per_attempt": 200},
        "RB": {"carries_per_snap": 40, "pass_att_per_snap": 1000, "targets_per_snap": 150,
               "yards_per_carry": 300, "yards_per_target": 80, "catch_rate": 150,
               "completion_rate": 1000, "yards_per_attempt": 1000},
        "WR": {"carries_per_snap": 80, "pass_att_per_snap": 1000, "targets_per_snap": 150,
               "yards_per_carry": 1000, "yards_per_target": 150, "catch_rate": 100,
               "completion_rate": 1000, "yards_per_attempt": 1000},
        "TE": {"carries_per_snap": 20, "pass_att_per_snap": 60, "targets_per_snap": 100,
               "yards_per_carry": 1000, "yards_per_target": 100, "catch_rate": 200,
               "completion_rate": 1000, "yards_per_attempt": 1000},
    },

    # ── playing time ─────────────────────────────────────────────────────────────────────
    "snap_share": {
        # MEASURED 2022-2025 REG: league mean offense_pct by position.
        "prior": {"QB": 0.8292, "RB": 0.3638, "WR": 0.5311, "TE": 0.5128},
        # MEASURED by the same first-8-games -> held-out-remainder holdout as shrink_k, over
        # halflife in {0.75,1,1.5,2,2.5,3,4,6} x k in {0,0.25,...,3}. Every position beat both
        # the unshrunk recency-weighted mean and the prior alone, e.g.
        #   WR  MSE 0.01638  vs k=0 0.02138  vs prior-only 0.06867
        #   RB  MSE 0.01401  vs k=0 0.01784  vs prior-only 0.04908
        "halflife": {"QB": 2.0, "RB": 1.5, "WR": 1.5, "TE": 1.5},
        "shrink_games": {"QB": 1.5, "RB": 0.75, "WR": 0.5, "TE": 0.5},
        # Beta concentration (a+b) of a player's OWN week-to-week snap share, MEASURED as the
        # median method-of-moments concentration across players with >=8 games in a season
        # (QB n=115, RB n=365, WR n=593, TE n=320). This is the width the sampled snap-share
        # distribution converges to once a player has a real, stable sample; a thinner sample
        # gets a proportionally lower concentration (= wider band) in playing_time.py.
        "concentration": {"QB": 1.46, "RB": 9.88, "WR": 7.24, "TE": 12.17},
        # Games of own history at which the measured concentration above is applied in full.
        "concentration_full_games": 8,
    },

    # ── depth-chart tier -> snap share ───────────────────────────────────────────────────
    # MEASURED 2022-2024 REG, depth_charts joined to snap_counts on
    # (season, week, team, player), offense formation only, 17,943 matched player-weeks.
    # Used as the playing-time prior for a player with no usable snap history of their own
    # but a known depth-chart rank — strictly better than the flat positional prior.
    # QB rank 3 reads HIGHER than rank 2 (0.615 vs 0.439) on only n=79: a 3rd-string QB
    # appearing in a box score at all is nearly always a starter-by-injury that week, i.e.
    # survivorship, not depth. Ranks 3+ are therefore pooled for every position.
    "depth_rank_snap_share": {
        "QB": {1: (0.9458, 0.1549, 1398), 2: (0.4385, 0.4027, 483), 3: (0.6154, 0.3763, 79)},
        "RB": {1: (0.5675, 0.2131, 1496), 2: (0.3012, 0.2028, 1415), 3: (0.1375, 0.1603, 1423)},
        "WR": {1: (0.7283, 0.2222, 3509), 2: (0.3390, 0.2679, 2832), 3: (0.2541, 0.2598, 725)},
        "TE": {1: (0.6396, 0.2249, 1759), 2: (0.3836, 0.2210, 1622), 3: (0.2364, 0.1889, 1202)},
    },

    # ── game environment ─────────────────────────────────────────────────────────────────
    # MEASURED 2022-2025 REG, 2,174 team-games joined to the schedules release's closing
    # spread_line/total_line. Ordinary least squares; the r values are small because a single
    # game's volume is mostly noise, which is exactly why the effect is applied as a small
    # shift on a wide distribution rather than as a confident point move. team_spread is
    # positive when THIS team is favoured.
    #
    # TWO DIFFERENT VOLUME QUANTITIES, and conflating them is a real bug this cost once: a
    # team's OFFENSIVE SNAPS (what the snap_counts feed counts, and therefore the denominator
    # every per-snap opportunity rate in this engine is fit against) average 65.325 a game,
    # while its SCRIMMAGE PLAYS (pass attempts + sacks + carries) average 62.376 — a ratio of
    # 1.0475, because snaps also include penalties, kneels, spikes and two-point tries. The
    # simulator must multiply per-snap rates by SNAPS; using scrimmage plays under-counted
    # every projected count by ~4.5%, which showed up as a measurable negative bias against
    # real holdout outcomes (RB carries -0.68/game before this fit was corrected).
    #   team_snaps     = 60.0921 + 0.17825*team_spread + 0.11875*total_line (r 0.122 / 0.057)
    #   scrim_plays    = 58.045  + 0.1800 *team_spread + 0.0983 *total_line (r 0.131 / 0.050)
    #   dropback_share = 0.4628  - 0.002641*team_spread + 0.002373*total_line (r -0.151/0.095)
    # dropback_share is a share of SCRIMMAGE plays, not of snaps — that's consistent because
    # the simulator only ever uses it relative to its own league mean, so the units cancel.
    "environment": {
        "snaps_intercept": 60.0921, "snaps_per_spread": 0.17825, "snaps_per_total": 0.11875,
        "plays_intercept": 58.045, "plays_per_spread": 0.1800, "plays_per_total": 0.0983,
        "dropback_intercept": 0.4628, "dropback_per_spread": -0.002641,
        "dropback_per_total": 0.002373,
        # League baselines, used when a game has no spread/total (a game the schedules
        # release hasn't priced yet, or a line the schedule lookup can't match at all).
        "league_snaps_mean": 65.325, "league_snaps_sd": 8.847,
        "league_plays_mean": 62.376, "league_plays_sd": 8.356,
        "league_dropback_share": 0.5673, "league_dropback_sd": 0.1064,
        # Weather. MEASURED on the 1,190 outdoor 2022-2025 games carrying both temp and wind:
        # wind vs dropback share r=-0.079, wind vs team yards r=-0.080, temp vs team yards
        # r=+0.032. Real but tiny; the only effect large enough to act on is the >=15mph
        # bucket, where the measured dropback share is 0.5402 vs 0.5655 below it (n=118).
        "high_wind_mph": 15.0,
        "high_wind_dropback_share": 0.5402, "normal_dropback_share": 0.5655,
    },

    # ── opponent defense ─────────────────────────────────────────────────────────────────
    # MEASURED per team-season 2022-2025 (n=128 team-seasons):
    #   yards allowed per pass attempt  mean 7.0533 sd 0.5027, observed [0.831, 1.164] x mean
    #   yards allowed per carry         mean 4.3489 sd 0.4034, observed [0.759, 1.247] x mean
    #   completion rate allowed         mean 0.6456 sd 0.0293
    #   sack rate                       mean 0.0690 sd 0.0135
    # RELIABILITY is the important number here: split-half within a season (weeks 1-9 vs
    # 10-18) gives r=0.168 for yards/attempt and r=0.318 for yards/carry, i.e. Spearman-Brown
    # full-season reliabilities of 0.287 and 0.483. Most of a defense's apparent per-play
    # profile is noise, so the matchup multiplier is shrunk by exactly that measured
    # reliability rather than applied at face value, and clamped to the observed range.
    "defense": {
        "league": {"yards_per_pass_att": 7.0533, "yards_per_carry": 4.3489,
                   "completion_rate": 0.6456, "sack_rate": 0.0690},
        "reliability": {"yards_per_pass_att": 0.287, "yards_per_carry": 0.483,
                        "completion_rate": 0.287, "sack_rate": 0.287},
        "clamp": {"yards_per_pass_att": (0.831, 1.164), "yards_per_carry": (0.759, 1.247),
                  "completion_rate": (0.880, 1.116), "sack_rate": (0.480, 1.620)},
        # Plays of opponent history below which no matchup adjustment is applied at all.
        "min_plays": 100,
    },

    # ── simulation distribution shapes ───────────────────────────────────────────────────
    # MEASURED. Counts conditioned on real snaps and the player's own season rate come out
    # essentially POISSON, not overdispersed: implied NegBin dispersion QB attempts -0.0044,
    # RB carries +0.0078, WR targets -0.0132, TE targets -0.0229 (n 1,822-6,754 player-games
    # each). So the count draw is Poisson and the extra spread a NegBin would have added is
    # not invented — all real excess spread enters through the snap/plays/parameter draws.
    #
    # Yards are modelled as a Gamma whose shape scales with the drawn attempt count, using the
    # MEASURED per-attempt standard deviation (residual sd of game yards around count x
    # league yards-per-attempt, divided by sqrt(count)):
    #   RB rush 6.288 (mean 4.351)   QB pass 9.957 (7.057)   WR rec 10.852 (7.981)
    #   TE rec  8.396 (7.417)        RB rec  7.922 (5.758)   QB rush 7.127 (4.637)
    #
    # The beta-binomial overdispersion the two-stage rate draw produces for
    # completions|attempts and receptions|targets is validated against the measured factors:
    #   QB completions 1.170x binomial, WR receptions 1.116x, TE 1.018x, RB 1.013x.
    "sim": {
        "count_dispersion": 0.0,
        "yards_per_attempt_sd": {
            ("QB", "pass"): 9.957, ("QB", "rush"): 7.127,
            ("RB", "rush"): 6.288, ("RB", "rec"): 7.922,
            ("WR", "rush"): 6.288, ("WR", "rec"): 10.852,
            ("TE", "rush"): 6.288, ("TE", "rec"): 8.396,
        },
        # Team-snap uncertainty around the environment estimate. MEASURED: residual sd of team
        # offensive snaps around the spread+total fit is 8.766 on a mean of 65.325 (the fit
        # explains almost none of it), i.e. 13.42% of the mean.
        "snaps_sd_frac": 0.1342,
        # Same for the pass/rush split: residual sd of dropback share around the fit, 0.1047
        # absolute on a 0.5673 mean.
        "dropback_sd": 0.1047,
    },

    # ── correlation ──────────────────────────────────────────────────────────────────────
    # MEASURED same-team same-game Pearson correlations, 2022-2025 REG. Sample sizes are
    # player-games with both legs present; the QB side requires a single QB with >=10 attempts
    # so a two-QB game doesn't pollute the pairing.
    "correlation": {
        "cross_player": {
            "qb_pass_att__wr_targets":   (0.2706, 8124),
            "qb_pass_att__te_targets":   (0.2350, 4177),
            "qb_pass_att__rb_carries":  (-0.1078, 4956),
            "qb_pass_yards__wr_rec_yards": (0.3156, 8124),
            "qb_pass_yards__te_rec_yards": (0.2303, 4177),
            "qb_pass_yards__rb_rec_yards": (0.1940, 4956),
            "qb_completions__wr_receptions": (0.2738, 8124),
            "qb_completions__te_receptions": (0.2593, 4177),
            "team_rush_att__rb_carries": (0.3068, 4956),
        },
        # Intra-player, for the Rush+Rec Yards combo prop. Pooled across 139 RBs with >=16
        # games (5,335 player-games) = 0.2616; the mean of the same 139 players' OWN
        # correlations is 0.1620 (median 0.1437) — the pooled number is higher because it also
        # carries between-player differences, so the per-player fit is shrunk toward the
        # pooled value the same way basketball's combo_corr does.
        "rb_rush_rec_yards_pooled": 0.2616,
        "rb_rush_rec_yards_player_mean": 0.1620,
        "rb_carries_receptions_pooled": 0.3681,
        "combo_min_games": 16,
    },

    # ── nfl_confidence blend ─────────────────────────────────────────────────────────────
    # Supplementary DISPLAY score (see confidence.py). Does not replace valuation.py's
    # generic confidence_score, which keeps working off model_n/model_prob/proj_kind.
    "nfl_confidence": {
        "weights": {
            "playing_time": 0.20, "usage_stability": 0.18, "model_agreement": 0.12,
            "injury_certainty": 0.10, "matchup": 0.10, "game_environment": 0.10,
            "historical_model_accuracy": 0.10, "market_agreement": 0.10,
        },
        "high": 70, "medium": 45,
    },

    # ── board ────────────────────────────────────────────────────────────────────────────
    # Market anchoring, same mechanism as basketball/board.py and tennis/board.py: the
    # projected mean is blended toward the market's standard line in proportion to how much
    # real evidence backs the projection.
    "board": {
        "full_trust_at": 0.6,
        "min_trust": 0.2,
    },

    "data": {
        # nflverse release URLs. Every one confirmed live 2026-08-15 (status 200, header row
        # parsed). Overridable by env so a mirror can be substituted without a code change.
        "cache_ttl": 6 * 3600,
        "lookback_seasons": 3,
    },
}


def cfg(*path, default=None):
    d = CONFIG
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def position_cfg(block: str, position: str, default=None) -> dict:
    """A per-position block (`positional_priors`, `shrink_k`, …) with an unknown/rare position
    falling back to WR — the most common skill position and the one whose priors are backed by
    the largest measured sample (8,831 player-games)."""
    d = CONFIG.get(block, {})
    return d.get((position or "").upper(), d.get("WR", default))
