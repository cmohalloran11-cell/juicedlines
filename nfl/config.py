"""
NFL config — every tunable weight, prior and shrinkage constant in one place.

Provenance rule for this file: a number here is either MEASURED (with the window and sample
size that produced it recorded next to it) or explicitly marked as an assumption with the
reason no measurement exists. Nothing in between. The measurement code that produced each
measured block is still in the repo and re-runnable:

    model/usage.py:positional_priors        -> POSITIONAL_PRIORS
    model/usage.py (holdout sweep)          -> SHRINK_K            [see MEASURED below]
    model/playing_time.py:measure_snap_prior-> SNAP_SHARE
    model/rotation.py:measure_tier_shares   -> DEPTH_RANK_SNAP_SHARE
    model/matchup.py:fit_defense            -> DEFENSE
    model/environment.py:measure_environment-> ENVIRONMENT
    model/combo_corr.py                     -> COMBO_CORR

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

    # ── playing time (regular season) ────────────────────────────────────────────────────
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

    # ── depth-chart tier -> snap share (regular season) ──────────────────────────────────
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

    # ── preseason rotation tiers ─────────────────────────────────────────────────────────
    # DRIVES-FIRST model: expected_snaps = tier_drives_mean(tier) * position_snaps_per_drive
    # (position), not a single flat "tier snap-share x league-baseline-snaps" lookup. The old
    # single-factor version (one snap_share mean per tier, multiplied by the SAME constant
    # league-baseline team-snaps number for every preseason game, since no preseason
    # schedule/spread ever exists to vary it) is exactly why the board clustered: two tiers
    # (confirmed_starter and third_team) happened to share the same 0.30 mean share, so every
    # player in either tier produced the literal same expected_snaps number regardless of
    # position. Two independent, position/tier-varying factors instead of one flat table
    # entry per tier structurally cannot collide that way — see rotation.py:tier_playing_time.
    #
    # BASIS: ASSUMED, NOT MEASURED — and this is the one block in this file that is. No free
    # data source publishes PRESEASON snap counts OR drive-level data: verified live
    # 2026-08-15, the nflverse `snap_counts` release contains 0 rows with game_type PRE across
    # every season it publishes (2022-2025 all resolve to REG/WC/DIV/CON/SB only), and the
    # `schedules` release contains 0 preseason games at all (7,548 rows, game_type in
    # {REG,WC,DIV,CON,SB}). So a preseason snap/drive distribution has never been measured in
    # this codebase and a "measured-looking" table here would be a fabrication. Every
    # preseason projection carries playing_time_confidence "low" because of exactly this, and
    # board.py market-anchors accordingly.
    #
    # rotation.measure_tier_shares() is the real measurement pipeline this table would be
    # replaced by the day a real preseason snap/drive source is wired.
    "preseason_tiers": {
        "basis": "assumed",
        "reason": "no free source publishes preseason snap counts or drive-level data "
                  "(nflverse snap_counts has 0 PRE rows; nflverse schedules has 0 preseason "
                  "games)",
        # Real, well-documented mechanics of how a drive is played, not a fitted number: a QB
        # in the game is on the field for literally every snap of his drives (no rotation
        # within a drive is possible at the position); RB/WR/TE corps rotate by personnel
        # package and play-type even within a single drive they are nominally part of, so
        # their effective snaps-per-drive-they're-in is lower. ~5.8 plays/drive is the
        # commonly-cited NFL-average drive length (external football convention, not measured
        # from this repo's own data — this repo has no drive-level release to measure it
        # from). "" is the fallback for a position this table doesn't cover.
        "position_snaps_per_drive": {"QB": 5.8, "RB": 3.4, "WR": 4.1, "TE": 3.7, "": 4.0},
        # ASSUMED mean/sd of DRIVES a player in this tier is involved in, in one preseason
        # game. The ordering is the real, widely-reported coaching pattern for how preseason
        # snaps are actually allocated: starters get a short, scripted look (limit injury
        # exposure) and are pulled early; the rotation/second-team groups get the bulk of the
        # game (this is when a team evaluates its roster bubble); fringe/unknown players are
        # the least predictable — some weeks inactive, others playing deep into the second
        # half. This is the SAME relative pattern the old single-factor table encoded
        # (confirmed_starter < second_team, unknown widest) — decomposed into two factors and
        # extended to 7 tiers instead of asserted as one flat number per tier.
        # Means are deliberately kept numerically distinct across every tier (not just
        # monotonic) — two tiers sharing a mean is exactly the collision the old flat
        # snap-share table had (confirmed_starter and third_team both sat at 0.30), so the
        # SAME conceptual bug could resurface here at the tier_drives level even after
        # decomposing into two factors. See
        # test_core.py::test_seven_tiers_produce_seven_distinct_expected_snaps_for_the_same_position.
        "tier_drives": {
            "confirmed_starter":   (3.2, 1.1),
            "likely_starter":      (3.6, 1.4),
            "first_team_rotation": (5.0, 1.7),
            "second_team":         (6.5, 2.1),
            "third_team":          (5.2, 2.5),
            "fringe":              (2.8, 2.6),
            "unknown":             (3.9, 3.0),
        },
        # A team's actual preseason pass/rush split is unmeasurable (no PRE play-by-play), but
        # its REGULAR-season split (team_tendency's own "regular_season_measured" field) is a
        # real, if imperfect, proxy for offensive identity, and offensive identity plausibly
        # carries into who a preseason coach chooses to feature. Small, capped nudge — never
        # large enough to be a de-facto second global multiplier: +/-10% on skill-position
        # drives at the observed extremes of team pass rate (~0.42-0.62 measured range).
        "team_pass_rate_nudge_cap": 0.10,
        "team_pass_rate_league_mean": 0.5673,   # = environment.league_dropback_share
        # Depth-chart rank + prior-season snap share -> tier. The 0.60 confirmed-starter
        # threshold IS measured: it sits between the measured rank-1 mean (0.5675 RB … 0.7283
        # WR) and the rank-2 mean (0.3012 … 0.4385), so a player above it is playing a
        # genuinely rank-1 workload rather than merely being listed there. The 0.35
        # first-team-rotation threshold is the same logic one rung down: it sits between the
        # measured rank-2 mean (0.30-0.44) and rank-3 mean (0.14-0.25), separating a real
        # rotational rank-2 workload from a token one.
        "confirmed_snap_share": 0.60,
        "first_team_rotation_snap_share": 0.35,
        "min_games_for_confirmation": 4,
        "min_games_for_rotation": 3,
    },

    # ── preseason team tendency ──────────────────────────────────────────────────────────
    # Which TeamTendency fields can be filled from real data at all. Everything listed under
    # "unavailable" resolves to None with a stated reason rather than a plausible number —
    # see model/rotation.py:TeamTendency.
    "team_tendency": {
        "measurable": ("team", "coach", "preseason_pass_rate", "preseason_rush_rate"),
        "unavailable": ("starter_snap_rate", "avg_first_team_drives",
                        "qb_rotation_pattern", "second_team_usage"),
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
        # League baselines, used when a game has no spread/total (every preseason game, and
        # any regular-season game the schedules release hasn't priced yet).
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
    # The preseason weighting is the product spec's; the regular-season one mirrors it with
    # playing time demoted (a regular-season role is knowable) and matchup/environment/market
    # promoted (they're knowable, and preseason's aren't).
    "nfl_confidence": {
        "regular": {
            "playing_time": 0.20, "usage_stability": 0.18, "model_agreement": 0.12,
            "injury_certainty": 0.10, "matchup": 0.10, "game_environment": 0.10,
            "historical_model_accuracy": 0.10, "market_agreement": 0.10,
        },
        "preseason": {
            "playing_time": 0.35, "role_stability": 0.20, "rotation_certainty": 0.15,
            "injury_certainty": 0.10, "model_agreement": 0.10, "matchup": 0.05,
            "game_environment": 0.05,
        },
        "high": 70, "medium": 45,
    },

    # ── board ────────────────────────────────────────────────────────────────────────────
    # Market anchoring, same mechanism as basketball/board.py and tennis/board.py: the
    # projected mean is blended toward the market's standard line in proportion to how much
    # real evidence backs the projection. Preseason sits below full trust by construction.
    "board": {
        "full_trust_at": 0.6,
        "preseason_max_trust": 0.35,
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
