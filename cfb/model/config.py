"""
cfb.model.config -- every constant the CFB engine uses, in one place, with its provenance.

PROVENANCE RULE (the same one nfl/config.py's module docstring states): a number here is
either MEASURED, with the window and sample size that produced it recorded beside it, or it
is a DEFINITION / MINIMUM-SAMPLE GATE that carries no statistical content of its own. Nothing
in between, and in particular nothing that looks measured but isn't.

There is not one measured statistical constant in this file, and that is deliberate rather
than an omission. This engine has never seen a live CFBD response -- no CFBD_API_KEY has
existed in any environment this code has run in (see cfb/data/cfbd_client.py's own module
docstring), so there is no window to record and no sample to cite. A positional prior, a
shrinkage strength, a per-attempt spread, an opponent factor, a pace coefficient or a
garbage-time slope written here as a literal would be a fabricated constant, which CLAUDE.md
treats as a correctness bug.

Instead every one of those quantities is FITTED AT RUNTIME from the real rows CFBD returns,
by the measurement functions in this package:

    priors.fit_positional_priors      -> per-position per-play rate means, empirical-Bayes
                                          shrinkage k, and per-attempt yard spreads
    priors.fit_level_factors          -> P4/G5/FCS competition-level translation for transfers
    priors.fit_recruiting_prior       -> recruiting rating -> usage-rate map for true freshmen
    opponent.fit_opponent_model       -> per-defense production factor + the FCS factor
    pace.fit_pace_model               -> team plays/game + the market spread/total response
    garbage_time.fit_garbage_time_model -> margin spread + the blowout usage discount
    rates.fit_carryover               -> year-over-year evidence carryover per rate

Each of those returns an explicit "unmeasured" basis (and the engine falls back to no
adjustment, never to a plausible-looking number) when the real sample behind it is thinner
than the gate below. That is the honest degradation, and it is what makes every number this
engine ships traceable to a real computation on real data.
"""
from __future__ import annotations

# Monte Carlo trials per player-game. Methodological, matching every other engine in this
# repo (nfl/config.py's model.n_sims, basketball's n=10000 default).
N_SIMS = 10000

# ── minimum-sample gates ─────────────────────────────────────────────────────────────────
# Each is the point below which a fit is reported as unmeasured rather than emitted. They are
# gates, not effects: raising one makes the engine more conservative, it never changes a
# projection's value directly.

# Player-games behind a position's prior before it is used as a shrinkage target. Same gate
# nfl/model/usage.py:positional_priors applies to its own per-position fit.
MIN_PLAYER_GAMES_FOR_PRIORS = 200
# Distinct players behind an empirical-Bayes shrinkage k. Below this the between-player
# variance estimate is itself noise, so k is not emitted.
MIN_PLAYERS_FOR_SHRINKAGE_FIT = 20
# Team-games behind the per-team pace estimate and the opponent-defense estimate.
MIN_TEAM_GAMES_FOR_PACE_FIT = 100
# Priced (spread AND total present) team-games before the market pace response is fitted.
# Same gate nfl/model/environment.py:measure_environment uses for the same regression.
MIN_PRICED_TEAM_GAMES_FOR_MARKET_FIT = 200
# Completed, priced games behind the margin-spread measurement.
MIN_GAMES_FOR_MARGIN_FIT = 100
# Team-games behind the garbage-time usage-discount regression.
MIN_TEAM_GAMES_FOR_GARBAGE_FIT = 200
# Player-games in each arm (vs FCS, vs FBS) before the FCS production factor is emitted.
MIN_GAMES_FOR_FCS_FACTOR = 100
# Matched (recruiting rating, first-season usage) pairs before the tier-C rating map is fitted.
MIN_RECRUITS_FOR_RATING_FIT = 50
# Players with a usable denominator in BOTH of two completed seasons before the year-over-year
# carryover is measured.
MIN_PLAYERS_FOR_CARRYOVER_FIT = 30

# ── definitions ──────────────────────────────────────────────────────────────────────────
# A blowout is defined here as a three-possession final margin. This is a LABEL, not a fitted
# effect: garbage_time.py fits the usage discount ON the resulting blowout probability, so a
# different threshold re-scales the same underlying signal (the fitted slope absorbs it) and
# does not invent an effect that isn't in the data. Nothing downstream reads this number
# except blowout_probability().
BLOWOUT_MARGIN = 21.0

# How many of a team's skill players count as its "starter group" for the garbage-time fit --
# a definition of who the discount is measured on, not a claim about depth charts. The
# regression measures whatever share those players actually hold; a different group size
# changes the share being regressed, not the direction or the evidence.
STARTER_GROUP_SIZE = 3

# Conference membership is a fact about the sport, not a measurement: these are the four
# power conferences as of the 2026 season. Realignment moves this most offseasons -- team
# tier is only ever read from the conference CFBD itself returns for a team, so a team that
# changes conference is re-tiered automatically on the next roster sync.
POWER_CONFERENCES = frozenset({"SEC", "Big Ten", "Big 12", "ACC"})

# ── board ────────────────────────────────────────────────────────────────────────────────
# Market anchoring, the same convention (and the same numbers) nfl/config.py's `board` block
# and basketball/board.py's _FULL_TRUST_AT/0.2 floor already ship under, so a CFB line and a
# WNBA line with the same evidence behind them defer to the market by the same amount.
BOARD_FULL_TRUST_AT = 0.6
BOARD_MIN_TRUST = 0.2
