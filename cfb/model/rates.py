"""
The three-tier prior fallback -- as a real empirical-Bayes chain, not a tier lookup.

Roster churn is the other thing that makes college football unlike a pro league. Every season
a third of the players a book posts props on either changed schools or have never taken a
college snap, so "average the player's last N games" has nothing to average for a large share
of the board. The three tiers are the three kinds of evidence that DO exist:

  cfb_prior_a  returning production at the same competition level
  cfb_prior_b  a transfer's production at their previous school, translated to the new level
  cfb_prior_c  a true freshman (or anyone with no college production at all) -- the recruiting
               rating is the only real signal, and it only speaks to usage, not efficiency

The tier is not a switch that selects one number. It selects what goes into the PRIOR of a
two-stage shrinkage chain, and the player's current-season production is then blended into
that prior with the codebase-standard (observed*n + prior*k)/(n+k):

    stage 1   prior_rate = (carry * history_num * level_factor + k * centre) /
                           (carry * history_den + k)
    stage 2   rate       = (current_num + prior_rate * prior_den) / (current_den + prior_den)
              eff_n      = current_den + prior_den

`k` is the league's own empirical-Bayes shrinkage strength (priors.py fits it by method of
moments). `centre` is the positional league mean, except for a tier-C usage rate where the
recruiting-rating fit supplies a better-centred prior of the same strength. `level_factor` is
the measured P4/G5/FCS translation, so a transfer's 5.4 yards a carry in the Mountain West
enters as what that level's production is worth in the Big Ten, not at face value.

`carry` is the year-over-year carryover: how much a completed prior season is worth as
evidence about this season, measured (rates.fit_carryover) as the real correlation of the same
rate across consecutive seasons for players who had both. It is measured on two COMPLETED
seasons rather than on the current partial one, because in September the current season has
almost no player with enough denominator to measure a correlation from -- and the answer would
then read as "last year tells us nothing" precisely when last year is all there is. Where it
can't be measured at all it is 0.0, which collapses the prior to the positional mean: wider,
more market-anchored, honest.

`eff_n` is the real evidence behind each estimate, and it is what the simulator's per-trial
parameter-uncertainty draw is scaled by -- so a true freshman and a returning 1,200-yard
rusher with the same point estimate do NOT get the same distribution width. That is the same
two-stage construction MLB, WNBA, tennis and NFL all use; see cfb/sim/engine.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import MIN_PLAYERS_FOR_CARRYOVER_FIT
from .priors import (EFFICIENCY_RATES, OPPORTUNITY_RATES, RATE_TERMS, PlayerTotals,
                     PositionPriors, RecruitingPrior, level_factor, recruiting_rate)

TIER_A, TIER_B, TIER_C = "cfb_prior_a", "cfb_prior_b", "cfb_prior_c"


@dataclass
class PlayerRates:
    player_id: str
    player: str = ""
    team: str = ""
    position: str = ""
    opportunity: dict = field(default_factory=dict)     # rate -> per team offensive play
    efficiency: dict = field(default_factory=dict)      # rate -> per attempt/reception
    eff_n: dict = field(default_factory=dict)           # rate -> shrinkage denominator
    # rate -> fraction of THIS rate's estimate coming from the player's own (current +
    # carried) data rather than the league prior. Board trust reads it per market rather than
    # as one blended number, because a running back's rushing projection can be almost pure
    # own-data while his receiving-yards projection is almost pure prior, and a single average
    # over every rate would understate the first and overstate the second.
    own_weight: dict = field(default_factory=dict)
    tier: str = TIER_C
    tier_reason: str = ""
    n_games: int = 0
    sample_weight: float = 0.0
    team_opportunity_share: Optional[float] = None
    level_factor_applied: Optional[float] = None


def fit_carryover(recent: dict, older: dict) -> dict[str, float]:
    """{rate: year-over-year correlation}, measured across players who carried a usable
    denominator in BOTH completed seasons. A rate with too few such players is absent, and the
    caller reads that as 0.0 -- last season contributes nothing beyond the league prior."""
    out: dict[str, float] = {}
    for rate, (num_f, den_f) in RATE_TERMS.items():
        xs, ys = [], []
        for pid, t_new in recent.items():
            t_old = older.get(pid)
            if t_old is None:
                continue
            d_new, d_old = t_new.totals[den_f], t_old.totals[den_f]
            if d_new <= 0 or d_old <= 0:
                continue
            xs.append(t_old.totals[num_f] / d_old)
            ys.append(t_new.totals[num_f] / d_new)
        if len(xs) < MIN_PLAYERS_FOR_CARRYOVER_FIT:
            continue
        if np.std(xs) <= 0 or np.std(ys) <= 0:
            continue
        r = float(np.corrcoef(xs, ys)[0, 1])
        out[rate] = max(0.0, min(1.0, r))
    return out


def classify_tier(current: Optional[PlayerTotals], history: Optional[PlayerTotals],
                  current_team: Optional[str]) -> tuple[str, str]:
    """Which kind of prior this player's evidence supports, and why -- surfaced on the line as
    proj_kind so a calibration query can score the three tiers separately (db.stat_gammas and
    every accuracy query in db.py already scope by proj_kind)."""
    if history is None or history.games == 0:
        return TIER_C, "no prior-season production"
    if current_team and history.team and history.team != current_team:
        return TIER_B, f"prior-season production at {history.team}"
    return TIER_A, "returning prior-season production"


def fit_player_rates(player_id: str, priors: PositionPriors,
                     current: Optional[PlayerTotals] = None,
                     history: Optional[PlayerTotals] = None,
                     *, player: str = "", team: str = "", position: str = "",
                     carryover: Optional[dict] = None,
                     level_factors: Optional[dict] = None,
                     origin_tier: Optional[str] = None,
                     destination_tier: Optional[str] = None,
                     recruiting_prior: Optional[RecruitingPrior] = None,
                     recruiting_rating: Optional[float] = None,
                     team_opportunity_share: Optional[float] = None) -> PlayerRates:
    """Run the two-stage chain above for every rate this position has a prior for."""
    carryover = carryover or {}
    level_factors = level_factors or {}
    tier, reason = classify_tier(current, history, team)

    out = PlayerRates(player_id=player_id,
                      player=player or (current.player if current else "")
                             or (history.player if history else ""),
                      team=team, position=(position or "").upper(), tier=tier,
                      tier_reason=reason,
                      n_games=(current.games if current else 0) + (history.games if history else 0),
                      team_opportunity_share=team_opportunity_share)

    own_weights, applied_levels = [], []
    for rate, (num_f, den_f) in RATE_TERMS.items():
        centre = priors.mean.get(rate)
        if centre is None:
            continue
        k = float(priors.k.get(rate, 0.0))
        if k <= 0:
            continue

        lf = 1.0
        hist_num = hist_den = 0.0
        carry = float(carryover.get(rate, 0.0)) if tier in (TIER_A, TIER_B) else 0.0
        if history is not None and carry > 0:
            hist_num, hist_den = history.totals[num_f], history.totals[den_f]
            if tier == TIER_B:
                lf = level_factor(level_factors, rate, origin_tier, destination_tier)
                applied_levels.append(lf)

        if tier == TIER_C and rate in OPPORTUNITY_RATES:
            rr = recruiting_rate(recruiting_prior, rate, recruiting_rating)
            if rr is not None:
                centre = rr

        prior_den = carry * hist_den + k
        prior_rate = ((carry * hist_num * lf + k * centre) / prior_den) if prior_den > 0 else centre

        cur_num = current.totals[num_f] if current else 0.0
        cur_den = current.totals[den_f] if current else 0.0
        denom = cur_den + prior_den
        value = (cur_num + prior_rate * prior_den) / denom if denom > 0 else centre

        (out.opportunity if rate in OPPORTUNITY_RATES else out.efficiency)[rate] = value
        out.eff_n[rate] = denom
        own = cur_den + carry * hist_den
        w = own / (own + k) if (own + k) > 0 else 0.0
        out.own_weight[rate] = w
        own_weights.append(w)

    out.sample_weight = round(float(np.mean(own_weights)), 3) if own_weights else 0.0
    if applied_levels:
        out.level_factor_applied = round(float(np.mean(applied_levels)), 4)
    return out


def market_sample_weight(rates: PlayerRates, market_rates: tuple) -> float:
    """How much of the projection for ONE market is the player's own data: the mean own-weight
    across only the rates that market's simulation actually consumes."""
    ws = [rates.own_weight[r] for r in market_rates if r in rates.own_weight]
    return round(float(np.mean(ws)), 3) if ws else 0.0


def rate_names() -> tuple:
    return OPPORTUNITY_RATES + EFFICIENCY_RATES
