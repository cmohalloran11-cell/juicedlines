"""
Monte Carlo engine -- one CFB player-game to a distribution per market.

The projection is `team plays x usage share x efficiency` at every step (see
cfb/model/priors.py), and uncertainty enters at four independent points, each drawn PER TRIAL,
in the order a football game resolves them:

  1. team plays        Normal around cfb/model/pace.py's projection, with the MEASURED
                       residual spread of real team play counts around that fit. CFB's
                       play-count variance across offenses is the widest in the sport, so this
                       stage carries more of the total distribution here than its NFL
                       equivalent does.
  2. garbage time      cfb/model/garbage_time.py's fitted usage multiplier for this game's
                       blowout probability, scaled by how starter-like the player is. Applied
                       to the USAGE share only -- a starter pulled in the third quarter runs
                       fewer plays, he does not become a worse runner.
  3. parameter draw    the two-stage layer. Before any outcome is sampled, each rate gets its
                       own multiplier scaled by the real evidence behind THAT rate
                       (rates.eff_n, the shrinkage denominator). Count rates use
                       Gamma(eff_n, 1/eff_n) -- mean 1, CV 1/sqrt(eff_n), the standard
                       Gamma-Poisson effective-sample-size approximation MLB, WNBA and NFL all
                       use. The bounded rate (completion rate) uses
                       Beta(rate*eff_n, (1-rate)*eff_n), because a Gamma multiplier on a
                       probability can push it past 1 -- the construction tennis uses for a
                       serve probability.
                       This is what makes a true freshman priced off a recruiting rating come
                       out WIDER than a returning starter with the same point estimate, which
                       is the entire reason the three-tier prior records its evidence depth
                       rather than just its value.
  4. outcome           carries / receptions / pass attempts Poisson at the drawn per-trial
                       rate; completions Binomial on the drawn attempts (that composition is a
                       beta-binomial, so the overdispersion is produced rather than asserted);
                       yards Gamma with shape scaling in the drawn count, using the per-attempt
                       spread priors.py measured off the same box scores the rates came from;
                       touchdowns Poisson on the drawn opportunities.

Anytime TD is derived, not fitted separately: a trial scores if its drawn rushing or receiving
touchdowns exceed zero, which is what the market actually settles on, and it inherits the
correlation with carries/receptions from sharing their draws inside the trial.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..model.config import N_SIMS
from ..model.rates import PlayerRates

# Canonical board stat label (cfb.config.ODDS_MARKET_TO_STAT's values) -> simulated array.
MARKET_KEYS = {
    "Passing Yards": "pass_yards",
    "Rushing Yards": "rush_yards",
    "Receiving Yards": "rec_yards",
    "Receptions": "receptions",
    "Anytime TD": "anytime_td",
}

# Which markets a position's projection is meaningful for. A WR's simulated pass_yards is ~0,
# and shipping it would read as an enormous Under edge on a market no book would post for him;
# an unknown position is left ungated because the player's own fitted rates already answer it.
POSITION_MARKETS = {
    "QB": {"Passing Yards", "Rushing Yards", "Anytime TD"},
    "RB": {"Rushing Yards", "Receiving Yards", "Receptions", "Anytime TD"},
    "WR": {"Receiving Yards", "Receptions", "Rushing Yards", "Anytime TD"},
    "TE": {"Receiving Yards", "Receptions", "Anytime TD"},
}


# Which fitted rates each market's simulation actually consumes -- what board trust for that
# market is measured over (see model/rates.py::market_sample_weight).
MARKET_RATES = {
    "Passing Yards": ("pass_att_per_play", "yards_per_attempt"),
    "Rushing Yards": ("rush_att_per_play", "yards_per_carry"),
    "Receiving Yards": ("rec_per_play", "yards_per_reception"),
    "Receptions": ("rec_per_play",),
    "Anytime TD": ("rush_att_per_play", "rec_per_play", "rush_td_per_carry",
                   "rec_td_per_reception"),
}


def supports(position: Optional[str], stat_type: str) -> bool:
    markets = POSITION_MARKETS.get((position or "").upper())
    return stat_type in markets if markets else stat_type in MARKET_KEYS


def _theta(eff_n: Optional[float], n: int, rng: np.random.Generator) -> np.ndarray:
    """Per-trial multiplier on a count rate: mean 1, CV = 1/sqrt(eff_n)."""
    shape = max(float(eff_n or 0.0), 1.0)
    return rng.gamma(shape, 1.0 / shape, n)


def _bounded(rate: float, eff_n: Optional[float], n: int, rng: np.random.Generator) -> np.ndarray:
    r = min(max(float(rate), 1e-4), 1.0 - 1e-4)
    c = max(float(eff_n or 0.0), 2.0)
    return rng.beta(r * c, (1.0 - r) * c, n)


def _gamma_yards(count: np.ndarray, per_unit_mean: np.ndarray, per_unit_sd: float,
                 rng: np.random.Generator) -> np.ndarray:
    """Total yards over `count` attempts: mean count*m, variance count*sd^2, drawn as a Gamma
    so the result is right-skewed and non-negative. Zero attempts yields exactly zero yards,
    which is the real outcome rather than a Gamma degenerate draw."""
    out = np.zeros(len(count), dtype=float)
    mean = count * per_unit_mean
    var = count * (per_unit_sd ** 2)
    live = (count > 0) & (mean > 1e-9) & (var > 1e-9)
    if not live.any():
        return out
    shape = (mean[live] ** 2) / var[live]
    scale = var[live] / mean[live]
    out[live] = rng.gamma(np.clip(shape, 1e-3, None), scale)
    return out


def simulate(rates: PlayerRates, plays: float, plays_sd: float, unit_sd: dict,
             garbage_multiplier: float = 1.0, opponent_factor: float = 1.0,
             n: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> dict:
    """Distributions for every market this engine models. Returns {key: np.ndarray} plus the
    intermediate `team_plays` array, which the board surfaces as the projected pace."""
    rng = rng or np.random.default_rng()
    n = int(n or N_SIMS)
    plays = max(float(plays), 1.0)
    team_plays = np.clip(rng.normal(plays, max(1.0, float(plays_sd)), n),
                         0.5 * plays, 1.6 * plays)
    usage_plays = team_plays * float(garbage_multiplier)

    opp = rates.opportunity
    eff = rates.efficiency
    eff_n = rates.eff_n
    out: dict[str, np.ndarray] = {"team_plays": team_plays}

    carries = rng.poisson(np.clip(
        opp.get("rush_att_per_play", 0.0) * usage_plays
        * _theta(eff_n.get("rush_att_per_play"), n, rng), 0.0, None)).astype(float)
    receptions = rng.poisson(np.clip(
        opp.get("rec_per_play", 0.0) * usage_plays
        * _theta(eff_n.get("rec_per_play"), n, rng), 0.0, None)).astype(float)
    attempts = rng.poisson(np.clip(
        opp.get("pass_att_per_play", 0.0) * usage_plays
        * _theta(eff_n.get("pass_att_per_play"), n, rng), 0.0, None)).astype(float)
    out["rush_attempts"] = carries
    out["receptions"] = receptions
    out["pass_attempts"] = attempts

    completion_rate = _bounded(eff.get("completion_rate", 0.6),
                               eff_n.get("completion_rate"), n, rng)
    completions = rng.binomial(attempts.astype(np.int64), completion_rate).astype(float)
    out["completions"] = completions

    ypc = (eff.get("yards_per_carry", 0.0) * opponent_factor
           * _theta(eff_n.get("yards_per_carry"), n, rng))
    out["rush_yards"] = _gamma_yards(carries, ypc, float(unit_sd.get("yards_per_carry", 0.0)), rng)

    ypr = (eff.get("yards_per_reception", 0.0) * opponent_factor
           * _theta(eff_n.get("yards_per_reception"), n, rng))
    out["rec_yards"] = _gamma_yards(receptions, ypr,
                                    float(unit_sd.get("yards_per_reception", 0.0)), rng)

    ypa = (eff.get("yards_per_attempt", 0.0) * opponent_factor
           * _theta(eff_n.get("yards_per_attempt"), n, rng))
    out["pass_yards"] = _gamma_yards(attempts, ypa,
                                     float(unit_sd.get("yards_per_attempt", 0.0)), rng)

    rush_tds = rng.poisson(np.clip(
        carries * eff.get("rush_td_per_carry", 0.0)
        * _theta(eff_n.get("rush_td_per_carry"), n, rng), 0.0, None)).astype(float)
    rec_tds = rng.poisson(np.clip(
        receptions * eff.get("rec_td_per_reception", 0.0)
        * _theta(eff_n.get("rec_td_per_reception"), n, rng), 0.0, None)).astype(float)
    out["rush_tds"] = rush_tds
    out["rec_tds"] = rec_tds
    out["pass_tds"] = rng.poisson(np.clip(
        completions * eff.get("pass_td_per_completion", 0.0)
        * _theta(eff_n.get("pass_td_per_completion"), n, rng), 0.0, None)).astype(float)
    out["anytime_td"] = ((rush_tds + rec_tds) > 0).astype(float)
    return out


def market_array(sim: dict, stat_type: str) -> Optional[np.ndarray]:
    key = MARKET_KEYS.get(stat_type)
    return sim.get(key) if key else None


def prob_over(arr: np.ndarray, line: float) -> float:
    return float((arr > line).mean())


def summary(arr: np.ndarray) -> dict:
    q = np.percentile(arr, [10, 25, 50, 75, 90])
    return {"mean": round(float(arr.mean()), 2), "median": round(float(q[2]), 2),
            "std_dev": round(float(arr.std()), 2),
            "p10": round(float(q[0]), 2), "p25": round(float(q[1]), 2),
            "p50": round(float(q[2]), 2), "p75": round(float(q[3]), 2),
            "p90": round(float(q[4]), 2)}
