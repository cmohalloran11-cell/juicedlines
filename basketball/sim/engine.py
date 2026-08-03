"""
Monte-Carlo engine — turns a player's per-possession rates + projected minutes +
projected pace into a DISTRIBUTION per stat (not just a point estimate), so we can
price over/unders and carry variance.

Per sim we draw minutes and pace ONCE, then each base stat as an overdispersed
count (Negative-Binomial: var = μ·(1 + disp·μ)). Because minutes and pace are shared
across the stats within a sim, combos (PRA, stocks, …) come out correctly correlated.
The variance width is a league knob.

Two-stage uncertainty (2026-08): every simulated count used to condition on the rate
estimate as a fixed constant across all n trials — only OUTCOME (sampling) variance was
represented; the estimate's own PARAMETER uncertainty (how wrong the shrunk per-poss rate
itself might be, given how little/much real data supports it — rates.eff_poss) never
propagated into the distribution at all. A debut-game player and a 40-game veteran with
the identical point estimate got the identical spread, which understates real uncertainty
for the debut player and (to a lesser degree) overstates it for the veteran. Now each trial
first draws its own rate multiplier theta ~ Gamma(eff_poss, 1/eff_poss) (mean 1, CV =
1/sqrt(eff_poss) — the standard Gamma-Poisson effective-sample-size approximation), then
samples the outcome conditional on that trial's rate. This is the textbook two-stage
decomposition: Var(Y) = E[Var(Y|theta)] + Var(E[Y|theta]) — outcome variance plus parameter
variance, not outcome variance alone. Each stat draws its OWN independent theta (a hot
scoring night and a hot rebounding night aren't the same underlying uncertainty), while
minutes/pace stay a single shared draw per trial (that shared-ness is what makes combos
correlate correctly — see the module comment above).
"""

from __future__ import annotations

import numpy as np

from .. import BASE_STATS, COMBOS, DERIVED_STATS
from ..model.rates import PlayerRates, player_possessions


def _negbin(mu: np.ndarray, disp: float, rng: np.random.Generator) -> np.ndarray:
    """Draw counts with mean `mu` and variance mu*(1+disp*mu). disp→0 = Poisson."""
    mu = np.clip(mu, 1e-6, None)
    if disp <= 0:
        return rng.poisson(mu)
    r = 1.0 / disp                      # NegBin 'number of successes' (constant)
    p = 1.0 / (1.0 + disp * mu)         # per-sim success prob
    return rng.negative_binomial(r, p)


def simulate(rates: PlayerRates, proj_minutes: float, minutes_sd: float,
             matchup_pace: float, pace_sd_frac: float, game_len: float,
             disp: float, opp_adj: dict | None = None, n: int = 10000,
             rng: np.random.Generator | None = None,
             orb_share: float | None = None) -> dict:
    rng = rng or np.random.default_rng()
    opp_adj = opp_adj or {}

    minutes = np.clip(rng.normal(proj_minutes, max(0.1, minutes_sd), n), 0.0, game_len)
    pace = np.clip(rng.normal(matchup_pace, max(0.5, pace_sd_frac * matchup_pace), n),
                   0.5 * matchup_pace, 1.6 * matchup_pace)
    poss = player_possessions(minutes, game_len, pace)      # vectorized

    # Parameter-uncertainty stage: how much real evidence backs this rate estimate.
    # eff_poss<=0 shouldn't happen (fit_rates/prior_only_rates always add >=1 pseudo-
    # possession of prior), but floor defensively rather than let a Gamma shape of 0 blow up.
    eff_poss = max(float(getattr(rates, "eff_poss", 0.0) or 0.0), 1.0)

    out = {"minutes": minutes, "poss": poss}
    for s in BASE_STATS:
        theta = rng.gamma(eff_poss, 1.0 / eff_poss, n)         # mean 1, CV = 1/sqrt(eff_poss)
        mu = rates.per_poss.get(s, 0.0) * poss * opp_adj.get(s, 1.0) * theta
        out[s] = _negbin(mu, disp, rng)

    # Offensive/defensive rebounds are DERIVED, not fitted (see DERIVED_STATS): split each
    # simulated rebound binomially at the player's offensive share. This inherits the
    # rebound distribution's variance, stays correlated with `reb`, and guarantees
    # orb + drb == reb in every sim — which is what actually happens in a game.
    if orb_share is not None and "reb" in out:
        reb_i = np.rint(out["reb"]).astype(np.int64)
        np.clip(reb_i, 0, None, out=reb_i)
        orb = rng.binomial(reb_i, float(np.clip(orb_share, 0.01, 0.99)))
        out["orb"] = orb.astype(out["reb"].dtype, copy=False)
        out["drb"] = (reb_i - orb).astype(out["reb"].dtype, copy=False)
    return out


def market_array(sim: dict, key: str) -> np.ndarray | None:
    """Array for a base stat, a DERIVED stat (orb/drb), or a combo (summed from components)."""
    if key in BASE_STATS or key in DERIVED_STATS:
        return sim.get(key)          # derived stats are absent if no orb_share was supplied
    if key in COMBOS:
        parts = [sim[p] for p in COMBOS[key] if p in sim]
        return sum(parts) if parts else None
    return None


def prob_over(arr: np.ndarray, line: float) -> float:
    """P(stat strictly over the line) — the over side of an O/U."""
    return float((arr > line).mean())


def summary(arr: np.ndarray) -> dict:
    return {
        "mean": round(float(arr.mean()), 2),
        "sd": round(float(arr.std()), 2),
        "p15": float(np.percentile(arr, 15)),
        "p50": float(np.percentile(arr, 50)),
        "p85": float(np.percentile(arr, 85)),
    }
