"""
Monte-Carlo engine — turns a player's per-possession rates + projected minutes +
projected pace into a DISTRIBUTION per stat (not just a point estimate), so we can
price over/unders and carry variance.

Per sim we draw minutes and pace ONCE, then each base stat as an overdispersed
count (Negative-Binomial: var = μ·(1 + disp·μ)). Because minutes and pace are shared
across the stats within a sim, combos (PRA, stocks, …) come out correctly correlated.
The variance width is a league knob — WNBA tight, Summer League wide.
"""

from __future__ import annotations

import numpy as np

from .. import BASE_STATS, COMBOS
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
             rng: np.random.Generator | None = None) -> dict:
    rng = rng or np.random.default_rng()
    opp_adj = opp_adj or {}

    minutes = np.clip(rng.normal(proj_minutes, max(0.1, minutes_sd), n), 0.0, game_len)
    pace = np.clip(rng.normal(matchup_pace, max(0.5, pace_sd_frac * matchup_pace), n),
                   0.5 * matchup_pace, 1.6 * matchup_pace)
    poss = player_possessions(minutes, game_len, pace)      # vectorized

    out = {"minutes": minutes, "poss": poss}
    for s in BASE_STATS:
        mu = rates.per_poss.get(s, 0.0) * poss * opp_adj.get(s, 1.0)
        out[s] = _negbin(mu, disp, rng)
    return out


def market_array(sim: dict, key: str) -> np.ndarray | None:
    """Array for a base stat or a combo (summed from its components)."""
    if key in BASE_STATS:
        return sim.get(key)
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
