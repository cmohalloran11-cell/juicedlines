"""
Dixon-Coles goal expectancy + score-probability matrix.

expectancy(fixture, ratings) → (λ_home, μ_away): each side's expected goals from
attack × opponent-defense × league average, with a home-advantage bump only when a
host nation is actually playing at home (WC 2026 is otherwise neutral).

score_matrix(λ, μ, ρ) → an (N+1)×(N+1) matrix of P(home=h, away=a), with the
Dixon-Coles low-score correction ρ that couples 0-0/1-0/0-1/1-1 (independent
Poisson over-predicts those). The sim samples exact scores from this matrix.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from ..config import load
from ..data.base import Fixture


def expectancy(fx: Fixture, ratings: dict[str, dict]) -> tuple[float, float]:
    cfg = load()["model"]
    lg = cfg["league_avg_goals"]
    ra, rb = ratings.get(fx.home), ratings.get(fx.away)
    if not ra or not rb:
        return lg, lg
    lam = lg * ra["attack"] * rb["defense"]      # home expected goals
    mu = lg * rb["attack"] * ra["defense"]       # away expected goals
    if not fx.neutral and fx.host_home == fx.home:
        lam *= cfg["home_advantage"]
    elif not fx.neutral and fx.host_home == fx.away:
        mu *= cfg["home_advantage"]
    return float(max(0.2, lam)), float(max(0.2, mu))


def _tau(h: int, a: int, lam: float, mu: float, rho: float) -> float:
    if h == 0 and a == 0:
        return 1.0 - lam * mu * rho
    if h == 0 and a == 1:
        return 1.0 + lam * rho
    if h == 1 and a == 0:
        return 1.0 + mu * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def score_matrix(lam: float, mu: float, rho: float | None = None,
                 max_goals: int | None = None) -> np.ndarray:
    cfg = load()["model"]
    rho = cfg["dc_rho"] if rho is None else rho
    mg = cfg["max_goals"] if max_goals is None else max_goals
    h = poisson.pmf(np.arange(mg + 1), lam)
    a = poisson.pmf(np.arange(mg + 1), mu)
    M = np.outer(h, a)
    for i in (0, 1):
        for j in (0, 1):
            M[i, j] *= _tau(i, j, lam, mu, rho)
    M = np.clip(M, 0, None)
    return M / M.sum()
