"""Unit tests for the Poisson / Dixon-Coles simulation."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wc.data.base import Fixture, TeamStrength, Player
from wc.model.strength import ratings
from wc.model.goal_expectancy import expectancy, score_matrix, _tau
from wc.sim.engine import simulate


def test_score_matrix_is_a_distribution():
    M = score_matrix(1.4, 1.1, rho=-0.05)
    assert abs(M.sum() - 1.0) < 1e-9
    assert (M >= 0).all()


def test_score_matrix_marginal_mean_matches_lambda():
    # with rho=0 (independent Poisson) each marginal mean should equal its rate
    M = score_matrix(1.5, 1.0, rho=0.0, max_goals=15)
    home_goals = np.arange(M.shape[0])
    away_goals = np.arange(M.shape[1])
    assert abs((M.sum(axis=1) * home_goals).sum() - 1.5) < 0.03
    assert abs((M.sum(axis=0) * away_goals).sum() - 1.0) < 0.03


def test_dixon_coles_tau():
    # rho<0 lifts mass on 0-0 and 1-1 (independent Poisson under-weights draws)
    assert _tau(0, 0, 1.4, 1.1, -0.05) > 1.0
    assert _tau(1, 1, 1.4, 1.1, -0.05) > 1.0
    assert _tau(2, 3, 1.0, 1.0, -0.05) == 1.0        # untouched outside the low-score cells


def test_expectancy_scales_with_strength():
    strong = TeamStrength("Strong", 2.4, 0.6, 2.3, 0.7)
    weak = TeamStrength("Weak", 0.9, 1.8, 1.0, 1.7)
    fx = Fixture("t", "Strong", "Weak", "2026")
    lam, mu = expectancy(fx, ratings([strong, weak]))
    assert lam > mu                                   # strong team expected to outscore weak


def test_simulate_produces_valid_and_sensible_probs():
    fx = Fixture("t", "A", "B", "2026", knockout=True)
    sa = TeamStrength("A", 2.0, 0.8, 1.9, 0.9, 14, 0.55)
    sb = TeamStrength("B", 1.5, 1.0, 1.4, 1.1, 12, 0.50)
    rt = ratings([sa, sb])
    A = [Player("Striker", "A", "FW", 1400, 3.5, 1.4, 0.5, 0.30, 1.0, 0.15, 0, 0, 0.9),
         Player("Defender", "A", "DF", 1400, 0.5, 0.2, 0.05, 0.03, 1.5, 0.30, 0, 0, 0.9),
         Player("Keeper", "A", "GK", 1440, 0, 0, 0, 0, 0.1, 0.03, 0, 0.72, 0.95)]
    B = [Player("StrikerB", "B", "FW", 1400, 3.0, 1.2, 0.45, 0.28, 1.0, 0.15, 0, 0, 0.9)]
    sim = simulate(fx, rt, A, B, sa, sb, np.random.default_rng(1))
    ps = sim["players"]

    for name in ("Striker", "Defender"):
        for k in ("anytime", "two_plus"):
            assert 0.0 <= ps[name]["goal"][k] <= 1.0
    # a high-xG striker is likelier to score than a defender
    assert ps["Striker"]["goal"]["anytime"] > ps["Defender"]["goal"]["anytime"]
    # GK gets a saves market, no goal market
    assert "saves" in ps["Keeper"] and "goal" not in ps["Keeper"]
    assert 0.0 <= ps["Keeper"]["saves"]["over"][1.5] <= 1.0
    # knockout intensity is applied (>1)
    assert sim["card_intensity"] > 1.0


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn(); print("ok", fn.__name__)
