"""Unit tests: point/game/set math + simulator vs closed-form, and value math."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tennis.model.matchup import race_prob, hold_prob, set_win_prob, match_from_set, clamp_p
from tennis.model.rates import PlayerRates, TourBaselines
from tennis.sim.engine import simulate, prob_over
from tennis.value.finder import implied_prob, to_decimal, value_row


def _base():
    return TourBaselines("ATP", 0.636, 0.364, 0.079, 0.038, 6.7, {"Hard": 0.638})


def _player(name, spw, rpw, ace=0.07, df=0.035):
    return PlayerRates(name, name, "ATP", spw, rpw, ace, df, 6.7, n_matches=50)


def test_hold_prob_monotonic_and_bounded():
    assert 0 < hold_prob(0.5) < 1
    assert hold_prob(0.50) < hold_prob(0.62) < hold_prob(0.75)   # better server holds more
    assert abs(hold_prob(0.5) - 0.5) < 0.02                      # 50% server ≈ 50% hold


def test_race_prob_edges():
    assert race_prob(0.5, 7) == 0.5                              # symmetric tiebreak
    assert race_prob(0.8, 4) > 0.95 and race_prob(0.2, 4) < 0.05


def test_set_and_match_prob_sane():
    s = set_win_prob(hold_prob(0.65), hold_prob(0.60), 0.55)
    assert 0.5 < s < 0.9                                         # stronger server favored
    assert match_from_set(0.6, 5) > match_from_set(0.6, 3)      # more sets → favorite likelier
    assert abs(match_from_set(0.5, 5) - 0.5) < 1e-9


def test_clamp():
    assert clamp_p(0.99) == 0.85 and clamp_p(0.10) == 0.50


def test_sim_matches_closed_form_match_prob():
    base = _base()
    a, b = _player("A", 0.68, 0.40), _player("B", 0.62, 0.36)
    sim = simulate(a, b, "Hard", base, best_of=3, n=20000, rng=np.random.default_rng(3))
    from tennis.model.matchup import match_win_analytic
    sim_win = float((sim["winner"] == 0).mean())
    ana = match_win_analytic(a, b, "Hard", base, 3)
    assert abs(sim_win - ana) < 0.02                            # simulator ≈ analytic


def test_sim_distributions_valid():
    base = _base()
    sim = simulate(_player("A", 0.68, 0.40), _player("B", 0.62, 0.36), "Hard", base,
                   best_of=3, n=5000, rng=np.random.default_rng(1))
    assert (sim["total_games"] >= 12).all()                     # a BO3 has ≥ ~12 games
    assert 0 <= prob_over(sim["aces_a"], 5.5) <= 1
    assert (sim["sets_a"] + sim["sets_b"] >= 2).all()


def test_value_math():
    assert abs(implied_prob(-110) - 0.5238) < 0.001
    assert abs(to_decimal(+100) - 2.0) < 1e-9
    r = value_row("X", "aces", 5.5, "over", 100, 0.60)
    assert abs(r["edge"] - 0.10) < 1e-9 and abs(r["ev"] - 0.20) < 1e-9


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn(); print("ok", fn.__name__)
