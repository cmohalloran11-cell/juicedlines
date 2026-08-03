"""Unit tests: point/game/set math + simulator vs closed-form, and value math."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tennis.model.matchup import race_prob, hold_prob, set_win_prob, match_from_set, clamp_p, tiebreak_prob
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


def test_tiebreak_prob_matches_independent_monte_carlo():
    # 2026-08 regression guard: tiebreak_prob used to average A's serve/return win
    # probabilities into one constant and reuse the single-server race formula -- a
    # DIFFERENT process from the real alternating-server tiebreak, not an approximation of
    # it. Validate the exact DP against an independent brute-force Monte Carlo simulation of
    # the real point-by-point alternating process (deliberately NOT reusing any production
    # code, including the shared server_is_a helper, so this can't share a bug with it).
    rng = np.random.default_rng(7)

    def mc_tiebreak(psa, psb, n=200_000):
        pa = np.zeros(n, int); pb = np.zeros(n, int)
        done = np.zeros(n, bool); a_wins = np.zeros(n, bool)
        for point in range(200):  # far more points than any real tiebreak needs
            active = ~done
            if not active.any():
                break
            total = pa + pb
            # A serves point 1 (total==0), else alternates in pairs starting with B
            a_serves = (total == 0) | (((total - 1) // 2) % 2 == 1)
            p_a = np.where(a_serves, psa, 1.0 - psb)
            a_scores = active & (rng.random(n) < p_a)
            pa += a_scores; pb += (active & ~a_scores)
            newly_a = active & (pa >= 7) & (pa - pb >= 2)
            newly_b = active & (pb >= 7) & (pb - pa >= 2)
            a_wins |= newly_a
            done |= newly_a | newly_b
        return float(a_wins.mean())

    for psa, psb in [(0.68, 0.62), (0.70, 0.55), (0.75, 0.50), (0.6, 0.6)]:
        exact = tiebreak_prob(psa, psb)
        mc = mc_tiebreak(psa, psb)
        assert abs(exact - mc) < 0.01, (psa, psb, exact, mc)


def test_tiebreak_prob_beats_the_old_averaged_approximation():
    # The old formula (kept here inline, not imported, so this test can't accidentally start
    # passing again if someone reintroduces it elsewhere) systematically differs from the
    # correct alternating-server DP whenever the two players' serve/return splits are
    # asymmetric -- confirming the fix is a real, directionally-consistent change, not a
    # rounding-level tweak.
    def old_broken_tiebreak(psa, psb):
        return race_prob((psa + (1 - psb)) / 2, 7)

    for psa, psb in [(0.68, 0.62), (0.70, 0.55), (0.75, 0.50)]:
        old = old_broken_tiebreak(psa, psb)
        new = tiebreak_prob(psa, psb)
        assert abs(new - old) > 0.003, "fix should meaningfully move an asymmetric matchup"
    # symmetric matchups are the one case where both formulas agree (both reduce to 0.5)
    assert abs(tiebreak_prob(0.6, 0.6) - old_broken_tiebreak(0.6, 0.6)) < 1e-9


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
