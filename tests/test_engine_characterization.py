"""
Characterization tests for the projection/simulation engine.

These lock the CURRENT behavior of the statistically load-bearing pieces so the
planned refactors (decomposing analytics.py, Postgres migration, provenance) can be
made with a safety net. They assert invariants, not exact floats, so they stay valid
as long as the math means what it's documented to mean.

The crown-jewel invariant is the correlation-aware combo simulation
(projector_bridge._induce_corr): it must re-pair the H/R/RBI marginals WITHOUT
changing any single marginal — that's what keeps every single-stat P(over) identical
while making the combo sum respect real baseball dependence.
"""
from __future__ import annotations

import numpy as np
import pytest

import projector_bridge as pb
from projector.models import montecarlo as mc


# ── _induce_corr: marginals must be preserved, only pairing changes ───────────

def test_induce_corr_preserves_each_marginal_exactly():
    rng = np.random.default_rng(0)
    parts = [rng.normal(5, 2, 4000), rng.poisson(1.2, 4000).astype(float),
             rng.poisson(1.0, 4000).astype(float)]
    target = np.array(pb._COMBO_CORR_DEFAULT)
    out = pb._induce_corr([p.copy() for p in parts], target)
    # Each returned component is a permutation of its input: identical multiset of
    # values → identical mean, sd, and therefore identical single-stat P(over).
    for original, reordered in zip(parts, out):
        assert np.allclose(np.sort(original), np.sort(reordered)), \
            "marginal values changed — single-stat P(over) would shift"


def test_induce_corr_induces_positive_dependence():
    rng = np.random.default_rng(1)
    parts = [rng.normal(0, 1, 6000), rng.normal(0, 1, 6000), rng.normal(0, 1, 6000)]
    target = np.array(pb._COMBO_CORR_DEFAULT)  # strong positive off-diagonals
    out = pb._induce_corr(parts, target)
    corr = np.corrcoef(np.vstack(out))
    # Independent inputs → ~0 correlation; after inducing, off-diagonals move toward target.
    assert corr[0, 1] > 0.3 and corr[0, 2] > 0.2, "combo dependence was not induced"


def test_induce_corr_non_pd_target_is_a_noop():
    # A non-positive-definite target can't be Cholesky-factored → inputs returned as-is.
    parts = [np.arange(50.0), np.arange(50.0)[::-1].copy()]
    bad = [[1.0, 2.0], [2.0, 1.0]]  # off-diagonal > 1 → not PD
    out = pb._induce_corr(parts, bad)
    assert out is parts


# ── _empirical_combo_corr: shape/shrinkage/guardrails ─────────────────────────

def _logs(n, h, r, rbi):
    rng = np.random.default_rng(7)
    return [{"H": int(a), "R": int(b), "RBI": int(c)}
            for a, b, c in zip(rng.poisson(h, n), rng.poisson(r, n), rng.poisson(rbi, n))]


def test_empirical_combo_corr_none_below_min_games():
    assert pb._empirical_combo_corr(_logs(pb._COMBO_MIN_GAMES - 1, 1, 1, 1)) is None


def test_empirical_combo_corr_none_on_constant_column():
    logs = [{"H": 1, "R": 0, "RBI": 0} for _ in range(40)]  # zero-variance columns
    assert pb._empirical_combo_corr(logs) is None


def test_empirical_combo_corr_valid_matrix_properties():
    c = pb._empirical_combo_corr(_logs(60, 1.2, 0.9, 1.0))
    assert c is not None
    c = np.asarray(c)
    assert c.shape == (3, 3)
    assert np.allclose(c, c.T)                        # symmetric
    off = c[~np.eye(3, dtype=bool)]
    assert (off >= -0.95).all() and (off <= 0.95).all()   # off-diagonals clipped to ±0.95
    # Diagonal is exactly 1.0 (fixed 2026-07-27: clip now runs BEFORE fill_diagonal, so the
    # matrix is a valid correlation matrix — the Cholesky in _induce_corr is no longer warped).
    assert np.allclose(np.diag(c), 1.0)


# ── montecarlo primitives ─────────────────────────────────────────────────────

def test_prob_over_is_strict_fraction_above_line():
    s = np.array([0, 1, 2, 3, 4], dtype=float)
    assert mc.prob_over(s, 2) == pytest.approx(2 / 5)  # strictly >2 → {3,4}
    assert mc.prob_over(s, -1) == pytest.approx(1.0)
    assert mc.prob_over(s, 4) == pytest.approx(0.0)


def test_trial_counts_are_nonnegative_integers_with_right_mean():
    counts = mc.trial_counts(2.5, n=20000)
    assert (counts >= 0).all()
    assert np.allclose(counts, np.round(counts))          # integer counts
    assert counts.mean() == pytest.approx(2.5, abs=0.15)  # unbiased around expected


def test_negbinom_count_mean_and_support():
    s = mc.negbinom_count(3.0, n=20000)
    assert (s >= 0).all()
    assert s.mean() == pytest.approx(3.0, rel=0.1)
