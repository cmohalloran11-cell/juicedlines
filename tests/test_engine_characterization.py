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


# ── plate_appearances: exposing the engine's own per-sim PA draw as a stat ─────
# The batter sim already draws a per-simulation trial count (Poisson around exp_pa)
# to size every OTHER stat's binomial/multinomial draws — this just returns that
# same array as its own Projection instead of throwing it away, so "Plate
# Appearances" props (previously unprojected — no engine output existed for them)
# get a real distribution.

def test_plate_appearances_is_a_real_projection_matching_exp_pa():
    from projector.models import mlb_model as mm
    form = {"exp_pa": 4.2, "p_hit": 0.25, "p_hr": 0.04, "p_bb": 0.09, "p_k": 0.22,
            "role": "batter"}
    out = mm.project_batter(form, {}, n=20000, use_ensemble=False)
    assert "plate_appearances" in out
    pa = out["plate_appearances"]
    assert pa.mean == pytest.approx(4.2, rel=0.1)
    assert pa.samples is not None and (pa.samples >= 0).all()


def test_runs_allowed_resolves_to_its_own_stat_not_earned_runs():
    # Regression: "Runs Allowed" used to be a substring of the "earned runs allowed" alias
    # key, so _best_match silently resolved it to the (systematically lower) earned-runs
    # distribution — a category error, not a calibration issue. It must now hit an EXACT
    # match on "runs allowed" before any fuzzy search runs.
    assert pb._PIT["runs allowed"] == "runs_allowed"
    assert pb._best_match(pb._PIT, "runs allowed") != "earned_runs" or \
        pb._PIT.get("runs allowed") == "runs_allowed"   # exact entry wins regardless


def test_runs_allowed_is_scaled_above_earned_runs():
    from projector.models import mlb_model as mm
    form = {"exp_bf": 23.0, "exp_outs": 17.0, "p_k": 0.235, "p_bb": 0.075, "p_h": 0.21,
            "xera": 4.0, "role": "pitcher"}
    out = mm.project_pitcher(form, {}, n=20000, use_ensemble=False)
    assert "runs_allowed" in out
    # scaled by the measured league unearned-run ratio (~1.09) — must exceed earned runs,
    # never equal it (that would mean the scale silently regressed to a no-op).
    assert out["runs_allowed"].mean > out["earned_runs"].mean
    assert out["runs_allowed"].mean == pytest.approx(out["earned_runs"].mean * 1.0904, rel=0.02)


def test_median_direction_always_agrees_with_prob_over():
    # 2026-07-29 projection-realism pass: `projection` is the MEAN — informative, and
    # deliberately allowed to diverge in direction from P(over) on a skewed stat (that's
    # exactly what makes a backup catcher's projection look like a real number instead of a
    # flat 0). The mathematical guarantee lives in `median` instead: median > line can never
    # coincide with prob_over < 0.5, for ANY sample array/line combination, because median
    # and prob_over are computed from the identical sample array.
    rng = np.random.default_rng(1)
    for line in (0.5, 1.5, 2.5, 6.5):
        for _ in range(20):
            # negative-binomial-like: skewed, integer-valued, mean pulled up by a long tail
            samples = rng.negative_binomial(2, 0.4, 3000).astype(float)
            from projector.models.base import Projection
            proj_obj = Projection(stat="x", mean=float(samples.mean()),
                                   median=float(np.median(samples)), p25=0.0, p75=0.0,
                                   floor=0.0, ceiling=0.0, std=1.0, samples=samples)
            out = pb._payload(proj_obj, line)
            median, prob_over = out["median"], out["prob_over"]
            if median > line:
                assert prob_over >= 0.5, (median, line, prob_over)
            elif median < line:
                assert prob_over <= 0.5, (median, line, prob_over)


def test_plate_appearances_resolves_through_for_stat():
    # for_stat's alias table must map the human prop label to the engine key end to end.
    from projector.models.base import Projection
    import numpy as np
    samples = np.random.default_rng(0).poisson(4.2, 5000).astype(float)
    projs = {"plate_appearances": Projection(
        stat="plate_appearances", mean=4.2, median=4.0, p25=3.0, p75=5.0,
        floor=2.0, ceiling=7.0, std=2.0, samples=samples)}
    out = pb.for_stat(projs, "Plate Appearances", 3.5, is_pitcher=False)
    # projection is the MEAN (2026-07-29 projection-realism pass) — median is its own field.
    assert out is not None and out["projection"] == pytest.approx(4.2, abs=0.01)
    assert out["median"] == pytest.approx(4.0, abs=0.01)
    out2 = pb.for_stat(projs, "PA", 3.5, is_pitcher=False)   # the short alias too
    assert out2 is not None
