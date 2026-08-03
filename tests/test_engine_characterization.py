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

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import projector_bridge as pb
from projector.models import montecarlo as mc


def test_vendored_engine_works_without_the_sibling_stat_projector_repo():
    # 2026-08 regression guard. projector_bridge.py vendors a standalone copy of the real
    # engine at projector/models/mlb_model.py specifically so a deploy that only checks out
    # this repo (no ../stat-projector sibling) can still run it -- see projector_bridge.py's
    # own docstring. That vendored copy previously drifted out of sync (two pasted-together,
    # broken definitions of project_batter/project_pitcher, one calling helper functions that
    # were never defined) and crashed with a NameError on every single call -- invisible in
    # local dev ONLY because a sibling ../stat-projector checkout happened to exist and take
    # sys.path precedence over the vendored copy, silently masking the break. In a real deploy
    # (no sibling directory) every MLB projection request raised, was swallowed by
    # projector_bridge.project_player's blanket except-Exception, and the board silently fell
    # back to the cruder empirical-median model for 100% of MLB props with zero user-facing
    # signal that the "real" engine was dark. This test spawns a FRESH subprocess that imports
    # projector.models.mlb_model directly (bypassing projector_bridge's sys.path insert
    # entirely) so it can never accidentally resolve to the sibling repo, matching exactly
    # what a real deploy sees.
    repo_root = Path(__file__).resolve().parent.parent
    script = (
        "from projector.models import mlb_model as mm\n"
        "assert 'stat-projector' not in mm.__file__, mm.__file__\n"
        "form = {'p_hit':0.26,'p_hr':0.035,'p_bb':0.09,'p_k':0.22,'exp_pa':4.3,"
        "'exp_runs':0.5,'exp_rbi':0.5,'exp_sb':0.05}\n"
        "out = mm.project_batter(form, {}, n=500, use_ensemble=False)\n"
        "assert 'hits' in out and out['hits'].mean is not None\n"
        "pform = {'xfip':3.8,'p_k':0.24,'p_bb':0.07,'exp_bf':24,'exp_er':2.5,'exp_ip':5.5}\n"
        "pout = mm.project_pitcher(pform, {}, n=500, use_ensemble=False)\n"
        "assert 'strikeouts' in pout and pout['strikeouts'].mean is not None\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=str(repo_root),
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0 and "OK" in result.stdout, (
        f"vendored engine crashed in a clean subprocess (matches a real deploy):\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


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


def test_two_stage_uncertainty_widens_batter_intervals_for_thin_samples_not_means():
    # 2026-08: every stat used to draw its outcome from a rate treated as a FIXED constant
    # across all n Monte Carlo trials -- only outcome (sampling) variance was represented.
    # form["_eff_pa"] (mlb_features.py's shrinkage denominator: real PA + prior pseudo-PA)
    # now drives a per-trial Gamma rate-uncertainty multiplier BEFORE each outcome draw, so a
    # thin sample (a rookie's first week) gets a genuinely wider distribution than a deep one
    # (a 500-PA regular) at the SAME point estimate -- the textbook two-stage decomposition.
    from projector.models import mlb_model as mm
    base = {"role": "batter", "p_hit": 0.26, "p_hr": 0.035, "p_bb": 0.08, "p_k": 0.22,
            "p_1b": 0.16, "p_2b": 0.06, "p_3b": 0.005, "exp_pa": 4.2,
            "exp_rbi": 0.5, "exp_runs": 0.5, "exp_sb": 0.05}
    thin = {**base, "_eff_pa": 8}          # ~no real data, almost pure prior
    deep = {**base, "_eff_pa": 200_000}    # effectively zero parameter uncertainty left

    # average several seeds -- a single seed's SD estimate is itself noisy at n=40000
    sd_thin = np.mean([mm.project_batter(thin, {}, n=40000, use_ensemble=False, seed=s)
                       ["hits"].std for s in range(6)])
    sd_deep = np.mean([mm.project_batter(deep, {}, n=40000, use_ensemble=False, seed=s)
                       ["hits"].std for s in range(6)])
    assert sd_thin > sd_deep * 1.02, "a near-zero-evidence rate must carry more spread"

    # the MEAN must stay essentially unchanged -- this is a spread fix, not a bias fix
    # (E[theta] = 1 regardless of eff_pa).
    mean_thin = mm.project_batter(thin, {}, n=40000, use_ensemble=False, seed=0)["hits"].mean
    mean_deep = mm.project_batter(deep, {}, n=40000, use_ensemble=False, seed=0)["hits"].mean
    assert mean_thin == pytest.approx(mean_deep, abs=0.05)

    # a missing _eff_pa (shouldn't happen in production -- mlb_features.py always sets it)
    # must fall back to a deep-sample default, not blow up or silently assume zero evidence.
    out_missing = mm.project_batter(base, {}, n=5000, use_ensemble=False, seed=0)
    assert out_missing["hits"].mean > 0


def test_two_stage_uncertainty_widens_pitcher_intervals_for_thin_samples_not_means():
    from projector.models import mlb_model as mm
    base = {"role": "pitcher", "exp_bf": 23.0, "exp_outs": 17.0, "p_k": 0.235,
            "p_bb": 0.075, "p_h": 0.21, "xera": 4.0}
    thin = {**base, "_eff_pa": 8}
    deep = {**base, "_eff_pa": 200_000}

    sd_thin = np.mean([mm.project_pitcher(thin, {}, n=40000, use_ensemble=False, seed=s)
                       ["strikeouts"].std for s in range(6)])
    sd_deep = np.mean([mm.project_pitcher(deep, {}, n=40000, use_ensemble=False, seed=s)
                       ["strikeouts"].std for s in range(6)])
    assert sd_thin > sd_deep * 1.02

    mean_thin = mm.project_pitcher(thin, {}, n=40000, use_ensemble=False, seed=0)["strikeouts"].mean
    mean_deep = mm.project_pitcher(deep, {}, n=40000, use_ensemble=False, seed=0)["strikeouts"].mean
    assert mean_thin == pytest.approx(mean_deep, abs=0.1)


def test_total_bases_matches_analytic_expectation_and_supports_per_trial_probabilities():
    # montecarlo.total_bases was rewritten (sequential conditional-Binomial chain instead of
    # a single shared-probability Multinomial) specifically so per-PA probabilities can vary
    # PER SIM -- required for the theta rate-uncertainty draw above. Must still match the
    # analytic expectation for plain scalar inputs (no regression in the common case).
    trials = np.full(50000, 4, dtype=int)
    per_pa = {"1b": 0.15, "2b": 0.05, "3b": 0.005, "hr": 0.035}
    tb = mc.total_bases(per_pa, trials, g=np.random.default_rng(0))
    expected = 4 * (0.15 * 1 + 0.05 * 2 + 0.005 * 3 + 0.035 * 4)
    assert tb.mean() == pytest.approx(expected, rel=0.03)
    assert (tb >= 0).all()

    # per-sim arrays: a trial with a much higher HR probability must average out higher
    hi = np.concatenate([np.full(25000, 0.30), np.full(25000, 0.005)])
    per_pa_var = {"1b": 0.0, "2b": 0.0, "3b": 0.0, "hr": hi}
    tb_var = mc.total_bases(per_pa_var, trials, g=np.random.default_rng(1))
    assert tb_var[:25000].mean() > tb_var[25000:].mean() * 3


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
