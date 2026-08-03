"""
montecarlo.py — Monte-Carlo samplers for single-game stat distributions (scipy).

Each stat is simulated from a mechanistic model rather than assuming a Normal:
  • per-trial binary events (a hit, HR, K, BB)      -> Binomial(trials, p)
  • hit-type composition (for total bases)          -> Multinomial per PA
  • contextual counts (RBI, runs, ER, SB)           -> Negative-Binomial(mean, k)
  • innings / outs recorded                          -> rounded Normal, clipped

`N_SIMS = 10_000` by default (per spec).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

N_SIMS = 10_000
_RNG = np.random.default_rng(7)


def rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed) if seed is not None else _RNG


# ── trial counts (plate appearances / batters faced) ─────────────────────────

def trial_counts(expected: float, n: int = N_SIMS, lo: int = 0, hi: int | None = None,
                 g: np.random.Generator | None = None) -> np.ndarray:
    """Random PA/BF per game ~ Poisson(expected), clipped to a sane range."""
    g = g or _RNG
    arr = g.poisson(max(0.01, expected), n)
    if hi is None:
        hi = int(expected * 2 + 6)
    return np.clip(arr, lo, hi)


# ── binary per-trial events: hits, HR, BB, K, hits/walks allowed ─────────────

def binomial_event(p_event, trials: np.ndarray,
                   g: np.random.Generator | None = None) -> np.ndarray:
    """`p_event` may be a scalar (one probability for every sim) or an array the same
    shape as `trials` (a PER-SIM probability — e.g. after a parameter-uncertainty
    multiplier has been applied per trial). numpy's binomial broadcasts either way."""
    g = g or _RNG
    p = np.clip(p_event, 0.0, 1.0)
    return g.binomial(trials, p)


# ── total bases via hit-type multinomial ─────────────────────────────────────

def total_bases(per_pa: dict, trials: np.ndarray,
                g: np.random.Generator | None = None) -> np.ndarray:
    """
    per_pa: per-PA probabilities {'1b','2b','3b','hr'} (rest = outs/no-TB). Each may be
    a scalar (shared across every sim) OR an array shaped like `trials` (a PER-SIM
    probability, e.g. after a parameter-uncertainty multiplier). Total bases =
    1*1B + 2*2B + 3*3B + 4*HR.

    Implemented as a sequential chain of conditional Binomial draws (HR, then 3B among
    the PA that weren't a HR, then 2B among what's left, then 1B among what's left) —
    mathematically identical to a single Multinomial draw, but vectorises correctly
    when the per-PA probabilities vary sim-to-sim (a plain Multinomial call only takes
    one shared probability vector per size, so it can't represent that).
    """
    g = g or _RNG
    n = trials.shape[0]
    p1 = np.clip(np.broadcast_to(per_pa.get("1b", 0.0), (n,)).astype(float), 0.0, None)
    p2 = np.clip(np.broadcast_to(per_pa.get("2b", 0.0), (n,)).astype(float), 0.0, None)
    p3 = np.clip(np.broadcast_to(per_pa.get("3b", 0.0), (n,)).astype(float), 0.0, None)
    p4 = np.clip(np.broadcast_to(per_pa.get("hr", 0.0), (n,)).astype(float), 0.0, None)
    total = p1 + p2 + p3 + p4
    over = total > 1.0
    if np.any(over):                      # renormalize only the sims where probs sum > 1
        scale = np.where(over, 1.0 / np.maximum(total, 1e-9), 1.0)
        p1, p2, p3, p4 = p1 * scale, p2 * scale, p3 * scale, p4 * scale

    hr = g.binomial(trials, p4)
    rem1 = trials - hr
    p3_cond = np.clip(p3 / np.clip(1.0 - p4, 1e-9, None), 0.0, 1.0)
    triples = g.binomial(rem1, p3_cond)
    rem2 = rem1 - triples
    p2_cond = np.clip(p2 / np.clip(1.0 - p4 - p3, 1e-9, None), 0.0, 1.0)
    doubles = g.binomial(rem2, p2_cond)
    rem3 = rem2 - doubles
    p1_cond = np.clip(p1 / np.clip(1.0 - p4 - p3 - p2, 1e-9, None), 0.0, 1.0)
    singles = g.binomial(rem3, p1_cond)

    return (singles * 1 + doubles * 2 + triples * 3 + hr * 4).astype(float)


# ── contextual counts: RBI, runs, ER, SB ─────────────────────────────────────

def negbinom_count(mean, dispersion: float = 0.55, n: int = N_SIMS,
                   g: np.random.Generator | None = None) -> np.ndarray:
    """
    Negative-Binomial parameterised by mean and a dispersion knob (var/mean-1).
    Captures overdispersion of contextual counts. dispersion→0 ⇒ Poisson.

    `mean` may be a scalar (shared across every sim, the original behaviour — draws
    exactly `n` samples) or an array of length `n` (a PER-SIM mean, e.g. after a
    parameter-uncertainty multiplier — one draw per array element, `n` is then ignored
    since the output shape is already determined by the array).
    """
    g = g or _RNG
    arr = np.atleast_1d(np.asarray(mean, dtype=float))
    scalar_input = arr.shape == (1,) and np.isscalar(mean)
    arr = np.clip(arr, 1e-6, None)
    if dispersion <= 1e-6:
        return g.poisson(arr, n) if scalar_input else g.poisson(arr)
    var = arr * (1.0 + dispersion * arr)
    r = arr * arr / np.clip(var - arr, 1e-9, None)  # NB 'number of failures'
    p = r / (r + arr)                               # success prob
    return g.negative_binomial(r, p, n) if scalar_input else g.negative_binomial(r, p)


# ── innings pitched / outs recorded ──────────────────────────────────────────

def outs_recorded(expected_outs: float, sd: float = 4.0, n: int = N_SIMS,
                  g: np.random.Generator | None = None) -> np.ndarray:
    g = g or _RNG
    arr = np.round(g.normal(expected_outs, sd, n))
    return np.clip(arr, 0, 27)


# ── soccer: goals from per-shot xG (Poisson-binomial via Poisson approx) ─────

def goals_from_xg(total_xg: float, n: int = N_SIMS,
                  g: np.random.Generator | None = None) -> np.ndarray:
    """Goals ~ Poisson(total expected goals). Good approx to a Poisson-binomial."""
    g = g or _RNG
    return g.poisson(max(0.0, total_xg), n)


def rate_per_match(expected: float, dispersion: float = 0.3, n: int = N_SIMS,
                   g: np.random.Generator | None = None) -> np.ndarray:
    """Generic per-match count (shots, key passes, saves) with mild overdispersion."""
    return negbinom_count(expected, dispersion=dispersion, n=n, g=g)


def prob_over(samples: np.ndarray, line: float) -> float:
    return float(np.mean(np.asarray(samples) > line))
