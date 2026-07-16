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

def binomial_event(p_event: float, trials: np.ndarray,
                   g: np.random.Generator | None = None) -> np.ndarray:
    g = g or _RNG
    p = float(np.clip(p_event, 0.0, 1.0))
    return g.binomial(trials, p)


# ── total bases via hit-type multinomial ─────────────────────────────────────

def total_bases(per_pa: dict, trials: np.ndarray,
                g: np.random.Generator | None = None) -> np.ndarray:
    """
    per_pa: per-PA probabilities {'1b','2b','3b','hr'} (rest = outs/no-TB).
    Total bases = 1*1B + 2*2B + 3*3B + 4*HR.
    """
    g = g or _RNG
    p1, p2, p3, p4 = (max(0.0, per_pa.get(k, 0.0)) for k in ("1b", "2b", "3b", "hr"))
    p_out = max(1e-9, 1.0 - (p1 + p2 + p3 + p4))
    probs = np.array([p_out, p1, p2, p3, p4])
    probs = probs / probs.sum()
    out = np.zeros(trials.shape[0])
    # group sims by identical trial count for vectorised multinomial draws
    for t in np.unique(trials):
        idx = trials == t
        if t == 0:
            continue
        draws = g.multinomial(int(t), probs, size=int(idx.sum()))  # cols: out,1b,2b,3b,hr
        out[idx] = draws[:, 1] * 1 + draws[:, 2] * 2 + draws[:, 3] * 3 + draws[:, 4] * 4
    return out


# ── contextual counts: RBI, runs, ER, SB ─────────────────────────────────────

def negbinom_count(mean: float, dispersion: float = 0.55, n: int = N_SIMS,
                   g: np.random.Generator | None = None) -> np.ndarray:
    """
    Negative-Binomial parameterised by mean and a dispersion knob (var/mean-1).
    Captures overdispersion of contextual counts. dispersion→0 ⇒ Poisson.
    """
    g = g or _RNG
    mean = max(1e-6, float(mean))
    if dispersion <= 1e-6:
        return g.poisson(mean, n)
    var = mean * (1.0 + dispersion * mean)
    r = mean * mean / max(1e-9, var - mean)     # NB 'number of failures'
    p = r / (r + mean)                          # success prob
    return g.negative_binomial(r, p, n)


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
