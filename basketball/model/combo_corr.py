"""
Combo-stat correlation for basketball (PRA and similar combos) — same Iman-Conover
technique the MLB engine uses for H/R/RBI (see projector_bridge.py's
_empirical_combo_corr/_induce_corr), generalized here for basketball's pts/reb/ast.

Why this exists: the sim engine draws each base stat from a per-sim NegBin conditioned
on SHARED minutes/pace/possessions, which does induce some real correlation for free (a
big-minutes night lifts every stat together) — but measured on live sims that alone only
produces ~0.05-0.08 correlation between pts and reb, far below what real box scores show.
This measures the REAL per-player dependence from their own game log (falling back to a
pooled empirical default when a player's own sample is thin) and re-pairs the simulated
marginals to carry that correlation — exact same marginals, only the pairing changes.

Pooled default measured 2026-08 from 15 active WNBA players' own ESPN game logs (402
games total, real box scores, plain numpy corrcoef — not fabricated):
    corr(pts, reb) = 0.474, corr(pts, ast) = 0.338, corr(reb, ast) = 0.157
"""
from __future__ import annotations

import numpy as np

_COMBO_KEYS = ("pts", "reb", "ast")
_COMBO_CORR_DEFAULT = ((1.0, 0.474, 0.338),
                       (0.474, 1.0, 0.157),
                       (0.338, 0.157, 1.0))
_COMBO_MIN_GAMES = 15


def empirical_combo_corr(games: list, keys: tuple[str, ...] = _COMBO_KEYS,
                         min_games: int = _COMBO_MIN_GAMES):
    """Per-player corr matrix for `keys` (default pts/reb/ast) from the player's own
    game log. None below `min_games` — too thin to measure, caller should skip induction
    entirely rather than force the pooled default onto every player."""
    if len(games) < min_games:
        return None
    try:
        m = np.array([[float(getattr(g, k, 0.0) or 0.0) for k in keys] for g in games],
                     dtype=float)
        if (m.std(axis=0) <= 1e-9).any():          # a constant column → corr undefined
            return None
        c = np.corrcoef(m, rowvar=False)
        if not np.isfinite(c).all():
            return None
        # shrink toward the pooled default: even 30-40 games of a noisy 3x3 is a small sample
        d = np.array(_COMBO_CORR_DEFAULT)
        w = min(1.0, len(games) / 60.0) * 0.5
        c = w * c + (1 - w) * d
        c = np.clip(c, -0.95, 0.95)
        np.fill_diagonal(c, 1.0)
        return c
    except Exception:
        return None


def induce_corr(parts: list[np.ndarray], target, seed: int = 12345) -> list[np.ndarray]:
    """Rank-reorder each marginal so the joint carries ~`target` correlation. Iman-Conover
    style: every returned array holds EXACTLY the same values as its input (just
    re-ordered), so each single-stat distribution — and every P(over) built from it — is
    bit-identical. Only how the stats pair up within a single sim trial changes, which is
    exactly what a combo sum (PRA, etc.) depends on."""
    try:
        t = np.array(target, dtype=float)
        L = np.linalg.cholesky(t)
    except Exception:
        return parts                                # not positive-definite → leave alone
    n = len(parts[0])
    z = L @ np.random.default_rng(seed).standard_normal((len(parts), n))
    out = []
    for i, s in enumerate(parts):
        ranks = np.argsort(np.argsort(z[i]))        # rank of each sim under the target copula
        out.append(np.sort(s)[ranks])               # same values, re-paired to those ranks
    return out
