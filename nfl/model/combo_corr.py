"""
Intra-player combo correlation for the Rush+Rec Yards prop — the same Iman-Conover
rank-reordering the MLB engine uses for H/R/RBI and basketball/model/combo_corr.py uses for
pts/reb/ast, with the correlation MEASURED for football rather than carried over.

Why it's needed: the simulator draws rushing and receiving production from a shared snap and
team-plays draw, which correlates them a little for free, but a running back's real rushing
and receiving yards move together more than that — a game script that keeps him on the field
lifts both. Re-pairing the already-drawn marginals makes the SUM carry the real dependence
while leaving each single-stat distribution (and every P(over) built from it) bit-identical.

MEASURED 2022-2025 REG, nflverse stats_player, running backs with >=16 games in a season:
    pooled corr(rush_yards, rec_yards) = 0.2616 over 5,335 player-games (139 players)
    mean of those 139 players' OWN correlations = 0.1620 (median 0.1437)
    pooled corr(carries, receptions)   = 0.3681
The pooled number is higher than the per-player mean because it also carries between-player
differences (a pass-catching back has more of both), so a player's own measured correlation is
shrunk toward the PER-PLAYER mean, not toward the pooled one.
"""

from __future__ import annotations

import numpy as np

from ..config import cfg

_COMBO_KEYS = ("rush_yards", "rec_yards")


def pooled_default() -> float:
    return float(cfg("correlation", "rb_rush_rec_yards_player_mean", default=0.162))


def empirical_combo_corr(weeks, min_games: int | None = None):
    """2x2 target correlation matrix for (rush_yards, rec_yards) from a player's own game log,
    shrunk toward the measured per-player mean. None below the minimum-games floor — too thin
    to measure, and the caller should skip induction entirely rather than force a default onto
    a player nothing is known about (same contract as basketball's version)."""
    floor = int(min_games if min_games is not None else cfg("correlation", "combo_min_games", default=16))
    if len(weeks) < floor:
        return None
    m = np.array([[w.rush_yards, w.rec_yards] for w in weeks], dtype=float)
    if (m.std(axis=0) <= 1e-9).any():
        return None
    c = float(np.corrcoef(m, rowvar=False)[0, 1])
    if not np.isfinite(c):
        return None
    default = pooled_default()
    # Same shrinkage shape as basketball's: half weight at the floor, full-ish by ~60 games.
    w = min(1.0, len(weeks) / 60.0) * 0.5
    r = float(np.clip(w * c + (1.0 - w) * default, -0.95, 0.95))
    return np.array([[1.0, r], [r, 1.0]])


def induce_corr(parts: list[np.ndarray], target, seed: int = 12345) -> list[np.ndarray]:
    """Rank-reorder each marginal so the joint carries ~`target` correlation. Every returned
    array holds exactly the same values as its input, only re-paired, so each single-stat
    distribution is unchanged and only the combo sum moves."""
    try:
        t = np.array(target, dtype=float)
        L = np.linalg.cholesky(t)
    except (np.linalg.LinAlgError, ValueError):
        return parts
    n = len(parts[0])
    z = L @ np.random.default_rng(seed).standard_normal((len(parts), n))
    out = []
    for i, s in enumerate(parts):
        ranks = np.argsort(np.argsort(z[i]))
        out.append(np.sort(s)[ranks])
    return out
