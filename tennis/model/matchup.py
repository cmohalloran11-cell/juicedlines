"""
Matchup math: the serve-point model + closed-form game/set/match probabilities.

p_serve(A vs B) = spw_avg + (spw_A − spw_avg) − (rpw_B − rpw_avg), on the match
surface, clamped. Closed forms (win a game / tiebreak / set / match from a point-win
prob) drive the analytic match-win probability and validate the Monte-Carlo simulator.
"""

from __future__ import annotations

from functools import lru_cache
from math import comb

from ..config import cfg


def clamp_p(p: float) -> float:
    lo, hi = cfg("model", "p_serve_clamp")
    return float(min(hi, max(lo, p)))


def p_serve(rates_a, rates_b, surface, base) -> float:
    """A's probability of winning a point on A's serve vs B, on `surface`."""
    spw_a = rates_a.surface_spw.get(surface, rates_a.spw)
    rpw_b = rates_b.surface_rpw.get(surface, rates_b.rpw)
    return clamp_p(base.spw_avg + (spw_a - base.spw_avg) - (rpw_b - base.rpw_avg))


def race_prob(p: float, target: int) -> float:
    """P(win a race to `target` points, win-by-2) given per-point prob p.
    target=4 → hold a service game; target=7 → win a tiebreak."""
    p = min(0.999, max(0.001, p)); q = 1 - p
    win = 0.0
    for lose in range(target - 1):                 # opp scores 0..target-2 before we reach target
        win += comb(target - 1 + lose, lose) * (p ** target) * (q ** lose)
    deuce = comb(2 * (target - 1), target - 1) * (p ** (target - 1)) * (q ** (target - 1))
    return win + deuce * (p * p) / (p * p + q * q)


def hold_prob(p_srv: float) -> float:
    return race_prob(p_srv, 4)


def tiebreak_prob(psa: float, psb: float) -> float:
    """A wins a tiebreak; approximate iid per-point prob = mean of A-serve and A-return."""
    return race_prob((psa + (1 - psb)) / 2, 7)


def set_win_prob(ha: float, hb: float, tb_a: float, a_serves_first: bool = True) -> float:
    """A wins a set (games to 6, win-by-2, tiebreak at 6-6), exact DP over the game-model."""
    @lru_cache(maxsize=None)
    def f(ga: int, gb: int) -> float:
        if ga >= 6 and ga - gb >= 2:
            return 1.0
        if gb >= 6 and gb - ga >= 2:
            return 0.0
        if ga == 6 and gb == 6:
            return tb_a
        server_a = ((ga + gb) % 2 == 0) == a_serves_first
        hold = ha if server_a else hb
        pa_game = hold if server_a else (1 - hold)     # prob A wins this game
        return pa_game * f(ga + 1, gb) + (1 - pa_game) * f(ga, gb + 1)
    return f(0, 0)


def match_from_set(s: float, best_of: int) -> float:
    """A wins the match given per-set prob s (sets ~iid — used for the analytic check)."""
    need = best_of // 2 + 1
    return sum(comb(need - 1 + bl, bl) * (s ** need) * ((1 - s) ** bl) for bl in range(need))


def match_win_analytic(rates_a, rates_b, surface, base, best_of: int = 3) -> float:
    """Closed-form match-win prob for A — validates the simulator and blends with Elo."""
    psa = p_serve(rates_a, rates_b, surface, base)
    psb = p_serve(rates_b, rates_a, surface, base)
    ha, hb = hold_prob(psa), hold_prob(psb)
    tb_a = tiebreak_prob(psa, psb)
    s = set_win_prob(ha, hb, tb_a)
    return match_from_set(s, best_of)
