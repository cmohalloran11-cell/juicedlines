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


def tiebreak_prob(psa: float, psb: float, a_serves_first: bool = True) -> float:
    """A wins a 7-point (win-by-2) tiebreak.

    2026-08 fix: this used to average A's serve-point and return-point win probabilities
    into ONE constant and hand it to `race_prob` (the same closed form used for a service
    game) -- but a tiebreak isn't a race at one stationary probability. It alternates
    server every 1-2 points (A serves point 1; then players alternate serving two points
    at a time), and A's true per-point win prob is `psa` on A's serve points and `1-psb`
    on B's -- two different, known values, not a single averaged guess. Averaging them and
    reusing the single-server race formula is not an approximation of that process, it's a
    different (and provably not equal) process. This is now an exact DP over the real
    alternating-server sequence.

    `a_serves_first` is a simplification shared with `set_win_prob`'s default: within one
    set, parity guarantees whoever served game 1 also serves the tiebreak first (12 games
    played before 6-6 is always even), so this only needs to match that same assumption --
    it does not track which player opens each SET's serve, which alternates set to set in
    real tennis and isn't modelled anywhere else in this engine either.

    Solved bottom-up (not naive top-down recursion): a near-even matchup can stay tied at
    depth deep enough to blow Python's recursion limit before either side reaches a 2-point
    lead, even though the probability of that path is astronomically small. `MAX_POINTS`
    bounds the table at a score no realistic tiebreak reaches (P(tied at 60-60) is
    negligible for any per-point probability bounded away from exactly 0.5); the boundary
    value there is never actually read for a real input.
    """
    def server_is_a(points_played: int) -> bool:
        if points_played == 0:
            return a_serves_first
        first_server_again = ((points_played - 1) // 2) % 2 == 1
        return a_serves_first if first_server_again else not a_serves_first

    MAX_POINTS = 60
    f: dict[tuple[int, int], float] = {}
    for total in range(2 * MAX_POINTS, -1, -1):
        lo = max(0, total - MAX_POINTS)
        hi = min(total, MAX_POINTS)
        for pa in range(lo, hi + 1):
            pb = total - pa
            if pa >= 7 and pa - pb >= 2:
                f[(pa, pb)] = 1.0
            elif pb >= 7 and pb - pa >= 2:
                f[(pa, pb)] = 0.0
            elif pa >= MAX_POINTS or pb >= MAX_POINTS:
                f[(pa, pb)] = 0.5   # unreachable boundary; never read for a real match
            else:
                p_a_wins_point = psa if server_is_a(total) else (1.0 - psb)
                f[(pa, pb)] = (p_a_wins_point * f[(pa + 1, pb)]
                               + (1.0 - p_a_wins_point) * f[(pa, pb + 1)])
    return f[(0, 0)]


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
