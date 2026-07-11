"""
Monte-Carlo match simulator.

Simulates the match at the GAME level (each service game held with its closed-form
hold probability, tiebreaks resolved with their tiebreak probability) — fully
vectorized across N sims, so a full slate runs in well under a second. Serve counts
come out of the sim (service games × points-per-service-game); aces and double faults
are then Binomial draws on those serve points. Every count prop falls out of the
resulting distributions; match-win prob is blended with Elo by the caller.

Format is a per-match parameter: best-of-3/5 and whether the deciding set uses a
tiebreak (standard) or is played out (advantage).
"""

from __future__ import annotations

import numpy as np

from ..config import cfg
from ..model.matchup import p_serve, hold_prob, tiebreak_prob


def simulate(rates_a, rates_b, surface, base, best_of=3,
             final_set_advantage=False, n=None, rng=None) -> dict:
    n = n or cfg("model", "n_sims")
    rng = rng or np.random.default_rng()

    psa = p_serve(rates_a, rates_b, surface, base)
    psb = p_serve(rates_b, rates_a, surface, base)
    ha, hb = hold_prob(psa), hold_prob(psb)
    tb_a = tiebreak_prob(psa, psb)                    # A wins a tiebreak
    need = best_of // 2 + 1

    ga = np.zeros(n, int); gb = np.zeros(n, int)      # games in current set
    sa = np.zeros(n, int); sb = np.zeros(n, int)      # sets won
    tga = np.zeros(n, int); tgb = np.zeros(n, int)    # total games
    svg_a = np.zeros(n, int); svg_b = np.zeros(n, int)  # service games served
    brk_a = np.zeros(n, int); brk_b = np.zeros(n, int)  # breaks OF serve (A broke B / B broke A)
    done = np.zeros(n, bool)
    winner = np.zeros(n, int)                          # 0 = A, 1 = B

    for _ in range(400):                              # safety cap (advantage sets)
        active = ~done
        if not active.any():
            break
        server_a = ((tga + tgb) % 2 == 0)
        holds = rng.random(n) < np.where(server_a, ha, hb)
        a_wins_game = (server_a & holds) | (~server_a & ~holds)

        brk_a += (active & ~server_a & ~holds).astype(int)   # B served and lost
        brk_b += (active & server_a & ~holds).astype(int)    # A served and lost
        ga += (active & a_wins_game); gb += (active & ~a_wins_game)
        tga += (active & a_wins_game); tgb += (active & ~a_wins_game)
        svg_a += (active & server_a); svg_b += (active & ~server_a)

        is_final = (sa + sb) == (best_of - 1)
        # tiebreak at 6-6 (unless it's an advantage deciding set)
        tb = active & (ga == 6) & (gb == 6) & ~(is_final & final_set_advantage)
        if tb.any():
            a_tb = tb & (rng.random(n) < tb_a)
            b_tb = tb & ~(rng.random(n) < tb_a) if False else (tb & ~a_tb)
            ga[a_tb] += 1; tga[a_tb] += 1; sa[a_tb] += 1
            gb[b_tb] += 1; tgb[b_tb] += 1; sb[b_tb] += 1
            ga[tb] = 0; gb[tb] = 0

        set_a = active & (ga >= 6) & (ga - gb >= 2)
        set_b = active & (gb >= 6) & (gb - ga >= 2)
        sa += set_a; sb += set_b
        reset = set_a | set_b
        ga[reset] = 0; gb[reset] = 0

        newly = active & ((sa >= need) | (sb >= need))
        winner[newly & (sb >= need)] = 1
        done |= newly

    # serve points → aces / double faults (Binomial on realized serve points)
    sp_a = np.rint(svg_a * rates_a.pts_per_svgame).astype(int)
    sp_b = np.rint(svg_b * rates_b.pts_per_svgame).astype(int)
    aces_a = rng.binomial(np.maximum(0, sp_a), min(0.4, rates_a.ace_rate))
    aces_b = rng.binomial(np.maximum(0, sp_b), min(0.4, rates_b.ace_rate))
    dfs_a = rng.binomial(np.maximum(0, sp_a), min(0.3, rates_a.df_rate))
    dfs_b = rng.binomial(np.maximum(0, sp_b), min(0.3, rates_b.df_rate))

    return {
        "n": n, "p_serve_a": round(psa, 4), "p_serve_b": round(psb, 4),
        "hold_a": round(ha, 4), "hold_b": round(hb, 4),
        "winner": winner,                              # 0=A,1=B
        "games_a": tga, "games_b": tgb, "total_games": tga + tgb,
        "sets_a": sa, "sets_b": sb, "total_sets": sa + sb,
        "aces_a": aces_a, "aces_b": aces_b, "dfs_a": dfs_a, "dfs_b": dfs_b,
        "breaks_a": brk_a, "breaks_b": brk_b,
    }


def summary(arr) -> dict:
    a = np.asarray(arr, float)
    return {"mean": round(float(a.mean()), 3),
            "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)),
            "sd": round(float(a.std()), 3)}


def prob_over(arr, line: float) -> float:
    return round(float((np.asarray(arr) > line).mean()), 4)
