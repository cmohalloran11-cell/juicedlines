"""
Orchestrator — fit/cache the model per tour, project a match, read any market.

    from tennis import projections as P
    r = P.project_match("ATP", "Novak Djokovic", "Carlos Alcaraz", "Grass", best_of=5)
    P.market_prob(r, "Novak Djokovic", "Aces", 6.5, "over")

Rates + Elo are fit once per tour (Sackmann mirror is static) and cached. Players
not found fall back to a tour-baseline "unknown" with effective-sample 0 → lowest
confidence, so thin-sample / qualifier matches surface with wide intervals and can
be gated rather than skipped.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import numpy as np

from .config import cfg
from .data import get_history
from .model import rates as R
from .model.matchup import match_win_analytic
from .model.elo import EloModel
from .sim.engine import simulate, summary, prob_over

_FIT_YEARS = list(range(2016, 2023))     # mirror coverage; adapter skips missing years


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


@lru_cache(maxsize=4)
def _model(tour: str):
    matches = get_history().player_matches(tour, _FIT_YEARS)
    base, by_id = R.fit(matches, tour)
    elo = EloModel(tour).fit(matches)
    name_idx, last_idx = {}, {}
    for pid, pr in by_id.items():
        nm = _norm(pr.player)
        if nm:
            name_idx[nm] = pid
            last_idx.setdefault(nm.split()[-1], []).append(pid)
    return {"base": base, "by_id": by_id, "elo": elo, "name": name_idx, "last": last_idx}


def _baseline_player(base, tour: str) -> R.PlayerRates:
    return R.PlayerRates(player_id="", player="(unknown)", tour=tour, spw=base.spw_avg,
                         rpw=base.rpw_avg, ace_rate=base.ace_rate_avg, df_rate=base.df_rate_avg,
                         pts_per_svgame=base.pts_per_svgame_avg, n_matches=0)


def resolve(tour: str, name: str):
    """Name → PlayerRates (+ pid). Falls back to a baseline unknown player."""
    m = _model(tour)
    nm = _norm(name)
    pid = m["name"].get(nm)
    if not pid and nm:
        last = nm.split()[-1]
        cands = m["last"].get(last, [])
        if len(cands) == 1:
            pid = cands[0]
    if pid:
        return m["by_id"][pid], pid
    return _baseline_player(m["base"], tour), ""


def _confidence(eff_n: int) -> str:
    c = cfg("confidence")
    return "high" if eff_n >= c["high"] else "medium" if eff_n >= c["medium"] else "low"


def project_match(tour: str, player_a: str, player_b: str, surface: str,
                  best_of: int = 3, final_set_advantage: bool = False, n=None) -> dict:
    m = _model(tour)
    base, elo = m["base"], m["elo"]
    ra, pa_id = resolve(tour, player_a)
    rb, pb_id = resolve(tour, player_b)
    surface = surface if surface in ra.surface_spw or True else surface

    sim = simulate(ra, rb, surface, base, best_of=best_of,
                   final_set_advantage=final_set_advantage, n=n)
    win_sim = float((sim["winner"] == 0).mean())
    win_elo = elo.win_prob(pa_id, pb_id, surface) if (pa_id or pb_id) else 0.5
    blend = cfg("model", "match_prob_blend")
    win_a = blend * win_sim + (1 - blend) * win_elo

    eff_n = min(ra.eff_matches(surface), rb.eff_matches(surface))
    dist = {"total_games": sim["total_games"], "total_sets": sim["total_sets"],
            ra.player: {"aces": sim["aces_a"], "dfs": sim["dfs_a"], "games": sim["games_a"],
                        "sets": sim["sets_a"], "breaks": sim["breaks_a"]},
            rb.player: {"aces": sim["aces_b"], "dfs": sim["dfs_b"], "games": sim["games_b"],
                        "sets": sim["sets_b"], "breaks": sim["breaks_b"]}}
    return {
        "tour": tour, "surface": surface, "best_of": best_of,
        "player_a": ra.player, "player_b": rb.player,
        "win_prob_a": round(win_a, 4), "win_prob_b": round(1 - win_a, 4),
        "win_prob_a_sim": round(win_sim, 4), "win_prob_a_elo": round(win_elo, 4),
        "win_prob_a_analytic": round(match_win_analytic(ra, rb, surface, base, best_of), 4),
        "eff_matches": eff_n, "confidence": _confidence(eff_n),
        "markets": {
            "total_games": summary(sim["total_games"]),
            "total_sets": summary(sim["total_sets"]),
            ra.player: {"aces": summary(sim["aces_a"]), "dfs": summary(sim["dfs_a"]),
                        "games": summary(sim["games_a"])},
            rb.player: {"aces": summary(sim["aces_b"]), "dfs": summary(sim["dfs_b"]),
                        "games": summary(sim["games_b"])},
        },
        "_dist": dist,
    }


# book stat-type label (normalized) → (dist key, per-player?). Matches the real
# Underdog/PrizePicks tennis labels: games_played (match total), games_won (player),
# sets_played/sets_won, aces, double_faults, breakpoints_won. Per-set (period_1_*)
# and tie_breakers_played aren't modeled yet → skipped (fall through to no projection).
def _resolve_market(label: str):
    s = _norm(label)
    if "period" in s or "tie breaker" in s or "tiebreaker" in s or "1h" in s or "2h" in s:
        return (None, None)
    if "ace" in s:
        return ("aces", True)
    if "double" in s:
        return ("dfs", True)
    if "breakpoint" in s or "break point" in s:
        return ("breaks", True)
    if "game" in s:
        # "won" is a PLAYER's games won (even in "Total Games Won"); "played"/"total"
        # without "won" is the MATCH total. Check "won" first to avoid mis-mapping.
        if "won" in s:
            return ("games", True)
        return ("total_games", False) if ("played" in s or "total" in s) else ("games", True)
    if "set" in s:
        if "won" in s:
            return ("sets", True)
        return ("total_sets", False) if ("played" in s or "total" in s) else ("sets", True)
    return (None, None)


def market_dist(result: dict, player: str, label: str):
    key, per_player = _resolve_market(label)
    if key is None:
        return None
    if per_player:
        pm = result["_dist"].get(player)
        if pm is None:
            nm = _norm(player)
            pm = next((v for k, v in result["_dist"].items()
                       if isinstance(v, dict) and _norm(k) == nm), None)
        return pm.get(key) if pm else None
    return result["_dist"].get(key)


def market_prob(result: dict, player: str, label: str, line: float, side: str = "over"):
    arr = market_dist(result, player, label)
    if arr is None or line is None:
        return None
    p = prob_over(arr, float(line))
    return p if side == "over" else round(1 - p, 4)
