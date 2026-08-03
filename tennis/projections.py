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
import time
import unicodedata
from dataclasses import dataclass, replace

import numpy as np

from .config import cfg
from .data import get_history
from .data import espn as _espn
from .model import rates as R
from .model.matchup import match_win_analytic
from .model.elo import EloModel
from .sim.engine import simulate, summary, prob_over

_FIT_YEARS = list(range(2016, 2023))     # mirror coverage; adapter skips missing years
_MODEL_TTL = 3 * 3600     # re-pull ESPN's live layer + reseed Elo this often
_MODEL_CACHE: dict = {}


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


@dataclass
class _LiveResult:
    """Minimal record EloModel.fit() needs from an ESPN completed match — winner's
    perspective, one row per match (opp carries the loser)."""
    player: str
    opp: str
    surface: str
    date: str
    won: bool = True


def _live_elo_results(tour: str, days_back: int = 180) -> list[_LiveResult]:
    """Recent ESPN results, oldest first, ready for EloModel.fit(). Best-effort — an
    ESPN outage just means the live layer doesn't extend this cycle, not a crash."""
    try:
        matches = _espn.completed_matches(tour, days_back=days_back)
    except Exception:
        return []
    out = [
        _LiveResult(player=(m["a_name"] if m["a_winner"] else m["b_name"]),
                    opp=(m["b_name"] if m["a_winner"] else m["a_name"]),
                    surface=m["surface"], date=m["date"])
        for m in matches if m.get("a_name") and m.get("b_name")
    ]
    out.sort(key=lambda r: r.date)
    return out


def _model(tour: str):
    """Fit/cache per tour. The Sackmann mirror (`_FIT_YEARS`) is static, so that half is
    cheap to refit; the live half re-pulls ESPN every `_MODEL_TTL` so a returning
    veteran's Elo keeps moving and a mirror-absent current player (Sinner, Alcaraz — the
    mirror predates both) builds a real rating from actual current results instead of
    being stuck at a flat "average tour player" prior forever."""
    hit = _MODEL_CACHE.get(tour)
    if hit and time.time() - hit[0] < _MODEL_TTL:
        return hit[1]
    matches = get_history().player_matches(tour, _FIT_YEARS)
    base, by_id = R.fit(matches, tour)
    elo = EloModel(tour).fit(matches)
    live_n = 0
    try:
        live = _live_elo_results(tour)
        elo.fit(live)
        live_n = len(live)
    except Exception:
        pass
    try:
        elo.seed_from_rank(_espn.rankings(tour))
    except Exception:
        pass
    name_idx, last_idx = {}, {}
    for pid, pr in by_id.items():
        nm = _norm(pr.player)
        if nm:
            name_idx[nm] = pid
            last_idx.setdefault(nm.split()[-1], []).append(pid)
    model = {"base": base, "by_id": by_id, "elo": elo, "name": name_idx, "last": last_idx,
             "live_matches": live_n}
    _MODEL_CACHE[tour] = (time.time(), model)
    return model


def _baseline_player(base, tour: str) -> R.PlayerRates:
    # No real match history at all -- ALL of spw/rpw's weight is the tour prior, i.e.
    # maximum parameter uncertainty (eff_*_pts = just the shrinkage pseudo-count).
    return R.PlayerRates(player_id="", player="(unknown)", tour=tour, spw=base.spw_avg,
                         rpw=base.rpw_avg, ace_rate=base.ace_rate_avg, df_rate=base.df_rate_avg,
                         pts_per_svgame=base.pts_per_svgame_avg, n_matches=0,
                         eff_serve_pts=cfg("model", "shrink_serve_pts"),
                         eff_return_pts=cfg("model", "shrink_return_pts"))


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


def _model_agreement(win_sim: float, win_elo: float, win_analytic: float) -> float:
    """0..1 — how closely the three INDEPENDENT win-probability estimates (Monte-Carlo
    simulation, Elo, closed-form point-model) agree. Low agreement flags a genuinely
    uncertain/conflicting matchup even when the sample size looks fine — a real,
    complementary confidence signal, not a duplicate of eff_matches."""
    vals = [win_sim, win_elo, win_analytic]
    spread = max(vals) - min(vals)
    return round(max(0.0, 1.0 - spread / 0.35), 3)   # a 35pp spread -> 0 agreement


def _apply_fatigue(rates: R.PlayerRates, penalty: float) -> R.PlayerRates:
    """A small serve-rate shift for a tired player (spec's Fatigue Model) — applied to a
    COPY, never the shared cached PlayerRates (which every match for this player reuses)."""
    if not penalty:
        return rates
    return replace(rates, spw=rates.spw + penalty,
                   surface_spw={s: v + penalty for s, v in rates.surface_spw.items()})


def project_match(tour: str, player_a: str, player_b: str, surface: str,
                  best_of: int = 3, final_set_advantage: bool = False, n=None,
                  fatigue_a: float = 0.0, fatigue_b: float = 0.0) -> dict:
    m = _model(tour)
    base, elo = m["base"], m["elo"]
    ra, pa_id = resolve(tour, player_a)
    rb, pb_id = resolve(tour, player_b)
    ra, rb = _apply_fatigue(ra, fatigue_a), _apply_fatigue(rb, fatigue_b)
    surface = surface if surface in ra.surface_spw or True else surface

    sim = simulate(ra, rb, surface, base, best_of=best_of,
                   final_set_advantage=final_set_advantage, n=n)
    win_sim = float((sim["winner"] == 0).mean())
    # Elo is keyed by normalized name now (bridges the historical mirror + live ESPN
    # results — see model/elo.py), so this always has SOMETHING sensible: a real fitted
    # rating, a rank-based prior for a mirror-absent player, or the neutral tour-average
    # start for a total unknown — no need to special-case "found vs not found" here.
    win_elo = elo.win_prob(player_a, player_b, surface)
    blend = cfg("model", "match_prob_blend")
    win_a = blend * win_sim + (1 - blend) * win_elo

    # Confidence takes the BETTER of two independent sample-size signals: the point
    # model's own serve/return data (`eff_n`, Sackmann-only) and Elo's live-match count
    # (`elo.eff_matches`, which includes recent ESPN results). This is what actually fixes
    # "the model always defers to market" for a player like Sinner/Alcaraz — the Sackmann
    # mirror predates their careers (eff_n=0 forever), but they play real matches every
    # week that now feed a live, current Elo rating, so their WIN-PROBABILITY confidence
    # can be real even while their per-shot serve/return rates still lean on tour baseline.
    eff_n = min(ra.eff_matches(surface), rb.eff_matches(surface))
    elo_eff_n = min(elo.eff_matches(player_a), elo.eff_matches(player_b))
    conf_eff_n = max(eff_n, elo_eff_n)
    # Key the per-player markets by the REQUESTED names, not the resolved ones. `resolve` falls
    # back to a baseline literally named "(unknown)" for anyone absent from the data, so keying
    # by ra.player/rb.player meant a caller asking for the name it passed in got None and dropped
    # the line — and when BOTH players were unknown the two dict keys COLLIDED and player A's
    # distributions were silently overwritten by B's. Callers only ever know the name they asked
    # for, so that's the contract. (An unresolved player still projects: baseline rates, 0
    # effective matches → low confidence → the board anchors it to the market line.)
    key_a, key_b = player_a, player_b
    dist = {"total_games": sim["total_games"], "total_sets": sim["total_sets"],
            key_a: {"aces": sim["aces_a"], "dfs": sim["dfs_a"], "games": sim["games_a"],
                    "sets": sim["sets_a"], "breaks": sim["breaks_a"]},
            key_b: {"aces": sim["aces_b"], "dfs": sim["dfs_b"], "games": sim["games_b"],
                    "sets": sim["sets_b"], "breaks": sim["breaks_b"]}}
    return {
        "tour": tour, "surface": surface, "best_of": best_of,
        "player_a": player_a, "player_b": player_b,
        "resolved_a": ra.player, "resolved_b": rb.player,   # "(unknown)" = not in the data
        "win_prob_a": round(win_a, 4), "win_prob_b": round(1 - win_a, 4),
        "win_prob_a_sim": round(win_sim, 4), "win_prob_a_elo": round(win_elo, 4),
        "win_prob_a_analytic": round(match_win_analytic(ra, rb, surface, base, best_of), 4),
        "eff_matches": eff_n, "elo_eff_matches": elo_eff_n,
        "confidence": _confidence(conf_eff_n),
        "model_agreement": _model_agreement(win_sim, win_elo,
                                            match_win_analytic(ra, rb, surface, base, best_of)),
        "markets": {
            "total_games": summary(sim["total_games"]),
            "total_sets": summary(sim["total_sets"]),
            key_a: {"aces": summary(sim["aces_a"]), "dfs": summary(sim["dfs_a"]),
                    "games": summary(sim["games_a"])},
            key_b: {"aces": summary(sim["aces_b"]), "dfs": summary(sim["dfs_b"]),
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
    # "winner" catches per-set/match winner side-bets ("1st Set Winner", "Match Winner") —
    # binary who-wins props, not a count market this resolves lines/probabilities for.
    # (_norm strips digits, so "1st Set Winner" normalizes to "st set winner" — check the
    # substring, not the literal "1st".) period/tiebreak/half markets are the other
    # documented not-modelled-yet exclusions.
    if ("winner" in s or "period" in s or "tie breaker" in s or "tiebreaker" in s
            or "1h" in s or "2h" in s):
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
