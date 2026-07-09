"""
Monte-Carlo fixture simulation.

1. Sample N exact scorelines from the Dixon-Coles matrix → per-sim (home goals,
   away goals) and a match tempo (total goals vs reference) for volume scaling.
2. Goals: allocate each team's simulated goals to players via Binomial(team_goals,
   xg_share × minutes-fraction) — share of xG, not raw minutes.
3. Shots on target: Poisson(sot90 × minutes × tempo).
4. Cards: Poisson((yellow+red)/90 × minutes × knockout/rivalry intensity).
5. GK saves: shots-faced ~ Poisson(opponent shots × on-target rate × tempo), then
   Binomial(shots_faced, save%).
Minutes-fraction and a confidence flag come from each player's start probability.
"""

from __future__ import annotations

import numpy as np

from ..config import load
from ..model.goal_expectancy import expectancy, score_matrix

SOT_LINES = [0.5, 1.5, 2.5]
SAVE_LINES = [1.5, 2.5, 3.5, 4.5]
_SOT_OF_SHOTS = 0.34          # share of a team's shots that are on target (GK shots-faced)


def _minutes_frac(start_prob: float) -> float:
    return float(min(1.0, max(0.30, start_prob)))


def _confidence(start_prob: float) -> str:
    c = load()["confidence"]
    if start_prob >= c["confirmed"]:
        return "confirmed"
    if start_prob >= c["probable"]:
        return "probable"
    return "rotation risk"


def simulate(fx, ratings, players_home, players_away,
             strength_home, strength_away, rng=None) -> dict:
    cfg = load()["model"]
    n, mg = cfg["n_sims"], cfg["max_goals"]
    rng = rng or np.random.default_rng()

    lam, mu = expectancy(fx, ratings)
    flat = score_matrix(lam, mu).ravel()
    idx = rng.choice(flat.size, size=n, p=flat)
    hg, ag = idx // (mg + 1), idx % (mg + 1)             # home/away goals per sim
    total = hg + ag
    tempo = np.clip(total / max(0.5, cfg["tempo_ref_goals"]), 0.6, 1.6)

    intensity = 1.0
    ic = load()["intensity"]
    if fx.knockout:
        intensity *= ic["knockout"]
    if fx.rivalry:
        intensity *= ic["rivalry"]

    players: dict[str, dict] = {}
    for roster, team_goals, opp_shots in (
        (players_home, hg, (strength_away.shots_pg if strength_away else 12.0)),
        (players_away, ag, (strength_home.shots_pg if strength_home else 12.0)),
    ):
        for p in roster:
            mf = _minutes_frac(p.start_prob)
            e = {"team": p.team, "pos": p.position, "confidence": _confidence(p.start_prob),
                 "start_prob": round(p.start_prob, 2)}

            if p.position == "GK":
                faced = rng.poisson(np.maximum(0.1, opp_shots * _SOT_OF_SHOTS * tempo))
                saves = rng.binomial(faced, min(0.95, max(0.40, p.save_pct or 0.70)))
                e["saves"] = {"exp": round(float(saves.mean()), 2),
                              "over": {l: round(float((saves > l).mean()), 4) for l in SAVE_LINES}}
            else:
                share = float(min(0.60, p.xg_share * mf))
                pg = rng.binomial(team_goals, share) if share > 0 else np.zeros(n, int)
                e["goal"] = {"anytime": round(float((pg >= 1).mean()), 4),
                             "two_plus": round(float((pg >= 2).mean()), 4),
                             "exp": round(float(pg.mean()), 3)}
                sot = rng.poisson(np.maximum(0.0, p.sot90 * mf * tempo))
                e["sot"] = {"exp": round(float(sot.mean()), 2),
                            "over": {l: round(float((sot > l).mean()), 4) for l in SOT_LINES}}
                cards = rng.poisson(max(0.0, (p.yellow90 + p.red90) * mf * intensity), size=n)
                e["card"] = {"yes": round(float((cards >= 1).mean()), 4)}

            players[p.name] = e

    return {"fixture": fx.id, "home": fx.home, "away": fx.away, "stage": fx.stage,
            "lambda_home": round(lam, 3), "mu_away": round(mu, 3),
            "exp_total_goals": round(float(total.mean()), 2),
            "card_intensity": round(intensity, 3), "players": players}
