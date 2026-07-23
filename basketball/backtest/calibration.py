"""
Date-strict backtest — walk forward through each player's games, predicting game i
using ONLY the games before it (no leakage), and score bias / MAE / calibration.

Reported per league. WNBA is the core validator (healthy samples). For Summer
League the honest check is whether low-confidence projections actually miss more
and whether the translation priors are biased by source league — both gated on the

    from basketball.backtest import calibration as c
    c.run("WNBA", stat="pts")
"""

from __future__ import annotations

import numpy as np

from ..config import league_cfg, cfg
from ..data import gamelog_source
from ..model import rates as R, priors as PR, minutes as MIN
from ..model.pace import matchup_pace
from ..sim import engine as E


def run(league: str = "WNBA", stat: str = "pts", min_prior: int = 6,
        max_players: int = 40, sims: int = 1500, seed: int = 0) -> dict:
    src = gamelog_source()
    players = src.players(league)
    lc = league_cfg(league)
    game_len, lg_pace = lc.get("game_minutes", 40), src.league_pace(league)
    halflife = cfg("model", "recency_halflife")
    rng = np.random.default_rng(seed)

    pairs = []            # (proj_mean, actual, p_over, line, conf)
    used = 0
    for nm, ref in players.items():
        if used >= max_players:
            break
        gl = src.gamelog(league, ref.id)
        for g in gl:
            g.player, g.team, g.team_id = ref.name, ref.team, ref.team_id
        if len(gl) < min_prior + 3:
            continue
        used += 1
        for i in range(0, len(gl) - min_prior):
            hist = gl[i + 1:]                      # strictly older (recent-first list)
            if len(hist) < min_prior:
                break
            prior = PR.positional_prior_poss(ref.position, lg_pace, league)
            rates = R.fit_rates(hist, league, prior, game_len, lg_pace,
                                lc.get("shrink_poss", 200), halflife)
            pmin, psd = MIN.project_minutes(rates.minutes_sample, league,
                                            lc.get("minutes_shrink_games", 3),
                                            lc.get("min_sd_frac", 0.15))
            sim = E.simulate(rates, pmin, psd, matchup_pace(lg_pace),
                             lc.get("pace_sd_frac", 0.06), game_len,
                             lc.get("disp", 0.12), n=sims, rng=rng)
            arr = sim[stat]
            proj = float(arr.mean())
            actual = gl[i].stat(stat)
            line = round(proj * 2) / 2 - 0.5      # a plausible .5 book line near proj
            pairs.append((proj, actual, float((arr > line).mean()), line))

    n = len(pairs)
    if not n:
        return {"league": league, "stat": stat, "n": 0, "note": "not enough games"}
    bias = sum(p - a for p, a, _, _ in pairs) / n
    mae = sum(abs(p - a) for p, a, _, _ in pairs) / n

    # calibration: bucket predicted P(over) into deciles vs realized over-rate
    buckets: dict = {}
    for p, a, po, line in pairs:
        buckets.setdefault(min(9, int(po * 10)), []).append((po, 1.0 if a > line else 0.0))
    ece, rows = 0.0, []
    for b in sorted(buckets):
        v = buckets[b]
        mp = sum(x[0] for x in v) / len(v)
        mr = sum(x[1] for x in v) / len(v)
        ece += abs(mp - mr) * len(v) / n
        rows.append({"bucket": b, "n": len(v), "pred": round(mp, 3), "real": round(mr, 3)})
    return {"league": league, "stat": stat, "players": used, "n": n,
            "bias": round(bias, 3), "mae": round(mae, 3), "ece": round(ece, 3),
            "calibration": rows}
