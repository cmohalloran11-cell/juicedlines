"""
mlb_features.py — build a model `form` for a batter or pitcher.

Implements the Bayesian blend (recent 60% / season 30% / career 10%) and then
shrinks the blended *empirical* rate toward a *predictive-metric* prior (xBA,
xwOBA, xFIP, SwStr%, …) using regress_to_prior. Predictive metrics are the
inputs, NOT the raw stats — exactly per spec.

`game_logs`: list of per-game dicts (oldest→newest). Batter keys: PA, AB, H,
'2B','3B', HR, BB, SO, RBI, R, SB. Pitcher keys: BF, IP/outs, SO, BB, H, ER.
`predictive`: dict of Statcast/FanGraphs metrics (xBA, xwOBA, barrel_pct, …).
`splits`: optional {'vs_L': {...rates}, 'vs_R': {...rates}}.
"""

from __future__ import annotations

from typing import Any

from ..config import load
from ..models.base import blend_rate, regress_to_prior, _f


# rate key → projected stat (for per-stat recency weighting)
_KEY2STAT = {"p_hit": "hits", "p_hr": "home_runs", "p_bb": "walks",
             "p_k": "strikeouts", "p_1b": "total_bases", "p_2b": "total_bases",
             "p_3b": "total_bases", "exp_rbi": "rbis", "exp_runs": "runs",
             "exp_sb": "stolen_bases"}


def _agg(logs: list[dict], keys: list[str]) -> dict[str, float]:
    return {k: sum(_f(g.get(k)) for g in logs) for k in keys}


def _rate(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den > 0 else default


# ── batter ───────────────────────────────────────────────────────────────────

def build_batter_form(game_logs: list[dict], predictive: dict | None = None,
                      splits: dict | None = None, career: dict | None = None) -> dict:
    cfg = load()["blend"]
    predictive = predictive or {}
    logs = sorted(game_logs, key=lambda g: g.get("date", ""))
    recent = logs[-cfg["recent_games"]:]

    def comp(window: list[dict]) -> dict[str, float]:
        a = _agg(window, ["PA", "AB", "H", "2B", "3B", "HR", "BB", "SO", "RBI", "R", "SB"])
        pa = max(1.0, a["PA"])
        singles = a["H"] - a["2B"] - a["3B"] - a["HR"]
        return {
            "p_hit": _rate(a["H"], pa), "p_hr": _rate(a["HR"], pa),
            "p_bb": _rate(a["BB"], pa), "p_k": _rate(a["SO"], pa),
            "p_1b": _rate(singles, pa), "p_2b": _rate(a["2B"], pa),
            "p_3b": _rate(a["3B"], pa),
            "exp_rbi": _rate(a["RBI"], max(1, len(window))),
            "exp_runs": _rate(a["R"], max(1, len(window))),
            "exp_sb": _rate(a["SB"], max(1, len(window))),
            "pa": pa,
        }

    r, s = comp(recent), comp(logs)
    c = career or s                                  # career baseline (fallback: season)
    season_pa = s["pa"]

    by_stat = cfg.get("recency_by_stat", {})
    form: dict[str, Any] = {"role": "batter"}
    for key in ["p_hit", "p_hr", "p_bb", "p_k", "p_1b", "p_2b", "p_3b",
                "exp_rbi", "exp_runs", "exp_sb"]:
        # per-stat recency weight (fitted from backtest), else the blend default
        rw = by_stat.get(_KEY2STAT.get(key, ""), cfg["recent_weight"])
        w = {"recent_weight": rw, "season_weight": (1 - rw) * 0.85,
             "career_weight": (1 - rw) * 0.15, "recent_games": cfg["recent_games"]}
        blended = blend_rate(r[key], s[key], c.get(key, s[key]), w)
        prior = _batter_prior(key, predictive, blended)
        form[key] = regress_to_prior(blended, prior, season_pa, cfg["shrinkage_pa"])

    form["exp_pa"] = _f(predictive.get("lineup_pa"), s["pa"] / max(1, len(logs)) or 4.2)
    if splits:
        form["platoon"] = splits
    form["_predictive"] = {k: predictive.get(k) for k in
                           ("xBA", "xwOBA", "xSLG", "barrel_pct", "hard_hit_pct",
                            "sprint_speed") if k in predictive}
    return form


def _batter_prior(key: str, pred: dict, fallback: float) -> float:
    """Map a predictive metric to a per-PA rate prior (else fall back to empirical)."""
    if key == "p_hit" and pred.get("xBA") is not None:
        return _f(pred["xBA"]) * 0.91            # xBA is per-AB; ~0.91 AB/PA
    if key == "p_hr":
        if pred.get("barrel_pct") is not None:
            return min(0.12, 0.55 * _f(pred["barrel_pct"]))   # barrels ≈ HR engine
        if pred.get("xISO") is not None:
            return min(0.12, 0.18 * _f(pred["xISO"]))
    if key == "p_k" and pred.get("k_pct") is not None:
        return _f(pred["k_pct"])
    if key == "p_bb" and pred.get("bb_pct") is not None:
        return _f(pred["bb_pct"])
    return fallback


# ── pitcher ──────────────────────────────────────────────────────────────────

def build_pitcher_form(game_logs: list[dict], predictive: dict | None = None,
                       career: dict | None = None) -> dict:
    cfg = load()["blend"]
    predictive = predictive or {}
    logs = sorted(game_logs, key=lambda g: g.get("date", ""))
    recent = logs[-cfg["recent_games"]:]

    def comp(window: list[dict]) -> dict[str, float]:
        a = _agg(window, ["BF", "SO", "BB", "H", "ER", "outs"])
        bf = max(1.0, a["BF"])
        games = max(1, len(window))
        return {"p_k": _rate(a["SO"], bf), "p_bb": _rate(a["BB"], bf),
                "p_h": _rate(a["H"], bf), "exp_bf": bf / games,
                "exp_outs": _rate(a["outs"], games), "bf": bf}

    r, s = comp(recent), comp(logs)
    c = career or s
    form: dict[str, Any] = {"role": "pitcher"}
    for key in ["p_k", "p_bb", "p_h", "exp_bf", "exp_outs"]:
        blended = blend_rate(r[key], s[key], c.get(key, s[key]), cfg)
        prior = _pitcher_prior(key, predictive, blended)
        form[key] = regress_to_prior(blended, prior, s["bf"], cfg["shrinkage_pa"])

    # xERA prior drives earned runs
    form["xera"] = _f(predictive.get("xERA"), _f(predictive.get("xFIP"), 4.0))
    form["_predictive"] = {k: predictive.get(k) for k in
                           ("xFIP", "xERA", "swstr_pct", "csw_pct", "gb_pct",
                            "k_bb_pct") if k in predictive}
    return form


def _pitcher_prior(key: str, pred: dict, fallback: float) -> float:
    if key == "p_k" and pred.get("csw_pct") is not None:
        return min(0.45, 1.15 * _f(pred["csw_pct"]) - 0.07)    # CSW% → K/BF (rough)
    if key == "p_k" and pred.get("swstr_pct") is not None:
        return min(0.45, 0.018 + 1.6 * _f(pred["swstr_pct"]))
    return fallback
