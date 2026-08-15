"""
NFL confidence — a SUPPLEMENTARY display score, not a replacement.

valuation.confidence_score keeps working for NFL exactly as it does for every other sport
(off model_n / model_prob / proj_kind); this adds the sport-specific decomposition a football
prop actually needs, in the same shape WNBA's `bball_confidence` ships in. It is attached to a
line as `nfl_confidence` (0-100 plus a high/medium/low flag) and `nfl_confidence_factors` (the
real weighted contributions, so the UI can show WHY rather than an opaque number).

Regular season and preseason use DIFFERENT factor sets, because the questions differ: in
September playing time is close to known and the matchup/environment carry real information;
in August playing time is the whole ballgame and the matchup is meaningless. Both weight sets
live in config.nfl_confidence.

Every factor is computed from a field the model actually produced. A factor with no real
input returns None and is DROPPED from the blend, with the remaining weights renormalised —
not scored at a neutral 0.5, which would quietly manufacture confidence out of a missing
signal.
"""

from __future__ import annotations

from typing import Optional

from .config import cfg


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _scale(x: Optional[float], lo: float, hi: float) -> Optional[float]:
    if x is None or hi == lo:
        return None
    return _clamp((float(x) - lo) / (hi - lo))


_CONF_LEVEL = {"high": 1.0, "medium": 0.6, "low": 0.25}


def _factors(proj: dict, line: dict) -> dict[str, tuple[Optional[float], str]]:
    """{factor: (0..1 score or None, human detail)} — every real signal, for both weightings."""
    pt = proj.get("playing_time")
    usage = proj.get("usage")
    env = proj.get("environment")
    out: dict[str, tuple[Optional[float], str]] = {}

    if pt is not None:
        # A narrow snap-share band IS playing-time certainty. Widths come straight off the
        # Beta the simulator draws from, so this can't disagree with the simulation.
        lo, hi = pt.snap_share_range()
        width = hi - lo
        out["playing_time"] = (_scale(width, 0.55, 0.05),
                               f"{pt.role}, snap share {lo*100:.0f}-{hi*100:.0f}% "
                               f"({pt.basis.replace('_', ' ')})")
    else:
        out["playing_time"] = (None, "no playing-time estimate")

    if usage is not None:
        out["usage_stability"] = (_scale(usage.sample_weight, 0.0, 0.7),
                                  f"{usage.sample_weight*100:.0f}% of the usage estimate is "
                                  f"his own sample ({usage.n_games} games)")
        out["role_stability"] = (_scale(usage.eff_games, 0.0, 8.0),
                                 f"{usage.eff_games:.1f} recency-weighted games of history")
    else:
        out["usage_stability"] = (None, "no usage sample")
        out["role_stability"] = (None, "no usage sample")

    tier = proj.get("rotation_tier")
    if tier:
        risk = proj.get("preseason_risk")
        out["rotation_certainty"] = (None if risk is None else _clamp(1.0 - float(risk)),
                                     f"{tier.replace('_', ' ')}")
    else:
        out["rotation_certainty"] = (None, "not a preseason projection")

    status = line.get("lineup_status")
    if status == "out":
        out["injury_certainty"] = (0.0, "ruled out")
    elif status == "questionable":
        out["injury_certainty"] = (0.35, "questionable")
    elif proj.get("playing_time") is not None and pt.playing_time_probability is not None:
        out["injury_certainty"] = (_clamp(pt.playing_time_probability),
                                   f"played in {pt.playing_time_probability*100:.0f}% of recent games")
    else:
        # nflverse publishes no injury report; weekly roster status is the only availability
        # signal and it doesn't exist for a preseason week.
        out["injury_certainty"] = (None, "no availability signal in this feed")

    agreement = line.get("model_agreement")
    if agreement is None and line.get("model_raw") is not None and line.get("model_proj") is not None:
        gap = abs(float(line["model_raw"]) - float(line["model_proj"])) / max(abs(float(line["model_proj"])), 0.5)
        agreement = _scale(gap, 0.6, 0.0)
    out["model_agreement"] = (agreement if agreement is None else _clamp(float(agreement)),
                              "model vs market-anchored projection"
                              if agreement is not None else "no raw-vs-blended comparison")

    defense = proj.get("opponent_defense")
    if defense is not None and proj.get("matchup"):
        out["matchup"] = (_scale(min(defense.games, 17), 0, 12),
                          f"opponent defense fit off {defense.games} games")
    else:
        out["matchup"] = (None, "opponent unknown or too little defensive history")

    if env is not None and env.basis == "market_priced":
        out["game_environment"] = (1.0, f"total {env.game_total}, team total {env.team_total}")
    else:
        out["game_environment"] = (None, "no market spread/total for this game")

    hist = line.get("historical_accuracy")
    out["historical_model_accuracy"] = (
        None if hist is None else _scale(hist, 0.45, 0.62),
        f"{float(hist)*100:.0f}% graded hit rate" if hist is not None
        else "no graded NFL history yet")

    trust = line.get("trust_weight")
    out["market_agreement"] = (
        None if trust is None else _clamp(float(trust)),
        f"{float(trust)*100:.0f}% model weight vs the market line" if trust is not None
        else "not market-anchored")

    return out


def nfl_confidence(proj: dict, line: dict) -> dict:
    """{"score": 0-100, "level": high|medium|low, "factors": [...], "season_type": ...}.

    `score` is None when NOT ONE factor could be computed — the honest answer for a projection
    with nothing behind it, rather than a number built from defaults.
    """
    season_type = proj.get("season_type", "regular")
    weights = cfg("nfl_confidence", season_type, default={}) or cfg("nfl_confidence", "regular")
    computed = _factors(proj, line)

    available = {k: w for k, w in weights.items()
                 if computed.get(k, (None, ""))[0] is not None}
    total_w = sum(available.values())
    factors = []
    for k, w in weights.items():
        score, detail = computed.get(k, (None, "not computed"))
        factors.append({
            "factor": k.replace("_", " ").title(),
            "weight": round(w * 100, 1),
            "value": None if score is None else round(100.0 * (w / total_w) * score, 1) if total_w else None,
            "max": round(100.0 * (w / total_w), 1) if total_w else None,
            "detail": detail,
        })
    if not total_w:
        return {"score": None, "level": None, "factors": factors, "season_type": season_type,
                "note": "not enough real signal to score confidence for this projection yet"}
    score = int(round(100.0 * sum(available[k] * computed[k][0] for k in available) / total_w))
    hi, md = cfg("nfl_confidence", "high", default=70), cfg("nfl_confidence", "medium", default=45)
    level = "high" if score >= hi else "medium" if score >= md else "low"
    return {"score": score, "level": level, "factors": factors, "season_type": season_type,
            "note": None}
