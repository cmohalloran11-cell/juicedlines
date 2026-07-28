"""
valuation.py — SimulationOS valuation engine (spec Vol V Part 3).

Turns a projection + its market line into the decision-support numbers the spec calls for:
expected value (Ch 148), Juice Score (Ch 149), Kelly fraction (Ch 150), and a confidence
score (Ch 103). Everything here is a PURE function of numbers already on the line object —
no I/O, no model calls — so it is fully deterministic and unit-testable.

Design principle (matches the product's "transparency over black box"): these are explicit,
documented heuristics, not a hidden score. Every output can be traced to its inputs.

Odds convention: the stored `over_implied` / `under_implied` are vig-included implied
probabilities from the book's price. For a side with implied prob q, the offered decimal
odds are D = 1/q, so:
    EV per $1 stake = model_p * D - 1 = model_p / q - 1
    Kelly fraction  = (D*model_p - 1) / (D - 1) = (model_p - q) / (1 - q)
We evaluate the side the model favors (over if model_prob >= 0.5, else under). Pick'em books
without a per-side price fall back to an even-money payout (D = 2.0).
"""
from __future__ import annotations

from typing import Any, Optional


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _favored_side(model_prob: float, line: dict[str, Any]) -> tuple[str, float, Optional[float]]:
    """Return (side, model_p_for_side, implied_p_for_side). implied may be None."""
    if model_prob >= 0.5:
        return "over", model_prob, line.get("over_implied")
    return "under", 1.0 - model_prob, line.get("under_implied")


def _decimal_odds(implied: Optional[float]) -> float:
    """Offered decimal odds from a vig-included implied prob; even-money fallback."""
    if implied and 0.0 < implied < 1.0:
        return 1.0 / implied
    return 2.0  # pick'em / unknown price → even money


def expected_value(model_prob: float, line: dict[str, Any]) -> Optional[float]:
    """EV per $1 staked on the model's favored side. Positive = model sees value. None if
    there is no probability to work with."""
    if model_prob is None:
        return None
    _side, p, implied = _favored_side(float(model_prob), line)
    d = _decimal_odds(implied)
    return round(p * d - 1.0, 4)


def kelly_fraction(model_prob: float, line: dict[str, Any], cap: float = 0.25) -> Optional[float]:
    """Fraction of bankroll to stake by the Kelly criterion on the favored side, capped
    (quarter-Kelly by default — full Kelly is too aggressive for noisy prop models)."""
    if model_prob is None:
        return None
    _side, p, implied = _favored_side(float(model_prob), line)
    d = _decimal_odds(implied)
    if d <= 1.0:
        return 0.0
    f = (d * p - 1.0) / (d - 1.0)
    return round(_clamp(f, 0.0, cap), 4)


def confidence_score(line: dict[str, Any]) -> int:
    """
    0–100 confidence in the projection itself (NOT how good the bet is). Blends:
      • sample size (more games → steadier estimate), saturating at ~30 games,
      • decisiveness of P(over) (how far the model is from a coin flip),
      • method (a full engine run with a distribution beats a bare empirical average).
    Heuristic and deliberately transparent — see the weights below.
    """
    n = line.get("model_n") or 0
    prob = line.get("model_prob")
    n_factor = _clamp(n / 30.0)
    prob_factor = _clamp(2.0 * abs(float(prob) - 0.5)) if prob is not None else 0.0
    method_factor = 1.0 if line.get("proj_kind") == "engine" else 0.5
    score = 100.0 * (0.5 * n_factor + 0.3 * prob_factor + 0.2 * method_factor)
    return int(round(_clamp(score / 100.0) * 100))


def confidence_factors(line: dict[str, Any]) -> list[dict[str, Any]]:
    """The REAL decomposition of `confidence_score` — the actual weighted contributions
    (each 0..max), so the UI can show what drives a prop's confidence instead of a constant.
    The three values sum to the confidence score (max 50 + 30 + 20 = 100)."""
    n = int(line.get("model_n") or 0)
    prob = line.get("model_prob")
    n_factor = _clamp(n / 30.0)
    prob_factor = _clamp(2.0 * abs(float(prob) - 0.5)) if prob is not None else 0.0
    method_factor = 1.0 if line.get("proj_kind") == "engine" else 0.5
    return [
        {"factor": "Sample Size", "value": round(50.0 * n_factor, 1), "max": 50,
         "detail": f"{n} game{'' if n == 1 else 's'} of history"},
        {"factor": "Decisiveness", "value": round(30.0 * prob_factor, 1), "max": 30,
         "detail": "how far P(over) is from a coin-flip"},
        {"factor": "Method", "value": round(20.0 * method_factor, 1), "max": 20,
         "detail": "full engine run" if method_factor == 1.0 else "empirical average"},
    ]


def juice_score(line: dict[str, Any], model_prob: Optional[float] = None) -> int:
    """
    0–100 composite ranking how JUICY a play is = decisiveness × confidence, the number the
    slate leaderboard sorts on. A high Juice Score means the model is both confident in the
    projection AND far from the line. This is a shortlisting signal, not a guarantee.
    """
    prob = model_prob if model_prob is not None else line.get("model_prob")
    if prob is None:
        return 0
    decisiveness = _clamp(2.0 * abs(float(prob) - 0.5))
    conf = confidence_score(line) / 100.0
    # Confidence GATES the score: a decisive-but-low-confidence prop (e.g. a market-derived
    # tennis line with an extreme probability) can't top the board over a well-sampled,
    # engine-projected MLB prop. The trailing (0.4 + 0.6·conf) factor is the gate.
    return int(round(100.0 * (0.45 * decisiveness + 0.55 * conf) * (0.4 + 0.6 * conf)))


def _std_from_band(line: dict[str, Any]) -> Optional[float]:
    """Approximate SD from the shipped p10–p90 band (≈ 2.5631 sigma wide). Labeled approximate
    because the underlying distribution is discrete/skewed — this is for display, not sampling."""
    lo, hi = line.get("model_floor"), line.get("model_ceiling")
    if lo is None or hi is None or hi <= lo:
        return None
    return round((float(hi) - float(lo)) / 2.5631, 3)


def simulation_object(line: dict[str, Any]) -> Optional[dict]:
    """
    The spec's Simulation object (Ch 28 / 140): the projection's distribution summary +
    over/under probabilities + sample size. Built from fields the engine already produced.
    None when the line has no projection.
    """
    proj = line.get("model_proj")
    if proj is None:
        return None
    prob_over = line.get("model_prob")
    return {
        "projectionId": line.get("id"),
        "mean": proj,
        "median": line.get("model_proj"),           # engine reports mean as the point estimate
        "standardDeviation": _std_from_band(line),  # approximate — see _std_from_band
        "floor": line.get("model_floor"),
        "ceiling": line.get("model_ceiling"),
        "p25": line.get("p25"),
        "p75": line.get("p75"),
        "overProbability": prob_over,
        "underProbability": (round(1.0 - float(prob_over), 3) if prob_over is not None else None),
        "sampleSize": line.get("model_n"),
        "sd_is_approximate": True,
    }


def valuation(line: dict[str, Any]) -> dict:
    """Full valuation bundle for one projection: the favored side + EV, Kelly, confidence,
    and Juice Score. Returns {available: False} when there's nothing to value."""
    prob = line.get("model_prob")
    proj = line.get("model_proj")
    if prob is None or proj is None or line.get("line") is None:
        return {"available": False, "reason": "No model projection/probability for this line."}
    side, _p, implied = _favored_side(float(prob), line)
    return {
        "available": True,
        "side": side,
        "line": line.get("line"),
        "projection": proj,
        "edge": line.get("model_edge"),
        "probability": prob,
        "impliedProbability": implied,
        "expectedValue": expected_value(prob, line),
        "kellyFraction": kelly_fraction(prob, line),
        "confidence": confidence_score(line),
        "juiceScore": juice_score(line),
    }
