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

Side selection (`recommend_side`) picks whichever AVAILABLE side has the higher EV — NOT
whichever the model gives >50% probability. This matters for multiplier books (Sleeper,
Underdog): a 30%-likely Over at a 4.0x payout can beat a 70%-likely Under at a thin 1.1x
payout, and a side with no price at all (the book doesn't offer it — e.g. some Home Run
props are Over-only) is never recommended even if the model likes it. Pick'em books without
a per-side price fall back to an even-money payout (D = 2.0) on whichever side(s) they do
offer (PrizePicks standard legs: same flat payout either way, both sides always offered).
"""
from __future__ import annotations

import os
from typing import Any, Optional


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _american_to_decimal(american: Any) -> Optional[float]:
    try:
        a = float(american)
    except (TypeError, ValueError):
        return None
    if a > 0:
        return 1.0 + a / 100.0
    if a < 0:
        return 1.0 + 100.0 / abs(a)
    return None


def _side_decimal_odds(line: dict[str, Any], side: str) -> Optional[float]:
    """Decimal odds actually offered for ONE side, or None if that side isn't offered at all.
    Real per-side implied prob (Underdog/Sleeper, from that side's own multiplier/price) wins;
    PrizePicks' flat pick'em price applies to either side (real PrizePicks: same payout no
    matter which way you pick on a standard leg). A side with neither — a genuine per-side
    price missing on a source that DOES report per-side prices (Underdog/Sleeper) — means the
    book doesn't actually offer that side for this leg (e.g. an Over-only Home Run prop); we
    must not invent a price for it."""
    implied = line.get(f"{side}_implied")
    if implied and 0.0 < implied < 1.0:
        return 1.0 / implied
    d = _american_to_decimal(line.get("pickem_price"))
    return d if d and d > 1.0 else None


# PrizePicks demon/goblin legs carry their own boosted-payout multiplier that this feed does
# NOT expose (pullers.py leaves pickem_price None for them on purpose). There is no real price
# to compute EV/Kelly against for these — returning a number anyway (even the even-money
# fallback) would be exactly the fabricated-number problem the Edge/EV audit removed. They are
# also structurally single-sided on PrizePicks: you can only take the boosted direction, never
# the Under — see recommend_side().
_UNPRICED_ODDS_TYPES = {"demon", "goblin"}


def is_unpriced(line: dict[str, Any]) -> bool:
    return (line.get("odds_type") or "standard").lower() in _UNPRICED_ODDS_TYPES


def recommend_side(model_prob: Optional[float], line: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The side to actually recommend: whichever AVAILABLE side has the higher EV, not
    whichever the model gives >50% to. Returns None when there's no probability, or the book
    offers neither side for this leg (don't recommend a bet that can't be placed).

    Demon/goblin are always "over" (PrizePicks doesn't let you take Under on a boosted leg)
    and carry no EV/price — the boosted payout isn't exposed by the feed, so we don't guess.
    """
    if model_prob is None:
        return None
    if is_unpriced(line):
        return {"side": "over", "p": float(model_prob), "decimal_odds": None, "ev": None}
    p = float(model_prob)
    candidates = []
    over_d = _side_decimal_odds(line, "over")
    if over_d is not None:
        candidates.append({"side": "over", "p": p, "decimal_odds": over_d,
                           "ev": round(p * over_d - 1.0, 4)})
    under_d = _side_decimal_odds(line, "under")
    if under_d is not None:
        q = 1.0 - p
        candidates.append({"side": "under", "p": q, "decimal_odds": under_d,
                           "ev": round(q * under_d - 1.0, 4)})
    if not candidates:
        return None
    return max(candidates, key=lambda c: c["ev"])


def expected_value(model_prob: float, line: dict[str, Any]) -> Optional[float]:
    """EV per $1 staked on the recommended side. Positive = model sees value. None if there's
    no probability, no priceable/offered side, or the line is a demon/goblin (unpriced)."""
    if model_prob is None or is_unpriced(line):
        return None
    rec = recommend_side(model_prob, line)
    return rec["ev"] if rec else None


def kelly_fraction(model_prob: float, line: dict[str, Any], cap: float = 0.25) -> Optional[float]:
    """Fraction of bankroll to stake by the Kelly criterion on the recommended side, capped
    (quarter-Kelly by default — full Kelly is too aggressive for noisy prop models)."""
    if model_prob is None or is_unpriced(line):
        return None
    rec = recommend_side(model_prob, line)
    if not rec:
        return None
    d, p = rec["decimal_odds"], rec["p"]
    if not d or d <= 1.0:
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


# Quality safeguard (spec: "flag/log/verify anything above a configurable threshold").
# 15% by default, matching the product's own tier guide (0-2% very small … 12-15%
# exceptional, >15% rare/review). Measured on the REAL user-facing pool — standard/boosted
# odds_type only, 1,336 live props — this is a sane cutoff: mean EV +1.5%, median -0.4%,
# only ~15% of props exceed it, 0% exceed 60%. (An earlier pass over ALL scored lines,
# including demon/goblin, measured mean EV ~34% — that number was an artifact of scoring
# unpriced demon/goblin legs, which are already excluded from everything a user sees
# (dashboard._projected()) because the feed doesn't expose their real payout multiplier;
# it was never the user-facing reality.) Tune via EV_REVIEW_THRESHOLD if desired.
EV_REVIEW_THRESHOLD = float(os.getenv("EV_REVIEW_THRESHOLD", "0.15"))


def audit_ev(line: dict[str, Any], threshold: Optional[float] = None) -> Optional[dict]:
    """Flag a projection whose EV exceeds `threshold` (env EV_REVIEW_THRESHOLD by default).
    Returns None when not flagged, else a dict with the reason and the exact inputs that
    produced it — so a human reviewer can check projection / line / calibration without
    re-deriving anything. Called per-line during the build; callers should log the result."""
    th = EV_REVIEW_THRESHOLD if threshold is None else threshold
    prob = line.get("model_prob")
    if prob is None:
        return None
    rec = recommend_side(float(prob), line)
    if not rec or rec["ev"] is None or rec["ev"] <= th:
        return None
    real_implied = line.get(f"{rec['side']}_implied")
    return {
        "flagged": True, "ev": rec["ev"], "threshold": th, "side": rec["side"],
        "model_prob_for_side": round(rec["p"], 4),
        "implied_prob_used": real_implied if (real_implied and 0 < real_implied < 1) else None,
        "used_pickem_fallback": not (real_implied and 0 < real_implied < 1),
        "player": line.get("player"), "stat": line.get("stat_type"),
        "line": line.get("line"), "projection": line.get("model_proj"),
        "source": line.get("source"), "id": line.get("id"),
    }


def valuation(line: dict[str, Any]) -> dict:
    """Full valuation bundle for one projection: the recommended side + EV, Kelly, confidence,
    and Juice Score. Returns {available: False} when there's nothing to value, or the book
    offers no side of this leg that we know how to price."""
    prob = line.get("model_prob")
    proj = line.get("model_proj")
    if prob is None or proj is None or line.get("line") is None:
        return {"available": False, "reason": "No model projection/probability for this line."}
    rec = recommend_side(float(prob), line)
    if not rec:
        return {"available": False, "reason": "Neither side of this line is offered by the book."}
    implied = (1.0 / rec["decimal_odds"]) if rec.get("decimal_odds") else None
    return {
        "available": True,
        "side": rec["side"],
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
