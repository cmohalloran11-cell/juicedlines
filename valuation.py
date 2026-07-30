"""
valuation.py — JUICED 2.0 Valuation Engine

Decision-support layer for SimulationOS.

Transforms a model projection + market line into:
- expected value
- fair line
- adjusted probability
- Kelly sizing
- confidence
- Juice Score
- risk rating
- play grade
- explainability factors

Design principles:
- No model calls
- No I/O
- Deterministic
- Transparent math
- Backwards compatible with existing dashboard consumers

The valuation layer does NOT create projections.
It evaluates projections created by sport models and simulations.
"""

from __future__ import annotations

import os
from typing import Any, Optional


# ============================================================
# Constants
# ============================================================

EV_REVIEW_THRESHOLD = float(
    os.getenv("EV_REVIEW_THRESHOLD", "0.15")
)

KELLY_CAP = float(
    os.getenv("KELLY_CAP", "0.25")
)


# ============================================================
# Helpers
# ============================================================

def _clamp(
    x: float,
    lo: float = 0.0,
    hi: float = 1.0
) -> float:
    return max(lo, min(hi, x))


def _safe_float(
    x: Any,
    default: float = 0.0
) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _american_to_decimal(
    american: Any
) -> Optional[float]:
    try:
        a = float(american)
    except (TypeError, ValueError):
        return None

    if a > 0:
        return 1 + a / 100

    if a < 0:
        return 1 + 100 / abs(a)

    return None


def _side_decimal_odds(
    line: dict[str, Any],
    side: str
) -> Optional[float]:

    implied = line.get(
        f"{side}_implied"
    )

    if implied and 0 < implied < 1:
        return 1 / implied

    fallback = _american_to_decimal(
        line.get("pickem_price")
    )

    if fallback and fallback > 1:
        return fallback

    return None


# ============================================================
# Odds / pricing
# ============================================================

UNPRICED_TYPES = {
    "demon",
    "goblin"
}


def is_unpriced(
    line: dict[str, Any]
) -> bool:

    return (
        str(
            line.get("odds_type", "standard")
        ).lower()
        in UNPRICED_TYPES
    )


def offered_sides(
    line: dict[str, Any]
) -> list[str]:

    sides = []

    if _side_decimal_odds(line, "over"):
        sides.append("over")

    if _side_decimal_odds(line, "under"):
        sides.append("under")

    return sides


# ============================================================
# Probability calibration
# ============================================================

def adjusted_probability(
    line: dict[str, Any],
    probability: Optional[float] = None
) -> Optional[float]:
    """
    Converts raw model probability into a reliability-adjusted probability.

    Instead of artificially shrinking EV after calculation,
    JUICED adjusts confidence BEFORE valuation.

    Uses:
    - sample size
    - confidence score
    - optional calibration factor
    """

    if probability is None:
        probability = line.get("model_prob")

    if probability is None:
        return None

    p = float(probability)

    confidence = confidence_score(line) / 100

    calibration = line.get(
        "calibration_factor",
        1.0
    )

    # uncertainty shrink toward 50%
    uncertainty = 0.65 + (
        0.35 * confidence
    )

    adjusted = (
        0.5 +
        ((p - 0.5) * uncertainty * calibration)
    )

    return round(
        _clamp(adjusted),
        4
    )


# ============================================================
# Fair line
# ============================================================

def fair_line(
    line: dict[str, Any]
) -> Optional[float]:
    """
    Returns the model's fair market number.
    """

    projection = line.get(
        "model_proj"
    )

    if projection is None:
        return None

    return round(
        float(projection),
        3
    )


def market_edge(
    line: dict[str, Any]
) -> Optional[float]:

    projection = line.get(
        "model_proj"
    )

    market = line.get(
        "line"
    )

    if projection is None or market is None:
        return None

    return round(
        float(projection) -
        float(market),
        3
    )


# ============================================================
# EV Engine
# ============================================================

def _side_probability(
    probability: float,
    side: str
) -> float:

    if side == "under":
        return 1 - probability

    return probability


def expected_value_side(
    probability: float,
    decimal_odds: float
) -> float:

    return (
        probability *
        decimal_odds
        -
        1
    )


def recommend_side(
    model_prob: Optional[float],
    line: dict[str, Any]
) -> Optional[dict[str, Any]]:

    if model_prob is None:
        return None

    if is_unpriced(line):

        return {
            "side": "over",
            "p": float(model_prob),
            "decimal_odds": None,
            "ev": None,
        }


    candidates = []

    for side in ("over", "under"):

        odds = _side_decimal_odds(
            line,
            side
        )

        if odds is None:
            continue

        p = _side_probability(
            float(model_prob),
            side
        )

        ev = expected_value_side(
            p,
            odds
        )

        candidates.append(
            {
                "side": side,
                "p": p,
                "decimal_odds": odds,
                "ev": round(ev, 5)
            }
        )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda x: x["ev"]
    )


def expected_value(
    model_prob: float,
    line: dict[str, Any]
) -> Optional[float]:

    if model_prob is None:
        return None

    rec = recommend_side(
        model_prob,
        line
    )

    if not rec:
        return None

    return rec["ev"]

# ============================================================
# Confidence Engine
# ============================================================

def confidence_score(
    line: dict[str, Any]
) -> int:
    """
    Projection confidence, not betting confidence.

    Inputs:
    - sample size
    - probability decisiveness
    - model method
    - optional model agreement
    """

    n = _safe_float(
        line.get("model_n")
    )

    probability = line.get(
        "model_prob"
    )

    sample_component = _clamp(
        n / 100
    )

    if probability is None:
        decisiveness = 0
    else:
        decisiveness = _clamp(
            abs(
                float(probability) - .5
            ) * 2
        )

    method_component = (
        1.0
        if line.get("proj_kind") == "engine"
        else .5
    )

    agreement = _clamp(
        _safe_float(
            line.get(
                "model_agreement",
                0.5
            )
        )
    )


    score = (
        sample_component * .30
        +
        decisiveness * .35
        +
        method_component * .15
        +
        agreement * .20
    )


    return int(
        round(
            _clamp(score) * 100
        )
    )


def confidence_factors(
    line: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Human readable confidence breakdown.
    """

    n = _safe_float(
        line.get("model_n")
    )

    probability = line.get(
        "model_prob"
    )

    sample = (
        _clamp(n / 100)
        * 30
    )

    decisive = (
        _clamp(
            abs(float(probability)-.5)*2
        )
        * 35
        if probability is not None
        else 0
    )

    method = (
        15
        if line.get("proj_kind") == "engine"
        else 7.5
    )

    agreement = (
        _clamp(
            _safe_float(
                line.get(
                    "model_agreement",
                    .5
                )
            )
        )
        * 20
    )

    return [
        {
            "factor": "Sample Size",
            "value": round(sample,1),
            "max":30,
            "detail": f"{int(n)} games"
        },
        {
            "factor": "Projection Separation",
            "value": round(decisive,1),
            "max":35,
            "detail":
                "distance from 50%"
        },
        {
            "factor": "Model Method",
            "value": round(method,1),
            "max":15,
            "detail":
                "simulation engine"
                if method == 15
                else
                "basic projection"
        },
        {
            "factor": "Model Agreement",
            "value": round(agreement,1),
            "max":20,
            "detail":
                "ensemble agreement"
        }
    ]


# ============================================================
# Risk Engine
# ============================================================

def risk_score(
    line: dict[str, Any]
) -> int:
    """
    Lower = safer.

    Factors:
    - variance
    - confidence
    - sample size
    """

    confidence = confidence_score(
        line
    )

    std = line.get(
        "model_std"
    )

    if std is None:
        floor = line.get(
            "model_floor"
        )
        ceiling = line.get(
            "model_ceiling"
        )

        if floor is not None and ceiling is not None:
            std = (
                float(ceiling)
                -
                float(floor)
            ) / 2.5631


    variance_penalty = 0

    if std is not None:
        variance_penalty = min(
            40,
            float(std) * 10
        )


    confidence_penalty = (
        100-confidence
    ) * .4


    return int(
        min(
            100,
            variance_penalty
            +
            confidence_penalty
        )
    )


def risk_label(
    line: dict[str, Any]
) -> str:

    score = risk_score(
        line
    )

    if score <= 30:
        return "LOW"

    if score <= 60:
        return "MEDIUM"

    return "HIGH"



# ============================================================
# Kelly Engine
# ============================================================

def kelly_fraction(
    model_prob: float,
    line: dict[str, Any],
    cap: float = KELLY_CAP
) -> Optional[float]:
    """
    Confidence-adjusted Kelly.

    Raw Kelly is dangerous on noisy props.
    """

    if model_prob is None:
        return None


    rec = recommend_side(
        model_prob,
        line
    )

    if not rec:
        return None


    odds = rec.get(
        "decimal_odds"
    )

    if not odds:
        return 0.0


    p = rec["p"]


    raw = (
        odds*p-1
    ) / (
        odds-1
    )


    confidence_multiplier = (
        confidence_score(line)
        /
        100
    )


    risk_multiplier = (
        1 -
        risk_score(line)/200
    )


    adjusted = (
        raw
        *
        confidence_multiplier
        *
        risk_multiplier
    )


    return round(
        max(
            0,
            min(
                cap,
                adjusted
            )
        ),
        4
    )


# ============================================================
# Juice Score
# ============================================================

def juice_score(
    line: dict[str, Any],
    model_prob: Optional[float] = None
) -> int:
    """
    JUICED ranking metric.

    Combines:
    - EV
    - probability edge
    - confidence
    - reliability
    - risk
    """

    probability = (
        model_prob
        if model_prob is not None
        else line.get("model_prob")
    )

    if probability is None:
        return 0


    ev = expected_value(
        probability,
        line
    )


    ev_score = _clamp(
        abs(ev or 0) / .15
    )


    prob_score = _clamp(
        abs(
            float(probability)-.5
        )
        /
        .25
    )


    confidence = (
        confidence_score(line)
        /
        100
    )


    risk_adjustment = (
        1 -
        risk_score(line)/150
    )


    score = (
        ev_score*.30
        +
        prob_score*.25
        +
        confidence*.30
        +
        _clamp(risk_adjustment)*.15
    )


    return int(
        round(
            _clamp(score)
            *
            100
        )
    )

"""
valuation.py — SimulationOS valuation engine.

Converts an existing projection + market line into decision metrics:

- Expected Value (EV)
- Recommended side
- Kelly sizing
- Confidence
- Juice Score
- Simulation summary
- Audit flags

This module intentionally does NOT:
- fetch data
- create projections
- call models
- know sport-specific logic

It only evaluates numbers already produced upstream.

Design principles:
1. Never invent prices.
2. Never hide uncertainty.
3. Never confuse confidence in a projection with value of a bet.
4. Keep every output traceable to inputs.
"""

from __future__ import annotations

import os
from typing import Any, Optional


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

DEFAULT_KELLY_CAP = 0.25

EV_REVIEW_THRESHOLD = float(
    os.getenv("EV_REVIEW_THRESHOLD", "0.15")
)

UNPRICED_TYPES = {
    "demon",
    "goblin",
}


# ---------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------

def _clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0
) -> float:
    return max(low, min(high, value))


def _safe_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
        if x != x:
            return None
        return x
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------
# Odds conversion
# ---------------------------------------------------------

def _american_to_decimal(
    american: Any
) -> Optional[float]:

    a = _safe_float(american)

    if a is None:
        return None

    if a > 0:
        return 1 + a / 100

    if a < 0:
        return 1 + 100 / abs(a)

    return None


def is_unpriced(
    line: dict[str, Any]
) -> bool:

    return (
        str(line.get("odds_type", "standard"))
        .lower()
        in UNPRICED_TYPES
    )


def _side_decimal_odds(
    line: dict[str, Any],
    side: str
) -> Optional[float]:
    """
    Returns actual decimal odds available for a side.

    Priority:
    1. side-specific implied probability
    2. pick'em fallback price

    Never fabricates a missing side.
    """

    implied = _safe_float(
        line.get(f"{side}_implied")
    )

    if implied and 0 < implied < 1:
        return 1 / implied


    pickem = _american_to_decimal(
        line.get("pickem_price")
    )

    if pickem and pickem > 1:
        return pickem


    return None

# ---------------------------------------------------------
# Side recommendation
# ---------------------------------------------------------

def recommend_side(
    model_prob: Optional[float],
    line: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """
    Selects the side with the highest expected value.

    Important:
    This does NOT simply choose over when model_prob > .50.

    Example:
    - Over probability: 60%
    - Under probability: 40%

    If the Over pays poorly and Under pays better,
    Under can still be the mathematically superior choice.

    Returns:
    {
        side,
        p,
        decimal_odds,
        ev
    }
    """

    if model_prob is None:
        return None

    p_over = _safe_float(model_prob)

    if p_over is None:
        return None


    # Demon/goblin:
    # PrizePicks does not expose the multiplier,
    # therefore no EV calculation is possible.
    # The only selectable direction is over.

    if is_unpriced(line):
        return {
            "side": "over",
            "p": p_over,
            "decimal_odds": None,
            "ev": None,
        }


    candidates = []


    # --------------------
    # OVER
    # --------------------

    over_odds = _side_decimal_odds(
        line,
        "over"
    )

    if over_odds:

        candidates.append({
            "side": "over",
            "p": p_over,
            "decimal_odds": over_odds,
            "ev": round(
                p_over * over_odds - 1,
                4
            ),
        })


    # --------------------
    # UNDER
    # --------------------

    under_odds = _side_decimal_odds(
        line,
        "under"
    )

    if under_odds:

        p_under = 1 - p_over

        candidates.append({
            "side": "under",
            "p": p_under,
            "decimal_odds": under_odds,
            "ev": round(
                p_under * under_odds - 1,
                4
            ),
        })


    if not candidates:
        return None


    return max(
        candidates,
        key=lambda x: x["ev"]
    )


# ---------------------------------------------------------
# Expected Value
# ---------------------------------------------------------

def expected_value(
    model_prob: Optional[float],
    line: dict[str, Any]
) -> Optional[float]:
    """
    EV per $1 wager.

    +0.05 = +5% expected return
    -0.03 = -3% expected return

    Returns None when:
    - no projection probability
    - no available price
    - unpriced promo
    """

    if model_prob is None:
        return None


    if is_unpriced(line):
        return None


    rec = recommend_side(
        model_prob,
        line
    )

    if not rec:
        return None


    return rec["ev"]



# ---------------------------------------------------------
# Kelly Criterion
# ---------------------------------------------------------

def kelly_fraction(
    model_prob: Optional[float],
    line: dict[str, Any],
    cap: float = DEFAULT_KELLY_CAP
) -> Optional[float]:
    """
    Fractional Kelly sizing.

    Uses quarter Kelly by default because prop models
    contain estimation error.

    Formula:
        f = (bp-q)/b

    Equivalent:
        (decimal*p - 1)/(decimal-1)

    """

    if model_prob is None:
        return None


    if is_unpriced(line):
        return None


    rec = recommend_side(
        model_prob,
        line
    )


    if not rec:
        return None


    odds = rec.get(
        "decimal_odds"
    )

    p = rec.get(
        "p"
    )


    if not odds or odds <= 1:
        return 0.0


    raw = (
        (odds * p - 1)
        /
        (odds - 1)
    )


    return round(
        _clamp(
            raw,
            0,
            cap
        ),
        4
    )

# ---------------------------------------------------------
# Confidence Engine
# ---------------------------------------------------------

def confidence_score(
    line: dict[str, Any]
) -> int:
    """
    Confidence in the PROJECTION.

    This is NOT:
        "How good is this bet?"

    It measures:
        - data reliability
        - projection certainty
        - model quality

    Components:

    Sample size      35%
    Probability      35%
    Method quality   30%

    The old system over-weighted sample size,
    causing every engine projection with enough
    games to look artificially confident.
    """

    n = _safe_float(
        line.get("model_n")
    ) or 0


    prob = _safe_float(
        line.get("model_prob")
    )


    # --------------------
    # Sample reliability
    # --------------------

    sample_factor = _clamp(
        n / 50
    )


    # --------------------
    # Probability separation
    # --------------------

    if prob is None:
        probability_factor = 0
    else:
        probability_factor = _clamp(
            abs(prob - .5) * 2
        )


    # --------------------
    # Method quality
    # --------------------

    kind = (
        line.get("proj_kind")
        or ""
    ).lower()


    if kind == "engine":
        method_factor = 1.0

    elif kind in {
        "ensemble",
        "simulation",
    }:
        method_factor = 0.9

    else:
        method_factor = 0.65



    score = (
        35 * sample_factor
        +
        35 * probability_factor
        +
        30 * method_factor
    )


    return int(
        round(
            _clamp(score / 100) * 100
        )
    )



def confidence_factors(
    line: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    UI breakdown of confidence score.
    """

    n = int(
        line.get("model_n")
        or 0
    )

    prob = _safe_float(
        line.get("model_prob")
    )


    sample = 35 * _clamp(
        n / 50
    )


    probability = (
        0
        if prob is None
        else
        35 * _clamp(
            abs(prob - .5) * 2
        )
    )


    kind = (
        line.get("proj_kind")
        or ""
    ).lower()


    method = (
        30
        if kind == "engine"
        else
        27
        if kind in {"simulation", "ensemble"}
        else
        20
    )


    return [
        {
            "factor": "Sample Size",
            "value": round(sample, 1),
            "max": 35,
            "detail": f"{n} historical samples"
        },
        {
            "factor": "Probability Separation",
            "value": round(probability, 1),
            "max": 35,
            "detail":
                "distance from 50/50"
        },
        {
            "factor": "Projection Method",
            "value": method,
            "max": 30,
            "detail":
                "model architecture quality"
        }
    ]



# ---------------------------------------------------------
# Juice Score
# ---------------------------------------------------------

def juice_score(
    line: dict[str, Any],
    model_prob: Optional[float] = None
) -> int:
    """
    Measures opportunity quality.

    NOT a guarantee.

    Juice =
        value confidence
        × projection confidence
        × probability separation

    High Juice requires:
        - model disagreement with market
        - reliable projection
        - strong probability edge
    """

    prob = (
        model_prob
        if model_prob is not None
        else line.get("model_prob")
    )


    if prob is None:
        return 0


    ev = expected_value(
        prob,
        line
    )


    if ev is None:
        return 0


    confidence = (
        confidence_score(line)
        /
        100
    )


    probability_edge = _clamp(
        abs(float(prob) - .5) * 2
    )


    # EV contribution:
    # 15% EV is considered elite.
    ev_factor = _clamp(
        abs(ev) / .15
    )


    score = (
        .45 * ev_factor
        +
        .30 * probability_edge
        +
        .25 * confidence
    )


    return int(
        round(
            _clamp(score)
            *
            100
        )
    )



# ---------------------------------------------------------
# Simulation Summary
# ---------------------------------------------------------

def _std_from_band(
    line: dict[str, Any]
) -> Optional[float]:

    floor = line.get(
        "model_floor"
    )

    ceiling = line.get(
        "model_ceiling"
    )


    if floor is None or ceiling is None:
        return None


    if ceiling <= floor:
        return None


    # p10 -> p90 covers ~2.56 sigma
    return round(
        (
            float(ceiling)
            -
            float(floor)
        )
        /
        2.5631,
        3
    )



def simulation_object(
    line: dict[str, Any]
) -> Optional[dict]:

    if line.get("model_proj") is None:
        return None


    prob = line.get(
        "model_prob"
    )


    return {
        "projectionId":
            line.get("id"),

        "mean":
            line.get("model_proj"),

        "median":
            line.get(
                "model_median",
                line.get("model_proj")
            ),

        "standardDeviation":
            _std_from_band(line),

        "floor":
            line.get("model_floor"),

        "ceiling":
            line.get("model_ceiling"),

        "p25":
            line.get("p25"),

        "p75":
            line.get("p75"),

        "overProbability":
            prob,

        "underProbability":
            (
                round(
                    1 - float(prob),
                    3
                )
                if prob is not None
                else None
            ),

        "sampleSize":
            line.get("model_n")
    }

# ---------------------------------------------------------
# EV Audit
# ---------------------------------------------------------

def audit_ev(
    line: dict[str, Any],
    threshold: Optional[float] = None
) -> Optional[dict]:
    """
    Flags unusually large EV estimates.

    This does NOT remove plays.

    It creates a review trail so extreme
    outputs can be investigated for:
        - bad calibration
        - stale lines
        - small sample sizes
        - projection bugs
    """

    threshold = (
        EV_REVIEW_THRESHOLD
        if threshold is None
        else threshold
    )


    prob = line.get(
        "model_prob"
    )

    if prob is None:
        return None


    rec = recommend_side(
        float(prob),
        line
    )


    if not rec:
        return None


    ev = rec.get(
        "ev"
    )


    if ev is None or ev <= threshold:
        return None


    implied = line.get(
        f"{rec['side']}_implied"
    )


    return {
        "flagged": True,

        "reason":
            "EV exceeds review threshold",

        "ev":
            round(ev, 4),

        "threshold":
            threshold,

        "side":
            rec["side"],

        "model_probability":
            round(
                rec["p"],
                4
            ),

        "market_probability":
            implied,

        "player":
            line.get("player"),

        "sport":
            line.get("sport"),

        "stat":
            line.get("stat_type"),

        "line":
            line.get("line"),

        "projection":
            line.get("model_proj"),

        "source":
            line.get("source"),

        "id":
            line.get("id"),
    }



# ---------------------------------------------------------
# Full valuation object
# ---------------------------------------------------------

def valuation(
    line: dict[str, Any]
) -> dict:
    """
    Complete valuation payload.

    Used by dashboard/API layers.

    Returns:
    {
        available,
        side,
        EV,
        Kelly,
        confidence,
        Juice Score
    }
    """

    prob = line.get(
        "model_prob"
    )

    projection = line.get(
        "model_proj"
    )


    if (
        prob is None
        or projection is None
        or line.get("line") is None
    ):
        return {
            "available": False,
            "reason":
                "Missing projection data"
        }



    rec = recommend_side(
        float(prob),
        line
    )


    if not rec:
        return {
            "available": False,
            "reason":
                "No playable side available"
        }


    implied = None

    if rec.get("decimal_odds"):
        implied = round(
            1 / rec["decimal_odds"],
            4
        )


    return {

        "available": True,


        "side":
            rec["side"],


        "line":
            line.get("line"),


        "projection":
            projection,


        "edge":
            line.get("model_edge"),


        "probability":
            prob,


        "impliedProbability":
            implied,


        "expectedValue":
            expected_value(
                prob,
                line
            ),


        "kellyFraction":
            kelly_fraction(
                prob,
                line
            ),


        "confidence":
            confidence_score(line),


        "confidenceFactors":
            confidence_factors(line),


        "juiceScore":
            juice_score(line),


        "simulation":
            simulation_object(line),

    }
