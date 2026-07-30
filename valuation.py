"""
valuation.py — JUICED 2.0 Valuation Engine

SimulationOS decision layer.

Converts an existing projection + market line into:
- expected value
- fair value
- recommended side
- Kelly sizing
- confidence
- Juice Score
- risk rating
- simulation summary
- audit information

This module DOES NOT:
- fetch data
- create projections
- call APIs
- know sport rules

It only evaluates numbers created upstream.

Design rules:
1. Never invent market prices.
2. Never hide uncertainty.
3. Never confuse projection confidence with bet confidence.
4. Every output must be traceable to inputs.
"""

from __future__ import annotations

import os
from typing import Any, Optional


# ============================================================
# Constants
# ============================================================

DEFAULT_KELLY_CAP = 0.25

EV_REVIEW_THRESHOLD = float(
    os.getenv(
        "EV_REVIEW_THRESHOLD",
        "0.15"
    )
)

UNPRICED_TYPES = {
    "demon",
    "goblin",
}


# ============================================================
# Basic Helpers
# ============================================================

def _clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0
) -> float:
    return max(
        low,
        min(
            high,
            value
        )
    )


def _safe_float(
    value: Any,
    default: Optional[float] = None
) -> Optional[float]:

    try:
        result = float(value)

        if result != result:
            return default

        return result

    except (
        TypeError,
        ValueError
    ):
        return default



# ============================================================
# Odds Conversion
# ============================================================

def _american_to_decimal(
    american: Any
) -> Optional[float]:

    odds = _safe_float(
        american
    )

    if odds is None:
        return None


    if odds > 0:
        return 1 + (
            odds / 100
        )


    if odds < 0:
        return 1 + (
            100 / abs(odds)
        )


    return None



def _decimal_to_implied(
    decimal: float
) -> float:

    return round(
        1 / decimal,
        4
    )



def is_unpriced(
    line: dict[str, Any]
) -> bool:

    return (
        str(
            line.get(
                "odds_type",
                "standard"
            )
        )
        .lower()
        in UNPRICED_TYPES
    )



def _side_decimal_odds(
    line: dict[str, Any],
    side: str
) -> Optional[float]:
    """
    Finds the actual available payout.

    Priority:

    1. sportsbook implied probability
    2. PrizePicks standard pick'em price

    Never assumes a price that does not exist.
    """


    implied = _safe_float(
        line.get(
            f"{side}_implied"
        )
    )


    if (
        implied is not None
        and
        0 < implied < 1
    ):
        return 1 / implied



    pickem = _american_to_decimal(
        line.get(
            "pickem_price"
        )
    )


    if pickem is not None:
        return pickem


    return None



def offered_sides(
    line: dict[str, Any]
) -> list[str]:

    sides = []


    for side in (
        "over",
        "under"
    ):

        if _side_decimal_odds(
            line,
            side
        ):

            sides.append(
                side
            )


    return sides



# ============================================================
# Market Math
# ============================================================

def fair_line(
    line: dict[str, Any]
) -> Optional[float]:
    """
    Model fair number.
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


    if (
        projection is None
        or
        market is None
    ):
        return None


    return round(
        float(projection)
        -
        float(market),
        3
    )



# ============================================================
# End Part 1
# ============================================================

# ============================================================
# Expected Value Engine
# ============================================================


def _side_probability(
    probability: float,
    side: str
) -> float:
    """
    Converts over probability into side probability.
    """

    if side == "under":
        return 1 - probability

    return probability



def expected_value_side(
    probability: float,
    decimal_odds: float
) -> float:
    """
    EV per $1 wager.

    Example:
        +0.05 = +5% expected return
    """

    return (
        probability * decimal_odds
    ) - 1



def recommend_side(
    model_prob: Optional[float],
    line: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """
    Determines the mathematically best available side.

    Does NOT blindly choose:
        probability > .50 = over

    Instead compares:

        Over EV
        Under EV

    and returns whichever has higher expected value.

    Returns:

    {
        "side": "over",
        "p": 0.61,
        "decimal_odds": 1.91,
        "ev": 0.165
    }
    """


    if model_prob is None:
        return None


    probability = _safe_float(
        model_prob
    )

    if probability is None:
        return None



    # --------------------------------------------------------
    # Promo lines
    # --------------------------------------------------------
    #
    # Demon/goblin lines do not expose payout multipliers.
    # They cannot have honest EV calculated.
    #
    # The only valid selectable side is over.
    #

    if is_unpriced(line):

        return {
            "side": "over",
            "p": probability,
            "decimal_odds": None,
            "ev": None,
        }



    candidates = []



    # --------------------------------------------------------
    # Over
    # --------------------------------------------------------

    over_odds = _side_decimal_odds(
        line,
        "over"
    )


    if over_odds:

        candidates.append(
            {
                "side": "over",
                "p": probability,
                "decimal_odds": over_odds,
                "ev": round(
                    expected_value_side(
                        probability,
                        over_odds
                    ),
                    5
                )
            }
        )



    # --------------------------------------------------------
    # Under
    # --------------------------------------------------------

    under_odds = _side_decimal_odds(
        line,
        "under"
    )


    if under_odds:

        under_probability = (
            1 - probability
        )

        candidates.append(
            {
                "side": "under",
                "p": under_probability,
                "decimal_odds": under_odds,
                "ev": round(
                    expected_value_side(
                        under_probability,
                        under_odds
                    ),
                    5
                )
            }
        )



    if not candidates:
        return None



    return max(
        candidates,
        key=lambda x: x["ev"]
    )



def expected_value(
    model_prob: Optional[float],
    line: dict[str, Any]
) -> Optional[float]:
    """
    Returns expected value.

    Returns None when:

    - no probability exists
    - no playable price exists
    - promo has unknown multiplier
    """


    if model_prob is None:
        return None


    if is_unpriced(line):
        return None



    recommendation = recommend_side(
        model_prob,
        line
    )


    if not recommendation:
        return None



    return recommendation["ev"]



# ============================================================
# Probability Adjustment Engine
# ============================================================


def adjusted_probability(
    line: dict[str, Any],
    probability: Optional[float] = None
) -> Optional[float]:
    """
    Reliability-adjusted probability.

    Raw model probabilities are often too extreme.

    This moves uncertain projections toward 50%.

    Inputs:
        - model probability
        - sample size
        - confidence
        - calibration factor

    """

    if probability is None:
        probability = line.get(
            "model_prob"
        )


    probability = _safe_float(
        probability
    )


    if probability is None:
        return None



    confidence = (
        confidence_score(line)
        /
        100
    )



    calibration = _safe_float(
        line.get(
            "calibration_factor",
            1.0
        ),
        1.0
    )



    # More uncertain models shrink harder.
    uncertainty_multiplier = (
        0.65
        +
        (
            0.35
            *
            confidence
        )
    )



    adjusted = (
        0.5
        +
        (
            (
                probability
                -
                0.5
            )
            *
            uncertainty_multiplier
            *
            calibration
        )
    )


    return round(
        _clamp(
            adjusted
        ),
        4
    )



# ============================================================
# Confidence Engine
# ============================================================


def confidence_score(
    line: dict[str, Any]
) -> int:
    """
    Measures confidence in the PROJECTION.

    NOT:
        "How good is this bet?"

    Measures:

    - sample size
    - probability separation
    - projection methodology
    - model agreement
    """

    n = _safe_float(
        line.get(
            "model_n"
        ),
        0
    )



    probability = _safe_float(
        line.get(
            "model_prob"
        )
    )



    sample_component = _clamp(
        n / 50
    )



    if probability is None:

        probability_component = 0

    else:

        probability_component = _clamp(
            abs(
                probability - 0.5
            )
            *
            2
        )



    method = (
        line.get(
            "proj_kind"
        )
        or ""
    ).lower()



    if method == "engine":

        method_component = 1.0

    elif method in (
        "simulation",
        "ensemble"
    ):

        method_component = 0.9

    else:

        method_component = 0.65



    agreement = _safe_float(
        line.get(
            "model_agreement",
            0.5
        ),
        0.5
    )


    agreement_component = _clamp(
        agreement
    )



    score = (
        sample_component * 0.35
        +
        probability_component * 0.35
        +
        method_component * 0.20
        +
        agreement_component * 0.10
    )



    return int(
        round(
            _clamp(score)
            *
            100
        )
    )



def confidence_factors(
    line: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Dashboard explanation data.
    """

    n = int(
        _safe_float(
            line.get(
                "model_n"
            ),
            0
        )
    )


    probability = _safe_float(
        line.get(
            "model_prob"
        )
    )


    sample = (
        _clamp(
            n / 50
        )
        *
        35
    )


    probability_score = (
        0
        if probability is None
        else
        _clamp(
            abs(
                probability - .5
            )
            *
            2
        )
        *
        35
    )



    method = (
        line.get(
            "proj_kind"
        )
        or ""
    ).lower()



    if method == "engine":

        method_score = 20

        method_detail = (
            "simulation engine"
        )

    elif method in (
        "simulation",
        "ensemble"
    ):

        method_score = 18

        method_detail = (
            "ensemble simulation"
        )

    else:

        method_score = 13

        method_detail = (
            "basic projection"
        )



    agreement = (
        _clamp(
            _safe_float(
                line.get(
                    "model_agreement",
                    .5
                ),
                .5
            )
        )
        *
        10
    )



    return [
        {
            "factor": "Sample Size",
            "value": round(sample,1),
            "max":35,
            "detail": f"{n} samples"
        },
        {
            "factor": "Probability Separation",
            "value": round(probability_score,1),
            "max":35,
            "detail": "distance from 50%"
        },
        {
            "factor": "Projection Method",
            "value": method_score,
            "max":20,
            "detail": method_detail
        },
        {
            "factor": "Model Agreement",
            "value": round(agreement,1),
            "max":10,
            "detail": "ensemble consistency"
        }
    ]

# ============================================================
# Risk Engine
# ============================================================


def _std_from_distribution(
    line: dict[str, Any]
) -> Optional[float]:
    """
    Attempts to recover standard deviation.

    Priority:

    1. Existing model_std
    2. p10-p90 range approximation
    """

    std = _safe_float(
        line.get(
            "model_std"
        )
    )

    if std is not None:
        return std



    floor = _safe_float(
        line.get(
            "model_floor"
        )
    )

    ceiling = _safe_float(
        line.get(
            "model_ceiling"
        )
    )


    if floor is None or ceiling is None:
        return None


    if ceiling <= floor:
        return None



    # p10 -> p90 contains roughly 2.563 sigma
    return round(
        (
            ceiling
            -
            floor
        )
        /
        2.5631,
        3
    )



def risk_score(
    line: dict[str, Any]
) -> int:
    """
    Betting uncertainty score.

    Lower = safer.

    Uses:

    - distribution width
    - projection confidence
    """

    confidence = confidence_score(
        line
    )

    std = _std_from_distribution(
        line
    )


    variance_penalty = 0


    if std is not None:

        variance_penalty = min(
            50,
            std * 10
        )


    confidence_penalty = (
        100 - confidence
    ) * .5



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
# Kelly Criterion
# ============================================================


def kelly_fraction(
    model_prob: Optional[float],
    line: dict[str, Any],
    cap: float = DEFAULT_KELLY_CAP
) -> Optional[float]:
    """
    Fractional Kelly sizing.

    Used as a theoretical sizing guide.

    Prop markets have:
    - model error
    - injury uncertainty
    - correlation risk

    Therefore capped.
    """

    if model_prob is None:
        return None


    if is_unpriced(line):
        return None



    recommendation = recommend_side(
        model_prob,
        line
    )


    if not recommendation:
        return None



    odds = recommendation.get(
        "decimal_odds"
    )

    probability = recommendation.get(
        "p"
    )


    if (
        not odds
        or probability is None
        or odds <= 1
    ):
        return 0.0



    raw_kelly = (
        (
            odds
            *
            probability
            -
            1
        )
        /
        (
            odds
            -
            1
        )
    )



    # Never allow negative sizing.
    # Never exceed cap.

    return round(
        max(
            0,
            min(
                cap,
                raw_kelly
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

    Measures opportunity quality.

    Components:

    45% expected value
    30% probability separation
    25% projection confidence

    This is NOT win probability.
    """

    probability = (
        model_prob
        if model_prob is not None
        else line.get(
            "model_prob"
        )
    )


    if probability is None:
        return 0



    ev = expected_value(
        probability,
        line
    )


    if ev is None:
        return 0



    ev_factor = _clamp(
        abs(ev)
        /
        .15
    )



    probability_factor = _clamp(
        abs(
            float(probability)
            -
            .5
        )
        *
        2
    )



    confidence_factor = (
        confidence_score(line)
        /
        100
    )



    score = (
        ev_factor * .45
        +
        probability_factor * .30
        +
        confidence_factor * .25
    )



    return int(
        round(
            _clamp(score)
            *
            100
        )
    )




# ============================================================
# Simulation Summary
# ============================================================


def simulation_object(
    line: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """
    Returns distribution summary for UI.
    """

    projection = line.get(
        "model_proj"
    )


    if projection is None:
        return None



    probability = line.get(
        "model_prob"
    )


    return {

        "projectionId":
            line.get(
                "id"
            ),


        "mean":
            projection,


        "median":
            line.get(
                "model_median",
                projection
            ),


        "floor":
            line.get(
                "model_floor"
            ),


        "ceiling":
            line.get(
                "model_ceiling"
            ),


        "p25":
            line.get(
                "p25"
            ),


        "p75":
            line.get(
                "p75"
            ),


        "standardDeviation":
            _std_from_distribution(
                line
            ),


        "overProbability":
            probability,


        "underProbability":
            (
                round(
                    1 - float(probability),
                    4
                )
                if probability is not None
                else None
            ),


        "sampleSize":
            line.get(
                "model_n"
            )
    }




# ============================================================
# EV Audit
# ============================================================


def audit_ev(
    line: dict[str, Any],
    threshold: Optional[float] = None
) -> Optional[dict[str, Any]]:
    """
    Flags suspiciously large EV outputs.

    Does not remove plays.

    Creates investigation records for:

    - stale lines
    - calibration problems
    - projection errors
    - bad inputs
    """

    threshold = (
        EV_REVIEW_THRESHOLD
        if threshold is None
        else threshold
    )


    probability = line.get(
        "model_prob"
    )


    if probability is None:
        return None



    recommendation = recommend_side(
        probability,
        line
    )


    if not recommendation:
        return None



    ev = recommendation.get(
        "ev"
    )


    if ev is None:
        return None


    if ev <= threshold:
        return None



    side = recommendation["side"]


    return {

        "flagged":
            True,


        "reason":
            "EV exceeds review threshold",


        "ev":
            round(
                ev,
                4
            ),


        "threshold":
            threshold,


        "side":
            side,


        "modelProbability":
            round(
                recommendation["p"],
                4
            ),


        "marketProbability":
            line.get(
                f"{side}_implied"
            ),


        "player":
            line.get(
                "player"
            ),


        "sport":
            line.get(
                "sport"
            ),


        "stat":
            line.get(
                "stat_type"
            ),


        "line":
            line.get(
                "line"
            ),


        "projection":
            line.get(
                "model_proj"
            ),


        "source":
            line.get(
                "source"
            ),


        "id":
            line.get(
                "id"
            )
    }

# ============================================================
# Complete Valuation Payload
# ============================================================


def valuation(
    line: dict[str, Any]
) -> dict[str, Any]:
    """
    Complete JUICED valuation object.

    This is the object API/dashboard layers can consume.

    Contains:

    - recommended side
    - EV
    - Kelly
    - confidence
    - risk
    - Juice Score
    - simulation distribution
    - audit flags
    """

    probability = line.get(
        "model_prob"
    )


    projection = line.get(
        "model_proj"
    )



    if (
        probability is None
        or projection is None
        or line.get("line") is None
    ):

        return {
            "available": False,
            "reason":
                "Missing projection data"
        }



    recommendation = recommend_side(
        probability,
        line
    )


    if recommendation is None:

        return {
            "available": False,
            "reason":
                "No playable market side"
        }



    decimal_odds = recommendation.get(
        "decimal_odds"
    )


    implied_probability = None


    if decimal_odds:

        implied_probability = round(
            1 / decimal_odds,
            4
        )



    return {

        "available":
            True,


        "id":
            line.get(
                "id"
            ),


        "player":
            line.get(
                "player"
            ),


        "sport":
            line.get(
                "sport"
            ),


        "stat":
            line.get(
                "stat_type"
            ),


        "side":
            recommendation.get(
                "side"
            ),


        "line":
            line.get(
                "line"
            ),


        "projection":
            projection,


        "fairLine":
            fair_line(
                line
            ),


        "marketEdge":
            market_edge(
                line
            ),


        "probability":
            probability,


        "adjustedProbability":
            adjusted_probability(
                line,
                probability
            ),


        "impliedProbability":
            implied_probability,


        "expectedValue":
            expected_value(
                probability,
                line
            ),


        "kellyFraction":
            kelly_fraction(
                probability,
                line
            ),


        "confidence":
            confidence_score(
                line
            ),


        "confidenceFactors":
            confidence_factors(
                line
            ),


        "riskScore":
            risk_score(
                line
            ),


        "riskLabel":
            risk_label(
                line
            ),


        "juiceScore":
            juice_score(
                line
            ),


        "simulation":
            simulation_object(
                line
            ),


        "audit":
            audit_ev(
                line
            )
    }




# ============================================================
# Compatibility Helpers
# ============================================================


def edge_percent(
    line: dict[str, Any]
) -> Optional[float]:
    """
    Dashboard helper.

    Converts model probability EV
    into display percentage.
    """

    ev = expected_value(
        line.get(
            "model_prob"
        ),
        line
    )


    if ev is None:
        return None


    return round(
        ev * 100,
        1
    )



def play_grade(
    line: dict[str, Any]
) -> str:
    """
    Human-readable opportunity grade.

    Used for UI labels.
    """

    juice = juice_score(
        line
    )


    confidence = confidence_score(
        line
    )


    if (
        juice >= 85
        and
        confidence >= 75
    ):
        return "A"



    if (
        juice >= 70
        and
        confidence >= 60
    ):
        return "B"



    if juice >= 50:
        return "C"



    return "D"




# ============================================================
# End valuation.py
# ============================================================
