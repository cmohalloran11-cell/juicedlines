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


def _dampen_ev(ev: float) -> float:
    """Compress the tail of a raw EV estimate toward a believable range for DISPLAY —
    2026-07-30, after live examples showed props reading +50-77% EV. Real, validated
    edges rarely exceed ~10% (the product's own tier guide calls >15% "extremely rare");
    a raw EV far above that is usually at least partly a calibration artifact — an
    overconfident probability estimate, a thin-data multiplier, or a brand-new model with
    no graded track record yet (tennis's live-Elo layer, shipped this session, has none).
    This is honest uncertainty-shrinkage, the same idea as MLB's own Platt calibration
    (shrink toward less confident), just applied at the EV step instead of the probability
    step so it covers every sport uniformly. Order-preserving (same sign, monotonic in
    magnitude), so it never changes which side recommend_side() picks — only the number
    shown for it. Does NOT touch the probability/Kelly math, only this display field."""
    threshold = 0.08
    mag = abs(ev)
    if mag <= threshold:
        return ev
    compressed = threshold + (mag - threshold) * 0.15   # the tail counts for 15% of its raw size
    return compressed if ev > 0 else -compressed


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
                           "ev": _dampen_ev(round(p * over_d - 1.0, 4))})
    under_d = _side_decimal_odds(line, "under")
    if under_d is not None:
        q = 1.0 - p
        candidates.append({"side": "under", "p": q, "decimal_odds": under_d,
                           "ev": _dampen_ev(round(q * under_d - 1.0, 4))})
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


# Every proj_kind tag that represents a genuine full Monte Carlo/simulation run — not just
# MLB's "engine" (2026-07-30: found WNBA's "basketball" and tennis's "tennis" are ALSO real
# per-possession/per-point simulations, but confidence_score's method bonus only ever
# recognized "engine", silently docking both sports 10 points (0.20 weight × 0.5 lost) their
# entire time on the board — a genuine bug, not a deliberate demotion. "model" (MLB's
# empirical-average fallback) and "market" (tennis fully deferred to the market line, no real
# model call) correctly stay at the lower 0.5 — they aren't full simulations.
# 2026-08: NFL ships two kinds rather than one — "nfl_regular" and "nfl_preseason" — because
# the two are genuinely different models (a preseason projection's playing time comes from
# rotation tiers, not from the player's own snap history), and a calibration query must be
# able to separate them. Both ARE full Monte Carlo runs, so both belong here.
_FULL_ENGINE_KINDS = frozenset({"engine", "basketball", "tennis", "nfl_regular", "nfl_preseason"})


def confidence_score(line: dict[str, Any]) -> int:
    """
    0–100 confidence in the projection itself (NOT how good the bet is). Blends:
      • sample size (more games → steadier estimate), saturating at ~30 games,
      • decisiveness of P(over) (how far the model is from a coin flip),
      • method (a full engine run with a distribution beats a bare empirical average).
    Heuristic and deliberately transparent — see the weights below.

    Weights re-tuned 2026-07-29 (projection-realism pass): the previous split (50/30/20)
    gave sample-size + method a combined 70-point FLOOR whenever n≥30 games and
    proj_kind=="engine" — true for most MLB props — so confidence was stuck in a
    compressed 70-100 band regardless of how genuinely uncertain the matchup was (a real
    coin-flip prop scored the same ~70 as a lopsided one). Decisiveness now carries the
    plurality of the weight, so a well-sampled engine prop that's a real toss-up scores a
    real ~50 ("medium"), not a floor-guaranteed ~70.
    """
    n = line.get("model_n") or 0
    prob = line.get("model_prob")
    n_factor = _clamp(n / 30.0)
    prob_factor = _clamp(2.0 * abs(float(prob) - 0.5)) if prob is not None else 0.0
    method_factor = 1.0 if line.get("proj_kind") in _FULL_ENGINE_KINDS else 0.5
    score = 100.0 * (0.30 * n_factor + 0.50 * prob_factor + 0.20 * method_factor)
    return int(round(_clamp(score / 100.0) * 100))


def confidence_factors(line: dict[str, Any]) -> list[dict[str, Any]]:
    """The REAL decomposition of `confidence_score` — the actual weighted contributions
    (each 0..max), so the UI can show what drives a prop's confidence instead of a constant.
    The three values sum to the confidence score (max 30 + 50 + 20 = 100)."""
    n = int(line.get("model_n") or 0)
    prob = line.get("model_prob")
    n_factor = _clamp(n / 30.0)
    prob_factor = _clamp(2.0 * abs(float(prob) - 0.5)) if prob is not None else 0.0
    method_factor = 1.0 if line.get("proj_kind") in _FULL_ENGINE_KINDS else 0.5
    return [
        {"factor": "Sample Size", "value": round(30.0 * n_factor, 1), "max": 30,
         "detail": f"{n} game{'' if n == 1 else 's'} of history"},
        {"factor": "Decisiveness", "value": round(50.0 * prob_factor, 1), "max": 50,
         "detail": "how far P(over) is from a coin-flip"},
        {"factor": "Method", "value": round(20.0 * method_factor, 1), "max": 20,
         "detail": "full engine run" if method_factor == 1.0 else "empirical average"},
    ]


def _scale(x: Optional[float], lo: float, hi: float, default: float = 0.5) -> float:
    """Linearly map x from [lo,hi] to [0,1], clamped. lo>hi inverts (lower x -> higher score).
    `default` (0.5, neutral — neither rewarded nor punished) is returned when x is unknown,
    so a prop missing one signal isn't dragged toward 0."""
    if x is None:
        return default
    if hi == lo:
        return default
    return _clamp((float(x) - lo) / (hi - lo))


# 2026-07-30 Juice Score rebuild. Previously Juice Score = decisiveness × confidence — which
# made it read as just a re-skinned Confidence (same complaint the product itself anticipated:
# "why do I need Juice Score if I already have Confidence?"). Confidence answers "how much do
# we trust the projection"; EV answers "how much value does the market offer"; Juice Score
# should answer a THIRD, genuinely different question — "given everything we know, how
# attractive is this prop overall" — built from components neither Confidence nor EV alone
# capture: is the projection stable, does the model agree with the market's own read, is this
# line actually cross-validated by other books, is the edge meaningful relative to the line
# itself, and is the underlying data complete. Every component below is computed from a REAL
# field already on the line object — nothing here is fabricated. One deliberate substitution:
# the product spec called for "4 independent model estimates" (simulation/Bayesian/regression/
# historical) for a Model Agreement component — that ensemble doesn't exist in this codebase
# (one model per sport). Tennis already computes a genuine 3-way model_agreement (simulation
# vs Elo vs closed-form serve/return model — see tennis/projections.py). MLB/WNBA don't have
# that, so their Model Agreement instead compares the model's RAW opinion (model_raw, the
# pre-market-anchor mean) against the final blended projection — a real, if narrower,
# "does the model's own read agree with what the market nudged it toward" signal.
# 2026-08-05 rebalance: proj_diff (the edge's size relative to the line — formerly
# "line_value" at a mere 5%) is now the DOMINANT factor at 35%, on explicit product
# direction that Juice Score should go a lot more off of how far the projection sits from
# the book's line. Every other weight scaled down by the same factor (0.65/0.95 ≈ 0.684) so
# their RELATIVE proportions to each other are unchanged — EV is still the second-largest
# factor, Confidence third, etc. — only proj_diff's share of the total grew. The base formula
# is still % of the line, not a raw diff or a volatility-normalized z-score — see
# _juice_components below for why: a 1.6 edge on an 18.5 line means more than the same 1.6 on
# a 3.5 line, and this is already safe for near-zero lines via the 0.5 floor. (2026-08-17: now
# also discounted by decisiveness — see _juice_components — so a big raw gap that doesn't
# actually move P(over) off a coin flip no longer scores at full value.)
_JUICE_WEIGHTS = {
    "proj_diff": 0.35,
    "ev": 0.21, "confidence": 0.17,
    "stability": 0.10,   # merged Projection Stability (10%) + Volatility (5%) — see below
    "agreement": 0.07, "market_quality": 0.07, "data_quality": 0.03,
}


def _juice_components(line: dict[str, Any], model_prob: Optional[float]) -> dict[str, tuple]:
    """(score 0-1, human detail string) for every Juice Score component."""
    out: dict[str, tuple] = {}

    # 1. Expected Value — the recommended side's real (dampened) EV. Unpriced (demon/goblin)
    # legs have no EV to reward or punish, so they get the neutral default, not a penalty for
    # a number that was never computable in the first place.
    #
    # 2026-08: scaled by the model's own MEASURED trust in this stat (attach_stat_trust →
    # stat_trust_gamma, from db.stat_gammas — the edge-regression slope of actual-vs-line on
    # model-vs-line, per stat, from the graded ledger). Before this, a stat the model has
    # historically had ZERO real edge on (gamma≈0 — e.g. MLB Runs) could score exactly as
    # high on the EV component as a stat it's proven itself on (gamma high — e.g. Hits),
    # purely because Juice Score never consulted the model's own track record. gamma=0.5
    # (the neutral default for a stat with too little graded history to have a measured
    # value yet) leaves the EV component untouched — this only discounts a stat once it's
    # been PROVEN untrustworthy, never merely because it hasn't been measured yet.
    rec = recommend_side(model_prob, line) if model_prob is not None else None
    ev = rec["ev"] if rec else None
    ev_score = _scale(ev, -0.08, 0.15)
    trust_mult = min(1.0, 2.0 * float(line.get("stat_trust_gamma", 0.5)))
    out["ev"] = (ev_score * trust_mult,
                f"{ev*100:+.1f}% EV" + (f" (×{trust_mult:.2f} stat trust)" if trust_mult < 0.99 else "")
                if ev is not None else "not priced")

    # 2. Confidence — reuse the existing, real confidence score.
    conf = confidence_score(line) / 100.0
    out["confidence"] = (conf, f"{int(round(conf*100))}/100 confidence")

    # 3. Distribution stability — coefficient of variation from the shipped p10-p90 band.
    # A tight distribution relative to its own projection is a more predictable outcome
    # (Plate Appearances) than a wide one (Home Runs); volatility is the same signal
    # (a volatile market IS one with an unstable projection), so they're merged into one
    # honestly-computed component rather than inventing a second, redundant number.
    sd, proj = _std_from_band(line), line.get("model_proj")
    cv = (sd / max(abs(float(proj)), 0.5)) if (sd is not None and proj is not None) else None
    out["stability"] = (_scale(cv, 2.0, 0.3), f"CV {cv:.2f}" if cv is not None else "n/a")

    # 4. Model agreement — tennis's real 3-way estimate if present, else the MLB/WNBA
    # raw-vs-blended-projection proxy (see module docstring above for why these differ).
    ma = line.get("model_agreement")
    if ma is not None:
        out["agreement"] = (_clamp(float(ma)), f"{int(round(float(ma)*100))}/100 (3-model)")
    else:
        raw = line.get("model_raw")
        gap = (abs(float(raw) - float(proj)) / max(abs(float(proj)), 0.5)
               if (raw is not None and proj is not None) else None)
        out["agreement"] = (_scale(gap, 0.6, 0.0),
                            f"{gap*100:.0f}% raw-vs-market gap" if gap is not None else "n/a")

    # 5. Market quality — how many distinct books currently list this player+stat
    # (analytics.attach_market_quality). A single-book prop can't be cross-validated.
    n_books = line.get("market_book_count")
    mq = _clamp(0.5 + 0.25 * (min(n_books, 3) - 1)) if n_books else 0.5
    out["market_quality"] = (mq, f"{n_books or 1} book{'s' if (n_books or 1) != 1 else ''}")

    # 6. Projection diff — the edge's size relative to the line itself (a +1.6 edge on an
    # 18.5 line means more than the same +1.6 on a 3.5 line). The DOMINANT component as of
    # 2026-08-05 (see _JUICE_WEIGHTS) — formerly "line_value" at a token 5%.
    #
    # 2026-08-17: discounted by how decisive model_prob actually is (same distance-from-0.5
    # signal Confidence's Decisiveness factor uses). A raw mean-vs-line gap does not always
    # mean a real edge — a wide or skewed distribution (e.g. a discrete counting stat like
    # Hits+Runs+RBIs) can put the MEAN well past the line while P(over) stays near a coin
    # flip, because the mode/median sits on the other side. Before this fix, proj_diff scored
    # that gap at full value regardless of decisiveness, so a coin-flip prop with a big raw
    # gap could out-score a genuinely decisive one — the "higher projection, lower over%"
    # props users flagged as reading too high. Floored at 0.25 (not zeroed) so a real, if
    # wide-distribution, edge still keeps a quarter credit rather than being wiped out.
    edge, ln = line.get("model_edge"), line.get("line")
    lv = (abs(float(edge)) / max(abs(float(ln)), 0.5)) if (edge is not None and ln is not None) else None
    decisiveness = _clamp(2.0 * abs(float(model_prob) - 0.5))
    corroboration = 0.25 + 0.75 * decisiveness
    proj_diff_raw = _scale(lv, 0.0, 0.5)
    out["proj_diff"] = (proj_diff_raw * corroboration,
                        (f"{lv*100:.0f}% of line" + (f" (×{corroboration:.2f} decisiveness)" if corroboration < 0.99 else ""))
                        if lv is not None else "n/a")

    # 7. Data quality — sample size + lineup certainty (reuses model_n/lineup_status, but as
    # a DATA-COMPLETENESS signal, not a trust-in-the-math signal like Confidence's use of n).
    n = line.get("model_n") or 0
    lineup_ok = 0.5 if line.get("lineup_status") == "questionable" else 1.0
    dq = _clamp(0.6 * (n / 20.0) + 0.4 * lineup_ok)
    out["data_quality"] = (dq, f"{n} games, {'questionable' if lineup_ok < 1 else 'confirmed'}")

    return out


def model_agreement_score(line: dict[str, Any]) -> Optional[float]:
    """Public accessor for the same 0..1 'does the model agree with itself/the market' signal
    used inside juice_score's agreement component (tennis's real 3-way agreement, else the
    raw-vs-blended-projection proxy — see _juice_components). Exposed standalone for callers
    (e.g. the Entry Optimizer) that want the raw number without recomputing the full Juice
    Score breakdown. None when there's no probability to score against."""
    prob = line.get("model_prob")
    if prob is None:
        return None
    return round(_juice_components(line, prob)["agreement"][0], 3)


def juice_score(line: dict[str, Any], model_prob: Optional[float] = None) -> int:
    """
    0–100 composite ranking how attractive a prop is OVERALL — not a re-skin of Confidence
    or EV, but a blend led by how far the projection sits from the line (proj_diff, the
    dominant factor as of 2026-08-05) plus EV, confidence, distribution stability,
    model-vs-market agreement, cross-book market quality, and data completeness (see
    _JUICE_WEIGHTS and _juice_components). Deliberately selective: a coin-flip, single-book,
    thin-sample prop should NOT land in the same range as a stable, well-agreed,
    cross-validated one with a real edge.
    """
    prob = model_prob if model_prob is not None else line.get("model_prob")
    if prob is None:
        return 0
    components = _juice_components(line, prob)
    composite = sum(_JUICE_WEIGHTS[k] * components[k][0] for k in _JUICE_WEIGHTS)
    return int(round(_clamp(composite) * 100))


def juice_factors(line: dict[str, Any], model_prob: Optional[float] = None) -> list[dict[str, Any]]:
    """The REAL decomposition of `juice_score` — weighted contribution + a human-readable
    detail for every component, so the UI can show exactly what makes a prop juicy instead
    of a single opaque number."""
    prob = model_prob if model_prob is not None else line.get("model_prob")
    if prob is None:
        return []
    components = _juice_components(line, prob)
    labels = {"proj_diff": "Proj vs Line", "ev": "Expected Value", "confidence": "Confidence",
              "stability": "Stability", "agreement": "Model Agreement",
              "market_quality": "Market Quality", "data_quality": "Data Quality"}
    return [
        {"factor": labels[k], "value": round(_JUICE_WEIGHTS[k] * components[k][0] * 100, 1),
         "max": round(_JUICE_WEIGHTS[k] * 100, 1), "detail": components[k][1]}
        for k in _JUICE_WEIGHTS
    ]


def _std_from_band(line: dict[str, Any]) -> Optional[float]:
    """Approximate SD from the shipped p10–p90 band (≈ 2.5631 sigma wide). Labeled approximate
    because the underlying distribution is discrete/skewed — this is for display, not sampling."""
    lo, hi = line.get("model_floor"), line.get("model_ceiling")
    if lo is None or hi is None or hi <= lo:
        return None
    return round((float(hi) - float(lo)) / 2.5631, 3)


def volatility_cv(line: dict[str, Any]) -> Optional[float]:
    """Coefficient of variation (SD / |projection|) from the shipped p10-p90 band — the same
    real signal _juice_components' stability factor uses, exposed publicly so callers outside
    this module (e.g. dashboard.py's play cards) can show a real "Volatility" number instead
    of re-deriving it. None when there's no band or no projection."""
    sd, proj = _std_from_band(line), line.get("model_proj")
    if sd is None or proj is None or abs(float(proj)) < 1e-9:
        return None
    return round(sd / max(abs(float(proj)), 0.5), 3)


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
