"""
Tests for the SimulationOS valuation math and DataOS quality scoring.

Valuation is pure math on a line dict, so these assert the exact algebra (EV, Kelly) and the
monotonic behavior of the heuristic scores (confidence, juice). DataOS is scored from a fake
snapshot with a known shape.
"""
from __future__ import annotations

import valuation
import dataos


# ── valuation: exact algebra ──────────────────────────────────────────────────

def test_expected_value_matches_formula():
    # model 60% over, book implies 50% (even money) → EV = 0.6/0.5 - 1 = 0.20
    line = {"over_implied": 0.5, "under_implied": 0.5}
    assert valuation.expected_value(0.6, line) == 0.2
    # negative edge: model 40% over → favored side is UNDER at 60% vs implied 0.5 → +0.20
    assert valuation.expected_value(0.4, line) == 0.2


def test_expected_value_negative_when_model_trails_price():
    line = {"over_implied": 0.7, "under_implied": 0.7}   # juiced both ways
    # model 60% over vs 70% implied → 0.6/0.7 - 1 ≈ -0.143
    assert valuation.expected_value(0.6, line) < 0


def test_expected_value_uses_pickem_price_not_even_money():
    # Regression: PrizePicks standard legs have NO over_implied/under_implied (the feed
    # doesn't expose per-leg odds), so EV silently fell back to a 0%-vig 50/50 assumption
    # instead of the real ~57.7%-implied 2-pick pick'em price (-137 American) the ingestion
    # layer already computes and attaches as pickem_price. Every PrizePicks EV was too high.
    line_with_pickem = {"pickem_price": -137}
    ev = valuation.expected_value(0.70, line_with_pickem)
    # decimal odds for -137 = 1 + 100/137 ≈ 1.7299 → EV = 0.70*1.7299-1 ≈ 0.2109
    assert 0.20 < ev < 0.22
    # a line with real per-side odds must still win over pickem_price (never used as fallback
    # when we actually know the price)
    line_with_real_odds = {"over_implied": 0.5, "pickem_price": -137}
    assert valuation.expected_value(0.70, line_with_real_odds) == 0.4
    # neither a real per-side price nor a pick'em price → the book doesn't offer either side
    # priced, so there's nothing to recommend an EV for (no fabricated even-money guess —
    # this is the market-availability fix: never invent a price for an unpriced/unoffered side).
    assert valuation.expected_value(0.70, {}) is None


def test_demon_goblin_are_unpriced_not_fabricated():
    # PrizePicks doesn't expose the boosted payout multiplier for demon/goblin legs, so EV/Kelly
    # must honestly return None rather than fall back to even-money — even though the line has
    # a real projection and probability (they're 96% projected, just not priced).
    demon = {"model_prob": 0.9, "over_implied": 0.5, "pickem_price": -137, "odds_type": "demon"}
    goblin = {"model_prob": 0.9, "odds_type": "GOBLIN"}   # case-insensitive
    standard = {"model_prob": 0.9, "over_implied": 0.5}
    assert valuation.is_unpriced(demon) is True
    assert valuation.is_unpriced(goblin) is True
    assert valuation.is_unpriced(standard) is False
    assert valuation.expected_value(0.9, demon) is None
    assert valuation.expected_value(0.9, goblin) is None
    assert valuation.kelly_fraction(0.9, demon) is None
    assert valuation.expected_value(0.9, standard) is not None


def test_recommend_side_picks_higher_ev_not_higher_probability():
    # Multiplier-book example from the audit: 30%-likely Over at a 4.0x payout beats a
    # 70%-likely Under at a thin 1.1x payout — EV(over)=0.3*4-1=+0.20, EV(under)=0.7*1.1-1=-0.23.
    # Recommending on raw probability alone would wrongly pick Under here.
    line = {"over_implied": 1.0 / 4.0, "under_implied": 1.0 / 1.1}
    rec = valuation.recommend_side(0.30, line)
    assert rec["side"] == "over"
    assert rec["ev"] == 0.2
    assert valuation.expected_value(0.30, line) == 0.2


def test_recommend_side_never_recommends_unavailable_side():
    # Underdog/Sleeper single-sided market (e.g. an Over-only Home Run prop) — only
    # over_implied is populated. Even when the model favors Under, Under isn't a real bet
    # here, so it must never be recommended.
    line = {"over_implied": 0.4}   # no under_implied, no pickem_price
    rec = valuation.recommend_side(0.2, line)   # model thinks Under is likelier
    assert rec is not None and rec["side"] == "over"


def test_recommend_side_none_when_neither_side_priced():
    # The book offers neither side priced for this leg — don't recommend an unplaceable bet.
    assert valuation.recommend_side(0.6, {}) is None
    assert valuation.valuation({"model_prob": 0.6, "model_proj": 2.0, "line": 1.5})["available"] is False


def test_demon_goblin_always_recommend_over_never_under():
    # PrizePicks doesn't let you take Under on a boosted leg — even when the model's own
    # probability favors Under, the recommended side must still be Over.
    demon_model_dislikes_it = {"model_prob": 0.1, "odds_type": "demon"}
    rec = valuation.recommend_side(0.1, demon_model_dislikes_it)
    assert rec["side"] == "over" and rec["ev"] is None


def test_audit_ev_flags_above_threshold_not_below():
    # 95% model prob at the flat pick'em price (~57.7% implied) → EV ≈ +64%, well above
    # any sane threshold — must be flagged with the exact inputs a reviewer needs.
    hot = {"model_prob": 0.95, "pickem_price": -137, "player": "X", "stat_type": "Hits",
           "line": 0.5, "model_proj": 1.2, "source": "prizepicks", "id": "abc"}
    f = valuation.audit_ev(hot, threshold=0.15)
    assert f is not None and f["ev"] > 0.15 and f["used_pickem_fallback"] is True
    assert f["player"] == "X" and f["side"] == "over"
    # a modest, realistic edge must NOT be flagged
    mild = {"model_prob": 0.58, "over_implied": 0.524, "under_implied": 0.524}
    assert valuation.audit_ev(mild, threshold=0.15) is None
    # no probability at all → nothing to audit, not an error
    assert valuation.audit_ev({}, threshold=0.15) is None


def test_kelly_formula_and_cap():
    line = {"over_implied": 0.5, "under_implied": 0.5}   # decimal odds 2.0
    # f = (p - q)/(1 - q) = (0.6 - 0.5)/0.5 = 0.2
    assert valuation.kelly_fraction(0.6, line) == 0.2
    # huge edge is capped at quarter-Kelly (0.25)
    assert valuation.kelly_fraction(0.95, line) == 0.25
    # no edge → 0
    assert valuation.kelly_fraction(0.5, line) == 0.0


def test_confidence_factors_decompose_the_score():
    # The three real components must sum to the confidence score (the honest breakdown).
    line = {"model_prob": 0.72, "model_n": 18, "proj_kind": "engine"}
    factors = valuation.confidence_factors(line)
    assert [f["factor"] for f in factors] == ["Sample Size", "Decisiveness", "Method"]
    total = sum(f["value"] for f in factors)
    assert round(total) == valuation.confidence_score(line)
    # caps are the real weights (50/30/20) and no component exceeds its cap
    assert [f["max"] for f in factors] == [50, 30, 20]
    assert all(0 <= f["value"] <= f["max"] for f in factors)


def test_confidence_monotonic_in_sample_and_decisiveness():
    base = {"model_prob": 0.55, "model_n": 5, "proj_kind": "model"}
    more_games = {**base, "model_n": 40}
    more_decisive = {**base, "model_prob": 0.75}
    engine = {**base, "proj_kind": "engine"}
    assert valuation.confidence_score(more_games) > valuation.confidence_score(base)
    assert valuation.confidence_score(more_decisive) > valuation.confidence_score(base)
    assert valuation.confidence_score(engine) > valuation.confidence_score(base)
    assert 0 <= valuation.confidence_score(base) <= 100


def test_simulation_object_and_valuation_bundle():
    line = {"id": "L1", "line": 1.5, "model_proj": 1.9, "model_edge": 0.4,
            "model_prob": 0.62, "model_floor": 0.5, "model_ceiling": 3.5,
            "model_n": 25, "proj_kind": "engine", "over_implied": 0.5, "under_implied": 0.5}
    sim = valuation.simulation_object(line)
    assert sim["overProbability"] == 0.62 and sim["underProbability"] == 0.38
    assert sim["sampleSize"] == 25 and sim["standardDeviation"] is not None
    val = valuation.valuation(line)
    assert val["available"] and val["side"] == "over"
    assert val["expectedValue"] == 0.24 and 0 < val["juiceScore"] <= 100


def test_valuation_unavailable_without_projection():
    assert valuation.valuation({"id": "x", "line": 1.5})["available"] is False


# ── DataOS ────────────────────────────────────────────────────────────────────

def test_data_health_scores_a_fresh_snapshot_high():
    from provenance import now_iso
    lines = [
        {"source": "prizepicks", "sport": "MLB", "model_proj": 1.7},
        {"source": "underdog", "sport": "WNBA", "model_proj": 18.0},
        {"source": "prizepicks", "sport": "MLB"},  # no projection
    ]
    h = dataos.health(lines, now_iso(), errors={})
    assert h["total_lines"] == 3 and h["projected_lines"] == 2
    assert h["live_sources"] == 2
    assert h["quality_score"] > 60          # fresh + 2 sources + 2/3 coverage
    assert h["projection_coverage"] == round(2 / 3, 3)


def test_data_health_penalizes_staleness_and_errors():
    old = "2020-01-01T00:00:00+00:00"
    h = dataos.health([{"source": "prizepicks", "sport": "MLB", "model_proj": 1.0}],
                      old, errors={"underdog": "down", "prizepicks": "429"})
    assert h["freshness"] == 0.0
    assert h["quality_score"] < 50
    assert h["source_errors"]


# ── direction-invariant validation (2026-07-29 Over/Under bias audit) ─────────

def test_validate_direction_flags_a_genuine_violation():
    ok = {"id": "ok", "model_proj": 2.0, "line": 1.5, "model_prob": 0.6}       # proj>line, prob>0.5: fine
    bad = {"id": "bad", "model_proj": 2.0, "line": 1.5, "model_prob": 0.4}     # proj>line, prob<0.5: violation
    also_bad = {"id": "bad2", "model_proj": 1.0, "line": 1.5, "model_prob": 0.6}  # proj<line, prob>0.5
    no_proj = {"id": "np", "line": 1.5, "model_prob": 0.6}                    # not enough data — skipped
    violations = dataos.validate_direction([ok, bad, also_bad, no_proj])
    assert {v["id"] for v in violations} == {"bad", "bad2"}


def test_validate_direction_reject_nulls_the_offending_line_in_place():
    bad = {"id": "bad", "model_proj": 2.0, "line": 1.5, "model_prob": 0.4, "model_edge": 0.5}
    dataos.validate_direction([bad], reject=True)
    assert bad["model_proj"] is None and bad["model_prob"] is None and bad["model_edge"] is None
    assert bad["direction_rejected"] is True


def test_direction_report_distribution_and_rejection():
    lines = [
        {"id": "o1", "model_proj": 2.0, "line": 1.5, "model_prob": 0.6},   # over, valid
        {"id": "u1", "model_proj": 1.0, "line": 1.5, "model_prob": 0.4},   # under, valid
        {"id": "bad", "model_proj": 2.0, "line": 1.5, "model_prob": 0.4},  # violation
    ]
    report = dataos.direction_report(lines, reject=True)
    assert report["checked"] == 3
    assert report["violations"] == 1
    # distribution reflects the board BEFORE rejection (2 over/under among the 2 valid rows —
    # the violator counted as its raw model_prob<0.5, i.e. "under", before being rejected)
    assert report["distribution"]["over"] == 1
    assert report["distribution"]["under"] == 2
    # and the violating line was actually rejected (nulled), not just logged
    assert lines[2]["model_proj"] is None
