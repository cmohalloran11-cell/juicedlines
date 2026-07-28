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


def test_kelly_formula_and_cap():
    line = {"over_implied": 0.5, "under_implied": 0.5}   # decimal odds 2.0
    # f = (p - q)/(1 - q) = (0.6 - 0.5)/0.5 = 0.2
    assert valuation.kelly_fraction(0.6, line) == 0.2
    # huge edge is capped at quarter-Kelly (0.25)
    assert valuation.kelly_fraction(0.95, line) == 0.25
    # no edge → 0
    assert valuation.kelly_fraction(0.5, line) == 0.0


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
