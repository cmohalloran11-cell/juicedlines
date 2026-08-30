"""
Tests for the rebuilt (v2) Juice Score — the SIGNED [-100, +100] score in valuation.juice_v2.

Every input it reads is PRE-anchor (model_pre_*), so these fixtures set those fields directly
rather than model_proj/model_prob: scoring the anchored projection is the exact bug the
rebuild exists to fix (proj - line is shrunk by t and is identically zero at t=0).

Validation of the score against real graded outcomes lives in reports/02-juice.md — these
assert the mechanism (sign, scaling, null paths, coherence gate, near-lock cap), not the
calibration.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import analytics
import valuation


def _line(**over):
    """A right-skewed counting-stat prop on a 1.5 line: median 2.0, mean 2.4 (g = +0.36 SD of
    skew), SD 1.1, P(over) 0.62 — the model leans over and its median agrees."""
    base = {
        "id": "x1", "sport": "MLB", "player": "Test Hitter", "stat_type": "Total Bases",
        "line": 1.5, "odds_type": "standard",
        "model_pre_prob": 0.62, "model_pre_mean": 2.4, "model_pre_median": 2.0,
        "model_pre_sd": 1.1, "model_anchor_t": 1.0,
        "model_n": 30, "stat_trust_gamma": 0.5, "lineup_status": "in",
    }
    base.update(over)
    return base


# ── signs agree: the score scales with the edge and with confidence ───────────────

def test_sign_follows_the_probability_edge():
    over = valuation.juice_v2(_line(model_pre_prob=0.70))
    under = valuation.juice_v2(_line(model_pre_prob=0.30, model_pre_median=1.0,
                                     model_pre_mean=1.3))
    assert over["juice"] > 0 and over["side"] == "over"
    assert under["juice"] < 0 and under["side"] == "under"
    # symmetric edges around break-even produce symmetric magnitudes
    assert over["juice"] == pytest.approx(-under["juice"], abs=1e-6)


def test_magnitude_is_monotone_in_the_probability_edge():
    scores = [valuation.juice_v2(_line(model_pre_prob=p))["juice"]
              for p in (0.52, 0.60, 0.70, 0.85)]
    assert scores == sorted(scores)
    assert all(0.0 < s <= 100.0 for s in scores)


def test_confidence_scales_the_magnitude_without_moving_the_sign():
    full = valuation.juice_v2(_line(model_pre_prob=0.75))
    thin = valuation.juice_v2(_line(model_pre_prob=0.75, model_n=3))
    unreliable = valuation.juice_v2(_line(model_pre_prob=0.75, stat_trust_gamma=0.05))
    assert full["c"] == 1.0
    assert 0.0 < thin["juice"] < full["juice"]
    assert 0.0 < unreliable["juice"] < full["juice"]


def test_a_stat_with_no_measured_history_is_not_punished_for_being_unmeasured():
    # gamma 0.5 is attach_stat_trust's neutral default for "too little graded history to have
    # a measured value yet" — it must map to a reliability factor of exactly 1.0, the same as
    # a stat proven trustworthy, so an unmeasured stat is never treated as a proven-bad one.
    neutral = valuation.juice_v2(_line(stat_trust_gamma=0.5))
    proven = valuation.juice_v2(_line(stat_trust_gamma=1.0))
    assert neutral["juice"] == proven["juice"]


def test_break_even_is_devigged_from_a_real_two_sided_price():
    # 0.55/0.55 is a symmetric 10% vig → fair 0.5; 0.60/0.50 leans the book toward the over.
    assert valuation.breakeven_prob({"over_implied": 0.55, "under_implied": 0.55}) == pytest.approx(0.5)
    assert valuation.breakeven_prob({"over_implied": 0.60, "under_implied": 0.50}) == pytest.approx(0.6 / 1.1)
    # pick'em pays the same either way, so its break-even is 0.5 whatever the flat payout is
    assert valuation.breakeven_prob({"pickem_price": -137}) == 0.5
    # demon/goblin: the boosted multiplier isn't in the feed, so there is no real break-even
    assert valuation.breakeven_prob({"odds_type": "demon"}) is None


def test_a_devigged_price_moves_the_edge_not_just_the_probability():
    fair = valuation.juice_v2(_line(model_pre_prob=0.58, over_implied=0.55, under_implied=0.55))
    juiced = valuation.juice_v2(_line(model_pre_prob=0.58, over_implied=0.66, under_implied=0.44))
    assert fair["b"] == pytest.approx(0.5)
    assert juiced["b"] == pytest.approx(0.6)
    assert fair["juice"] > 0 > juiced["juice"]     # the same 58% is a bet at 50%, not at 60%


# ── coherence: a model contradicting itself is an integrity error, not a low score ──

def test_coherence_fault_when_probability_and_median_disagree_beyond_skew():
    faulted = valuation.juice_v2(_line(
        model_pre_prob=0.46,          # e < 0 (model leans under)
        model_pre_median=2.5,         # z > 0 by ~0.9 SD (median well above the 1.5 line)
        model_pre_mean=2.55,          # g ~ 0.05 SD — nowhere near enough to explain it
    ))
    assert faulted["juice"] is None
    assert faulted["reason"] == "coherence_fault"
    f = faulted["coherence_fault"]
    assert f["kind"] == "probability_median_sign_disagreement"
    # the diagnostics a human needs to debug it must all be on the fault, not just a flag
    assert {"p", "b", "e", "z", "g", "direction", "m_mean", "m_median", "m_sd", "line",
            "player", "stat_type", "sport"} <= set(f)
    assert f["direction"] < 0 < f["z"]


def test_no_fault_when_the_distribution_skew_explains_the_disagreement():
    # The motivating real case: a right-skewed counting stat whose MEAN sits above the line
    # while P(over) < 0.5, because the book prices at the median. z is built on the median and
    # the residual disagreement is smaller than the mean/median gap, so this is correct model
    # behaviour and must score normally — not get flagged as a bug.
    ok = valuation.juice_v2(_line(
        model_pre_prob=0.46,          # e < 0
        model_pre_median=1.55,        # z = +0.045 SD — a hair over the 1.5 line
        model_pre_mean=1.90,          # g = +0.318 SD of right skew, far larger than |z|
    ))
    assert ok["coherence_fault"] is None
    assert ok["juice"] is not None and ok["juice"] < 0
    assert abs(ok["z"]) < abs(ok["g"])


def test_agreeing_signs_never_fault_however_large_the_gap():
    agree = valuation.juice_v2(_line(model_pre_prob=0.80, model_pre_median=6.0,
                                     model_pre_mean=6.1))
    assert agree["coherence_fault"] is None and agree["juice"] > 0


def test_audit_juice_coherence_collects_every_fault_for_the_review_queue():
    clean = _line()
    bad = _line(model_pre_prob=0.46, model_pre_median=2.5, model_pre_mean=2.55, player="Bad")
    faults = valuation.audit_juice_coherence([clean, bad, clean])
    assert len(faults) == 1 and faults[0]["player"] == "Bad"


# ── null paths: no signal is null, never a plausible-looking small number ──────────

def test_null_when_the_anchor_leaves_no_model_signal():
    # Every board hard-defers to the market line below t=0.2 (basketball/tennis/nfl board.py),
    # so there is no model opinion left to score.
    assert valuation.juice_v2(_line(model_anchor_t=0.0))["reason"] == "no_model_signal"
    assert valuation.juice_v2(_line(model_anchor_t=0.19))["juice"] is None
    assert valuation.juice_v2(_line(model_anchor_t=0.2))["juice"] is not None


def test_null_for_a_degenerate_zero_spread_distribution():
    # MLB "Doubles" emitted one constant projection for all 90 graded rows in clv_seed.db —
    # a stub, not a fitted distribution. There is no scale to normalize against.
    assert valuation.juice_v2(_line(model_pre_sd=0.0))["reason"] == "degenerate_distribution"


def test_null_when_the_pre_anchor_distribution_is_missing():
    assert valuation.juice_v2(_line(model_pre_prob=None))["reason"] == "no_pre_anchor_probability"
    assert valuation.juice_v2(_line(model_pre_sd=None))["reason"] == "no_distribution_moments"


def test_null_for_an_unpriced_demon_or_goblin_leg():
    assert valuation.juice_v2(_line(odds_type="goblin"))["reason"] == "unpriced"


# ── availability near lock ────────────────────────────────────────────────────────

def test_unknown_availability_near_lock_caps_the_magnitude_and_flags_stale():
    strong = dict(model_pre_prob=0.90, model_pre_median=3.0, model_pre_mean=3.2)
    known = valuation.juice_v2(_line(**strong, minutes_to_lock=30.0, lineup_status="in"))
    unknown = valuation.juice_v2(_line(**strong, minutes_to_lock=30.0, lineup_status=None))
    assert known["juice"] > 90 and known["stale"] is False and known["capped"] is False
    assert unknown["stale"] is True and unknown["capped"] is True
    assert unknown["juice"] == pytest.approx(valuation._JUICE_UNKNOWN_AVAILABILITY_CAP)
    # the cap clips, it never inflates: a weak prop inside the horizon keeps its own number
    weak = valuation.juice_v2(_line(model_pre_prob=0.53, minutes_to_lock=30.0,
                                    lineup_status=None))
    assert weak["stale"] is True and weak["capped"] is False and abs(weak["juice"]) < 50


def test_no_cap_outside_the_lock_horizon_or_without_a_clock():
    strong = dict(model_pre_prob=0.90, model_pre_median=3.0, model_pre_mean=3.2)
    far = valuation.juice_v2(_line(**strong, minutes_to_lock=240.0, lineup_status=None))
    no_clock = valuation.juice_v2(_line(**strong, lineup_status=None))
    assert far["stale"] is False and far["juice"] > 90
    assert no_clock["stale"] is False and no_clock["juice"] > 90


def test_attach_lock_clock_reads_the_board_clock_once_so_valuation_stays_pure():
    now = datetime(2026, 8, 28, 23, 0, tzinfo=timezone.utc)
    lines = [
        {"start_time": (now + timedelta(minutes=45)).isoformat().replace("+00:00", "Z")},
        {"start_time": (now - timedelta(minutes=10)).isoformat()},
        {"start_time": None},
        {"start_time": "not a timestamp"},
    ]
    analytics.attach_lock_clock(lines, now=now)
    assert lines[0]["minutes_to_lock"] == pytest.approx(45.0)
    assert lines[1]["minutes_to_lock"] == pytest.approx(-10.0)
    assert "minutes_to_lock" not in lines[2] and "minutes_to_lock" not in lines[3]


# ── version dispatch: v1 stays live until the validation supports flipping ─────────

def test_juice_score_serves_v1_by_default_and_v2_behind_the_flag(monkeypatch):
    line = _line(model_prob=0.62, model_proj=1.3, model_edge=-0.2, model_pre_prob=0.30,
                 model_pre_median=1.0, model_pre_mean=1.3)
    monkeypatch.setattr(valuation, "JUICE_VERSION", "1")
    v1 = valuation.juice_score(line)
    assert isinstance(v1, int) and 0 <= v1 <= 100          # unsigned, never None
    monkeypatch.setattr(valuation, "JUICE_VERSION", "2")
    v2 = valuation.juice_score(line)
    assert v2 < 0                                          # signed: this line leans under
    assert [f["factor"] for f in valuation.juice_factors(line)][:2] == [
        "Probability Edge", "Confidence"]


def test_v2_juice_score_is_none_rather_than_a_low_number_when_there_is_no_signal(monkeypatch):
    monkeypatch.setattr(valuation, "JUICE_VERSION", "2")
    assert valuation.juice_score(_line(model_anchor_t=0.0)) is None
    assert valuation.juice_factors(_line(model_anchor_t=0.0)) == []


def test_v2_factors_show_the_ablated_diagnostics_without_pretending_they_scored():
    factors = valuation.juice_v2_factors(_line())
    scored = [f for f in factors if f["value"] is not None]
    diagnostic = [f for f in factors if f["value"] is None]
    assert [f["factor"] for f in scored] == ["Probability Edge", "Confidence"]
    assert [f["factor"] for f in diagnostic] == ["Projection Differential", "Skew Gap"]
