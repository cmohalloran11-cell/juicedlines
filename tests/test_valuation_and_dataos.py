"""
Tests for the SimulationOS valuation math and DataOS quality scoring.

Valuation is pure math on a line dict, so these assert the exact algebra (EV, Kelly) and the
monotonic behavior of the heuristic scores (confidence, juice). DataOS is scored from a fake
snapshot with a known shape.
"""
from __future__ import annotations

import pytest

import valuation
import dataos


# ── valuation: exact algebra ──────────────────────────────────────────────────

def test_expected_value_matches_formula():
    # model 53% over, book implies 50% (even money) → raw EV = 0.53*2-1 = 0.06 — inside the
    # ±8% zone _dampen_ev leaves untouched, so this exercises the exact underlying formula.
    line = {"over_implied": 0.5, "under_implied": 0.5}
    assert valuation.expected_value(0.53, line) == pytest.approx(0.06, abs=1e-9)
    # negative edge: model 47% over → favored side is UNDER at 53% vs implied 0.5 → same +0.06
    assert valuation.expected_value(0.47, line) == pytest.approx(0.06, abs=1e-9)


def test_ev_dampening_compresses_large_estimates_for_display():
    # 2026-07-30: live examples showed props reading +50-77% EV — real edges rarely exceed
    # ~10% (the product's own tier guide calls >15% "extremely rare"), so a raw estimate that
    # large is presumed to be at least partly a calibration artifact, not a real edge, and is
    # compressed for display. Order-preserving (never flips which side wins), untouched below
    # the 8% threshold, only the excess above it is compressed.
    assert valuation._dampen_ev(0.05) == 0.05                 # small, real edge: untouched
    assert valuation._dampen_ev(0.08) == 0.08                 # exactly at the threshold: untouched
    assert valuation._dampen_ev(0.771) == pytest.approx(0.18365, abs=1e-6)  # live example
    assert valuation._dampen_ev(-0.42) == pytest.approx(-0.131, abs=1e-6)   # sign preserved
    # monotonic: a bigger raw EV never dampens to a smaller shown EV
    assert valuation._dampen_ev(0.5) < valuation._dampen_ev(0.7) < valuation._dampen_ev(0.9)


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
    # decimal odds for -137 = 1 + 100/137 ≈ 1.7299 → raw EV = 0.70*1.7299-1 ≈ 0.2109 →
    # dampened (_dampen_ev) to 0.08 + (0.2109-0.08)*0.15 ≈ 0.0996
    assert 0.09 < ev < 0.11
    # a line with real per-side odds must still win over pickem_price (never used as fallback
    # when we actually know the price) — raw EV 0.4 dampens to 0.08+(0.4-0.08)*0.15=0.128
    line_with_real_odds = {"over_implied": 0.5, "pickem_price": -137}
    assert valuation.expected_value(0.70, line_with_real_odds) == pytest.approx(0.128, abs=1e-9)
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
    # 70%-likely Under at a thin 1.1x payout — raw EV(over)=0.3*4-1=+0.20 (dampens to 0.098),
    # raw EV(under)=0.7*1.1-1=-0.23 (dampens to -0.1025). Recommending on raw probability
    # alone would wrongly pick Under here; the DAMPENING must never flip which side wins.
    line = {"over_implied": 1.0 / 4.0, "under_implied": 1.0 / 1.1}
    rec = valuation.recommend_side(0.30, line)
    assert rec["side"] == "over"
    assert rec["ev"] == pytest.approx(0.098, abs=1e-9)
    assert valuation.expected_value(0.30, line) == pytest.approx(0.098, abs=1e-9)


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


def test_confidence_has_no_guaranteed_floor_for_a_coin_flip():
    # 2026-07-29 projection-realism pass: a well-sampled (30+ games), engine-projected prop
    # that's a genuine coin-flip (prob=0.5) must NOT score in the 70s just from sample size +
    # method alone — that was the actual bug behind "everything shows 78-86% confidence".
    coin_flip = {"model_prob": 0.5, "model_n": 40, "proj_kind": "engine"}
    assert valuation.confidence_score(coin_flip) <= 55
    # a genuinely decisive, well-sampled engine prop should still score high.
    decisive = {"model_prob": 0.95, "model_n": 40, "proj_kind": "engine"}
    assert valuation.confidence_score(decisive) >= 90


def test_juice_score_is_not_a_reskin_of_confidence_or_ev():
    # 2026-07-30 rebuild: Juice Score must answer a genuinely different question than
    # Confidence ("how much do we trust the projection") or EV ("how much value does the
    # market offer") — not just track whichever of those is highest. Two props with
    # IDENTICAL confidence and EV can still get different Juice Scores because of stability/
    # agreement/market-quality/data-quality — prove that here.
    base = {"model_prob": 0.7, "model_n": 20, "proj_kind": "engine", "model_proj": 2.0,
            "line": 1.5, "model_edge": 0.5, "over_implied": 0.55, "under_implied": 0.55}
    thin_data = {**base, "market_book_count": 1, "lineup_status": "questionable"}
    well_covered = {**base, "market_book_count": 3, "lineup_status": None,
                    "model_floor": 1.5, "model_ceiling": 2.5}  # tighter, more stable band
    assert valuation.confidence_score(thin_data) == valuation.confidence_score(well_covered)
    assert valuation.juice_score(thin_data) < valuation.juice_score(well_covered)


def test_juice_score_is_selective_not_inflated():
    # The product's own tier guide wants juice scores to be selective — a genuine coin-flip
    # prop with no real edge, single-book, thin sample must NOT land anywhere near the top.
    weak = {"model_prob": 0.51, "model_n": 3, "proj_kind": "model", "model_proj": 1.51,
            "line": 1.5, "model_edge": 0.01, "market_book_count": 1}
    assert valuation.juice_score(weak) < 40
    # a genuinely strong prop — decisive, confident, stable, cross-booked, real edge — should
    # score meaningfully higher, but the components are real math, not a rubber stamp to 100.
    strong = {"model_prob": 0.85, "model_n": 30, "proj_kind": "engine", "model_proj": 3.5,
              "line": 1.5, "model_edge": 2.0, "model_floor": 3.0, "model_ceiling": 4.0,
              "over_implied": 0.55, "under_implied": 0.55, "market_book_count": 3,
              "lineup_status": None}
    assert valuation.juice_score(strong) > valuation.juice_score(weak)
    assert valuation.juice_score(strong) >= 60


def test_juice_factors_sum_to_juice_score():
    line = {"model_prob": 0.72, "model_n": 18, "proj_kind": "engine", "model_proj": 2.2,
            "line": 1.5, "model_edge": 0.7, "over_implied": 0.5, "under_implied": 0.5,
            "market_book_count": 2}
    factors = valuation.juice_factors(line)
    assert len(factors) == 7   # ev, confidence, stability, agreement, market_quality, line_value, data_quality
    assert round(sum(f["value"] for f in factors)) == valuation.juice_score(line)
    assert round(sum(f["max"] for f in factors)) == 100
    assert all(0 <= f["value"] <= f["max"] for f in factors)


def test_juice_score_missing_signals_stay_neutral_not_penalized():
    # A prop missing model_raw/model_agreement/market_book_count (e.g. a sport or path that
    # doesn't compute them yet) must fall back to a neutral 0.5 for that component, not 0 —
    # an unknown signal shouldn't be punished as if it were confirmed bad.
    minimal = {"model_prob": 0.6, "model_n": 10, "proj_kind": "engine"}
    assert valuation.juice_score(minimal) > 0


def test_confidence_factors_decompose_the_score():
    # The three real components must sum to the confidence score (the honest breakdown).
    line = {"model_prob": 0.72, "model_n": 18, "proj_kind": "engine"}
    factors = valuation.confidence_factors(line)
    assert [f["factor"] for f in factors] == ["Sample Size", "Decisiveness", "Method"]
    total = sum(f["value"] for f in factors)
    assert round(total) == valuation.confidence_score(line)
    # caps are the real weights (30/50/20 — decisiveness re-weighted heaviest 2026-07-29
    # so a well-sampled engine prop that's a real coin-flip doesn't get a guaranteed floor)
    assert [f["max"] for f in factors] == [30, 50, 20]
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


def test_confidence_full_engine_bonus_applies_to_wnba_and_tennis_too():
    # 2026-07-30: WNBA's per-possession sim (proj_kind="basketball") and tennis's serve/return
    # sim (proj_kind="tennis") are genuine full Monte Carlo runs, same as MLB's "engine" — but
    # confidence_score's method bonus only ever recognized "engine", silently docking both
    # sports 10 points their entire time on the board. Must score identically to "engine" now.
    base = {"model_prob": 0.7, "model_n": 20, "proj_kind": "model"}
    assert (valuation.confidence_score({**base, "proj_kind": "engine"})
            == valuation.confidence_score({**base, "proj_kind": "basketball"})
            == valuation.confidence_score({**base, "proj_kind": "tennis"}))
    # tennis's OWN fallback ("market" — fully deferred to the market line, no real model call)
    # correctly stays at the lower bonus, same as MLB's "model"/empirical-average fallback.
    assert (valuation.confidence_score({**base, "proj_kind": "market"})
            == valuation.confidence_score({**base, "proj_kind": "model"})
            < valuation.confidence_score({**base, "proj_kind": "engine"}))


def test_simulation_object_and_valuation_bundle():
    line = {"id": "L1", "line": 1.5, "model_proj": 1.9, "model_edge": 0.4,
            "model_prob": 0.62, "model_floor": 0.5, "model_ceiling": 3.5,
            "model_n": 25, "proj_kind": "engine", "over_implied": 0.5, "under_implied": 0.5}
    sim = valuation.simulation_object(line)
    assert sim["overProbability"] == 0.62 and sim["underProbability"] == 0.38
    assert sim["sampleSize"] == 25 and sim["standardDeviation"] is not None
    val = valuation.valuation(line)
    assert val["available"] and val["side"] == "over"
    # raw EV 0.62*2-1=0.24 dampens (_dampen_ev) to 0.08+(0.24-0.08)*0.15=0.104
    assert val["expectedValue"] == pytest.approx(0.104, abs=1e-9) and 0 < val["juiceScore"] <= 100


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


def test_validate_direction_checks_median_not_the_mean_projection():
    # 2026-07-29 projection-realism pass: `model_proj` is the MEAN and is ALLOWED to diverge
    # from the probability's direction on a skewed stat (a bench hitter's TB mean can be 0.6
    # while the median — and P(over 0.5)'s real direction — is 0). This must NOT be flagged:
    # the median (0.0) correctly agrees with prob_over (0.35 < 0.5).
    skewed_but_fine = {"id": "skew", "model_proj": 0.6, "model_median": 0.0,
                       "line": 0.5, "model_prob": 0.35}
    # A genuine violation must still be caught when the MEDIAN itself disagrees with prob.
    genuinely_bad = {"id": "bad", "model_proj": 0.6, "model_median": 0.6,
                     "line": 0.5, "model_prob": 0.35}
    violations = dataos.validate_direction([skewed_but_fine, genuinely_bad])
    assert {v["id"] for v in violations} == {"bad"}


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
