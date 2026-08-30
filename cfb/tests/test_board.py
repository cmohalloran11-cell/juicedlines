"""
End-to-end tests for attach_cfb -- the entry point analytics.attach_projections calls.

Driven entirely by cfb/tests/fakes.py's synthetic league: no CFBD_API_KEY exists in any
environment this repo has run in, so there is no live response to test against and nothing
here is an accuracy claim. What is asserted is the CONTRACT: which fields land on a line, that
the market anchor is applied on the median (the mean-vs-median bug that had 94% of the live
NFL board recommending Under would reproduce exactly here, since CFB yardage is
Gamma-shaped), that a line the engine cannot price stays unprojected rather than getting a
fabricated one, and that a projected CFB line is juice-scoreable under JUICE_VERSION=2.
"""
from __future__ import annotations

import pytest

import valuation
from cfb import projections as P
from cfb.board import attach_cfb
from cfb.tests import fakes as F


@pytest.fixture(autouse=True)
def _fake_league():
    P.set_source(F.FakeSource())
    P.set_positions_override(F.positions())
    yield
    P.set_source(None)
    P.set_positions_override(None)


def test_attach_cfb_writes_the_shared_valuation_fields_and_the_cfb_contract():
    lines = [F.line()]
    assert attach_cfb(lines) == 1
    l = lines[0]
    for f in ("model_proj", "model_prob", "model_edge", "model_floor", "model_ceiling",
              "model_median", "model_n", "proj_kind", "model_raw", "model_raw_prob",
              "trust_weight", "model_pre_mean", "model_pre_median", "model_pre_sd",
              "model_pre_prob", "model_anchor_t"):
        assert l.get(f) is not None, f
    assert 0.0 <= l["model_prob"] <= 1.0
    assert l["model_floor"] <= l["model_median"] <= l["model_ceiling"]
    assert l["model_edge"] == pytest.approx(l["model_proj"] - l["line"], abs=0.06)
    for f in ("projected_plays", "pace_basis", "usage_share", "blowout_probability",
              "garbage_time_multiplier", "starterness", "opponent", "opponent_factor",
              "opponent_is_bottom_quartile", "prior_tier_reason"):
        assert f in l, f


def test_proj_kind_names_the_prior_tier_and_counts_as_a_full_engine_run():
    lines = [F.line()]
    attach_cfb(lines)
    assert lines[0]["proj_kind"] == "cfb_prior_a"
    assert lines[0]["proj_kind"] in valuation._FULL_ENGINE_KINDS


def test_a_true_freshman_is_priced_from_the_recruiting_tier_and_defers_to_the_market():
    """No college production at all: every rate is the prior, so the model has no own-data
    weight, trust is zero and the board defers fully to the market -- which must read as a
    coinflip, not as a fabricated Under. This is the exact failure mode the mean-vs-median
    anchoring fix exists to prevent (see nfl/board.py's comment): a right-skewed Gamma array
    recentred on its MEAN would leave P(over) well below 0.5 on a line showing no edge."""
    lines = [F.line(player="Power 0 Freshman", position="RB", stat="Rushing Yards", value=54.5)]
    assert attach_cfb(lines) == 1
    l = lines[0]
    assert l["proj_kind"] == "cfb_prior_c"
    assert l["trust_weight"] == 0.0 and l["model_anchor_t"] == 0.0
    assert l["model_median"] == pytest.approx(54.5, abs=0.6)
    assert l["model_prob"] == pytest.approx(0.5, abs=0.03)
    # The displayed mean sits ABOVE the line by the stat's own right skew rather than being
    # forced onto it -- the honest consequence of anchoring the median.
    assert l["model_proj"] > l["model_median"]


def test_a_deeply_sampled_player_keeps_his_own_edge_instead_of_snapping_to_the_line():
    lines = [F.line(player="Power 0 RB1", position="RB", stat="Rushing Yards", value=140.5)]
    attach_cfb(lines)
    assert lines[0]["trust_weight"] > 0.9
    assert lines[0]["model_edge"] < -20.0
    assert lines[0]["model_prob"] < 0.2


def test_every_projected_line_is_juice_scoreable_under_juice_version_2(monkeypatch):
    lines = [F.line()]
    attach_cfb(lines)
    monkeypatch.setattr(valuation, "JUICE_VERSION", "2")
    detail = valuation.juice_v2(lines[0])
    assert detail["reason"] != "no_distribution_moments"
    assert detail["juice"] is not None
    assert detail["coherence_fault"] is None


def test_markets_a_position_cannot_produce_are_left_unprojected():
    lines = [F.line(player="Power 0 WR1", position="WR", stat="Passing Yards", value=210.5)]
    assert attach_cfb(lines) == 0
    assert lines[0].get("model_proj") is None


def test_a_player_the_league_has_never_heard_of_is_left_unprojected():
    lines = [F.line(player="Nobody At All", position="RB", stat="Rushing Yards", value=60.5)]
    assert attach_cfb(lines) == 0
    assert lines[0].get("model_proj") is None


def test_receptions_and_anytime_td_both_price():
    lines = [F.line(player="Power 0 WR1", position="WR", stat="Receptions", value=4.5),
             F.line(player="Power 0 RB1", position="RB", stat="Anytime TD", value=0.5)]
    assert attach_cfb(lines) == 2
    td = lines[1]
    assert 0.0 <= td["model_prob"] <= 1.0
    assert td["model_floor"] == 0.0 and td["model_ceiling"] == 1.0


def test_attach_cfb_ignores_non_cfb_lines():
    lines = [{"id": "mlb_1", "sport": "MLB", "model_proj": 2.0}]
    attach_cfb(lines)
    assert lines[0]["model_proj"] == 2.0


def test_attach_cfb_does_not_raise_on_missing_fields():
    attach_cfb([{"sport": "CFB"}, {}])


def test_no_data_source_leaves_every_cfb_line_unprojected_rather_than_crashing():
    """The no-CFBD_API_KEY production state: every fetch returns [], no league prior can be
    fitted, and the board shows the line with no pick attached -- the same honest degradation
    WNBA has with no BALLDONTLIE_API_KEY."""
    P.set_source(F.FakeSource(seasons=range(0, 0)))
    lines = [F.line()]
    assert attach_cfb(lines) == 0
    assert lines[0].get("model_proj") is None
