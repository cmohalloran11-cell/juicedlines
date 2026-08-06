"""
Tests for fantasy.scoring — the pure scoring engine. No hardcoded PPR/half-PPR anywhere: every
case here proves the point total changes when the passed-in scoring_settings dict changes,
never from an assumption baked into the function.
"""
from __future__ import annotations

from fantasy import scoring


def test_score_stats_applies_league_settings_exactly():
    stats = {"rec": 5, "rec_yd": 60, "rec_td": 1, "rush_yd": 10}
    ppr = {"rec": 1.0, "rec_yd": 0.1, "rec_td": 6.0, "rush_yd": 0.1}
    assert scoring.score_stats(stats, ppr) == 5 + 6.0 + 6.0 + 1.0  # 5*1 + 60*0.1 + 1*6 + 10*0.1 = 18.0


def test_score_stats_zero_ppr_ignores_receptions_value():
    stats = {"rec": 5, "rec_yd": 60}
    standard = {"rec": 0.0, "rec_yd": 0.1}  # standard (non-PPR) league: receptions worth 0
    half_ppr = {"rec": 0.5, "rec_yd": 0.1}
    assert scoring.score_stats(stats, standard) == 6.0
    assert scoring.score_stats(stats, half_ppr) == 8.5


def test_score_stats_ignores_unscored_keys():
    stats = {"pass_yd": 300, "pass_cmp": 25}
    settings = {"pass_yd": 0.04}  # league doesn't score completions at all
    assert scoring.score_stats(stats, settings) == 12.0


def test_score_stats_missing_stat_key_treated_as_zero():
    stats = {"rush_yd": 50}
    settings = {"rush_yd": 0.1, "rush_td": 6.0}
    assert scoring.score_stats(stats, settings) == 5.0


def test_score_players_attaches_points_without_mutating_input():
    projections = [{"player_id": "p1", "stats": {"rec": 4, "rec_yd": 40}}]
    settings = {"rec": 1.0, "rec_yd": 0.1}
    out = scoring.score_players(projections, settings)
    assert out[0]["fantasy_points"] == 8.0
    assert "fantasy_points" not in projections[0]  # original untouched
