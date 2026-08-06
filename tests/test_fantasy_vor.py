"""
Tests for fantasy.vor — replacement levels + VOR + tiering, all computed from a league's
actual roster shape/size (never a fixed assumption).
"""
from __future__ import annotations

from fantasy import vor


def test_starter_counts_splits_flex_across_eligible_positions():
    roster = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN", "BN", "K", "DEF"]
    counts = vor.starter_counts(roster, league_size=10)
    assert counts["QB"] == 10
    assert round(counts["RB"], 3) == round((2 + 1 / 3) * 10, 3)
    assert round(counts["WR"], 3) == round((2 + 1 / 3) * 10, 3)
    assert round(counts["TE"], 3) == round((1 + 1 / 3) * 10, 3)
    assert counts["K"] == 10
    assert counts["DEF"] == 10
    assert "BN" not in counts


def test_starter_counts_zero_for_league_with_no_te_slot():
    roster = ["QB", "RB", "WR", "BN"]
    counts = vor.starter_counts(roster, league_size=8)
    assert "TE" not in counts


def _players(position: str, points: list[float]) -> list[dict]:
    return [{"player_id": f"{position}{i}", "position": position, "fantasy_points": p}
            for i, p in enumerate(points)]


def test_replacement_level_clips_to_available_players():
    # Roster wants 4 RB starters league-wide (2 teams x 2 RB slots), but only 2 RBs exist in
    # the pool -- the replacement level must fall back to the worst available, not error.
    roster = ["RB", "RB", "BN"]
    scored = _players("RB", [15.0, 10.0])
    levels = vor.replacement_levels(scored, roster, league_size=2)
    assert levels["RB"] == 10.0  # last available, not a crash on out-of-range index


def test_replacement_level_picks_nth_ranked_player():
    roster = ["QB", "BN"]
    scored = _players("QB", [30.0, 20.0, 10.0])
    levels = vor.replacement_levels(scored, roster, league_size=2)
    assert levels["QB"] == 20.0  # 2nd-best QB is the replacement level for a 2-team league


def test_compute_vor_players_with_no_replacement_level_sort_last():
    roster = ["QB", "BN"]  # no RB slot at all in this league
    scored = _players("QB", [30.0, 20.0]) + _players("RB", [25.0])
    ranked = vor.compute_vor(scored, roster, league_size=2)
    assert ranked[-1]["position"] == "RB"
    assert ranked[-1]["vor"] is None
    assert ranked[0]["vor"] == 10.0  # 30 - replacement(20)


def test_assign_tiers_breaks_on_natural_gap():
    ranked = [{"player_id": f"p{i}", "vor": v}
              for i, v in enumerate([50, 48, 45, 20, 18, 15])]
    tiered = vor.assign_tiers(ranked)
    assert [p["tier"] for p in tiered] == [1, 1, 1, 2, 2, 2]


def test_assign_tiers_puts_unranked_players_in_trailing_tier():
    ranked = [{"vor": 50}, {"vor": 10}, {"vor": None}]
    tiered = vor.assign_tiers(ranked)
    assert tiered[-1]["vor"] is None
    assert tiered[-1]["tier"] > tiered[0]["tier"]
