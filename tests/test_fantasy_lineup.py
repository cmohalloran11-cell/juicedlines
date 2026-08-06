"""
Tests for fantasy.lineup — the pure optimal-starting-lineup assignment (phase 2). No I/O,
matching valuation.py's boundary.
"""
from __future__ import annotations

from fantasy import lineup


def _p(pid, position, points):
    return {"player_id": pid, "position": position, "fantasy_points": points}


ROSTER = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN"]
PLAYERS = [
    _p("qb1", "QB", 25.0),
    _p("rb1", "RB", 20.0), _p("rb2", "RB", 15.0), _p("rb3", "RB", 10.0),
    _p("wr1", "WR", 18.0), _p("wr2", "WR", 14.0), _p("wr3", "WR", 8.0),
    _p("te1", "TE", 9.0),
]


def test_optimal_lineup_fills_single_position_slots_with_top_players():
    result = lineup.optimal_lineup(PLAYERS, ROSTER)
    slots = {s["slot"]: s["player"]["player_id"] for s in result["starters"] if s["player"]}
    assert slots["QB"] == "qb1"
    # two RB slots -> the two best RBs (order among identical slots isn't semantically meaningful)
    rb_ids = [s["player"]["player_id"] for s in result["starters"] if s["slot"] == "RB"]
    assert set(rb_ids) == {"rb1", "rb2"}
    wr_ids = [s["player"]["player_id"] for s in result["starters"] if s["slot"] == "WR"]
    assert set(wr_ids) == {"wr1", "wr2"}
    assert slots["TE"] == "te1"


def test_optimal_lineup_flex_gets_best_remaining_eligible_player():
    result = lineup.optimal_lineup(PLAYERS, ROSTER)
    flex = next(s for s in result["starters"] if s["slot"] == "FLEX")
    assert flex["player"]["player_id"] == "rb3"   # best of the leftover {rb3=10, wr3=8}


def test_optimal_lineup_projected_points_and_bench():
    result = lineup.optimal_lineup(PLAYERS, ROSTER)
    assert result["projected_points"] == 25 + 20 + 15 + 18 + 14 + 9 + 10   # = 111.0
    assert [p["player_id"] for p in result["bench"]] == ["wr3"]


def test_optimal_lineup_empty_slot_when_no_eligible_player():
    thin_roster = ["QB", "TE", "BN"]
    thin_players = [_p("qb1", "QB", 25.0)]   # no TE at all
    result = lineup.optimal_lineup(thin_players, thin_roster)
    te_slot = next(s for s in result["starters"] if s["slot"] == "TE")
    assert te_slot["player"] is None
    assert result["projected_points"] == 25.0   # TE's absence doesn't crash or fabricate a score


def test_optimal_lineup_ignores_bench_slots():
    result = lineup.optimal_lineup(PLAYERS, ROSTER)
    assert all(s["slot"] != "BN" for s in result["starters"])


def test_lineup_delta_reports_points_gained_and_swaps():
    optimal = lineup.optimal_lineup(PLAYERS, ROSTER)
    scored_by_id = {p["player_id"]: p for p in PLAYERS}
    # Currently-set lineup benched rb3 (the correct FLEX play) in favor of wr3.
    current_starters = ["qb1", "rb1", "rb2", "wr1", "wr2", "te1", "wr3"]
    delta = lineup.lineup_delta(optimal, current_starters, scored_by_id)
    assert delta["current_points"] == 25 + 20 + 15 + 18 + 14 + 9 + 8   # = 109.0
    assert delta["optimal_points"] == 111.0
    assert delta["points_gained"] == 2.0
    assert delta["start"] == ["rb3"]
    assert delta["sit"] == ["wr3"]


def test_lineup_delta_zero_gain_when_already_optimal():
    optimal = lineup.optimal_lineup(PLAYERS, ROSTER)
    scored_by_id = {p["player_id"]: p for p in PLAYERS}
    current_starters = [s["player"]["player_id"] for s in optimal["starters"] if s["player"]]
    delta = lineup.lineup_delta(optimal, current_starters, scored_by_id)
    assert delta["points_gained"] == 0.0
    assert delta["start"] == [] and delta["sit"] == []
