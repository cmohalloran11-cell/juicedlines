"""
Tests for fantasy.draft_state — pure live-draft logic (best-available, run detection,
need-weighted recommendations). Everything here takes an already-fetched pick list; nothing
does I/O, matching valuation.py's boundary.
"""
from __future__ import annotations

from fantasy import draft_state


def test_remove_drafted_filters_by_player_id():
    board = [{"player_id": "a"}, {"player_id": "b"}, {"player_id": "c"}]
    picks = [{"player_id": "b"}]
    remaining = draft_state.remove_drafted(board, picks)
    assert [p["player_id"] for p in remaining] == ["a", "c"]


def test_positional_runs_flags_when_threshold_met_in_window():
    picks = [{"position": "RB"}, {"position": "WR"}, {"position": "RB"},
            {"position": "RB"}, {"position": "RB"}]
    runs = draft_state.positional_runs(picks, window=5, run_threshold=3)
    assert runs == {"RB": 4}


def test_positional_runs_ignores_picks_outside_window():
    old = [{"position": "RB"}] * 3
    recent = [{"position": "WR"}] * 2
    runs = draft_state.positional_runs(old + recent, window=2, run_threshold=3)
    assert runs == {}  # only the last 2 picks are considered, and 2 < threshold of 3


def test_remaining_needs_subtracts_drafted_starters():
    roster = ["QB", "RB", "RB", "WR", "FLEX", "BN"]
    drafted = [{"position": "RB"}]
    needs = draft_state.remaining_needs(roster, drafted)
    assert needs["QB"] == 1
    assert needs["RB"] == 1     # 2 slots - 1 already drafted
    assert needs["WR"] == 1
    assert needs["FLEX"] == 1   # FLEX need isn't pre-collapsed onto a base position


def test_recommend_boosts_players_at_open_need_positions():
    board = [
        {"player_id": "a", "position": "RB", "vor": 10.0},
        {"player_id": "b", "position": "WR", "vor": 10.0},
    ]
    roster = ["RB", "BN"]
    recs = draft_state.recommend(board, roster, drafted_by_user=[])
    assert recs[0]["player_id"] == "a"          # RB is the only open starting need
    assert recs[0]["need_weight"] == 1.15
    assert recs[1]["need_weight"] == 1.0


def test_recommend_gives_flex_boost_to_flex_eligible_position():
    board = [{"player_id": "a", "position": "TE", "vor": 5.0}]
    roster = ["QB", "FLEX", "BN"]   # QB already filled, only FLEX open
    drafted = [{"position": "QB"}]
    recs = draft_state.recommend(board, roster, drafted_by_user=drafted)
    assert recs[0]["need_weight"] == 1.08   # TE is FLEX-eligible, not a full starter boost


def test_recommend_respects_top_n():
    board = [{"player_id": str(i), "position": "WR", "vor": float(i)} for i in range(20)]
    recs = draft_state.recommend(board, ["WR", "BN"], drafted_by_user=[], top_n=5)
    assert len(recs) == 5
    assert recs[0]["player_id"] == "19"  # highest vor first
