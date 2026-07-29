"""
Tests for analytics._tennis_actual — the tennis grading pipeline's pure stat-derivation
function (2026-07-29 tennis engine rebuild's automatic calibration requirement). Given one
ESPN completed-match result (final set-by-set games), derives the actual value for
whichever market a logged prop was on. No network — the match dict is synthetic.
"""
from __future__ import annotations

import analytics


def _match(a_games, b_games):
    return {"a_games": a_games, "b_games": b_games}


def test_total_games_and_per_player_games_won():
    m = _match([6, 6], [4, 7])   # a wins set1 6-4, b wins set2 7-6 → a:12 games, b:11 games
    assert analytics._tennis_actual(m, "a", "Total Games") == 23
    assert analytics._tennis_actual(m, "a", "Games Won") == 12
    assert analytics._tennis_actual(m, "b", "Games Won") == 11


def test_sets_won_and_total_sets():
    m = _match([6, 6, 6], [4, 7, 2])   # a wins sets 1 and 3, b wins set 2 → straight-ish match
    assert analytics._tennis_actual(m, "a", "Sets Won") == 2
    assert analytics._tennis_actual(m, "b", "Sets Won") == 1
    assert analytics._tennis_actual(m, "a", "Total Sets") == 3
    assert analytics._tennis_actual(m, "b", "Total Sets") == 3


def test_ungradeable_markets_return_none_not_a_guess():
    # ESPN's free result has no point-level data — aces/DFs/break-points must be honestly
    # ungradeable (voided by grade_tennis), never a fabricated value.
    m = _match([6, 4], [3, 6])
    for label in ("Aces", "Double Faults", "Breakpoints Won"):
        assert analytics._tennis_actual(m, "a", label) is None


def test_unrecognized_market_label_returns_none():
    m = _match([6, 4], [3, 6])
    assert analytics._tennis_actual(m, "a", "1st Set Winner") is None
