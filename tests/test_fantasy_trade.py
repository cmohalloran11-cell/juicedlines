"""
Tests for fantasy.trade — pure trade evaluation (phase 4). No I/O; the route handler supplies
an already-VOR'd board keyed by canonical player_id.
"""
from __future__ import annotations

from fantasy import trade

BOARD = {
    "p1": {"player_id": "p1", "name": "A", "vor": 20.0},
    "p2": {"player_id": "p2", "name": "B", "vor": 15.0},
    "p3": {"player_id": "p3", "name": "C", "vor": 5.0},
    "p4": {"player_id": "p4", "name": "D", "vor": None},   # no replacement level for their position
}


def test_evaluate_trade_even_when_total_vor_matches():
    result = trade.evaluate_trade(BOARD, ["p1"], ["p2", "p3"])
    assert result["side_a"]["total_vor"] == 20.0
    assert result["side_b"]["total_vor"] == 20.0
    assert result["vor_diff"] == 0.0
    assert result["verdict"] == "Roughly even trade"


def test_evaluate_trade_reports_which_side_gains():
    result = trade.evaluate_trade(BOARD, ["p1", "p2"], ["p3"])
    assert result["vor_diff"] == 30.0
    assert "Side A gains 30.0 VOR" == result["verdict"]

    reversed_result = trade.evaluate_trade(BOARD, ["p3"], ["p1", "p2"])
    assert reversed_result["vor_diff"] == -30.0
    assert "Side B gains 30.0 VOR" == reversed_result["verdict"]


def test_evaluate_trade_flags_missing_players_instead_of_zero_valuing_silently():
    result = trade.evaluate_trade(BOARD, ["p1", "not-on-board"], ["p2"])
    assert result["side_a"]["missing_player_ids"] == ["not-on-board"]
    assert result["side_a"]["total_vor"] == 20.0   # only p1 counted, not silently treated as 0
    assert len(result["side_a"]["players"]) == 1


def test_evaluate_trade_null_vor_player_contributes_zero_not_an_error():
    result = trade.evaluate_trade(BOARD, ["p4"], ["p3"])
    assert result["side_a"]["total_vor"] == 0.0
    assert result["side_a"]["missing_player_ids"] == []   # p4 IS on the board, just vor=None


def test_evaluate_trade_within_threshold_is_even():
    board = {"p1": {"vor": 10.4}, "p2": {"vor": 10.0}}
    result = trade.evaluate_trade(board, ["p1"], ["p2"])
    assert result["verdict"] == "Roughly even trade"   # 0.4 diff is below EVEN_THRESHOLD
