"""
Tests for the backtesting harness metrics math (backtest.metrics / compare).

Pure functions on synthetic (line, projection, actual) points — no DB, no network.
"""
from __future__ import annotations

import backtest


def test_metrics_basic_mae_bias_hitrate():
    # projection always 0.5 above actual → bias +0.5, MAE 0.5
    pairs = [{"L": 1.5, "P": a + 0.5, "Y": a} for a in (1.0, 2.0, 3.0, 0.0)]
    m = backtest.metrics(pairs)
    assert m["n"] == 4
    assert m["mae"] == 0.5
    assert m["bias"] == 0.5


def test_metrics_hit_rate_counts_line_side():
    # model over (P>L) and actual over (Y>L) → hit; model over but actual under → miss
    pairs = [
        {"L": 1.5, "P": 2.0, "Y": 2.0},   # over/over → hit
        {"L": 1.5, "P": 2.0, "Y": 1.0},   # over/under → miss
        {"L": 1.5, "P": 1.0, "Y": 1.0},   # under/under → hit
        {"L": 1.5, "P": 1.5, "Y": 1.5},   # push → excluded
    ]
    m = backtest.metrics(pairs)
    assert m["decided"] == 3
    assert m["hit_rate"] == round(2 / 3, 4)


def test_metrics_empty():
    assert backtest.metrics([])["n"] == 0


def test_compare_flags_improvement():
    base = {"n": 100, "mae": 1.25, "hit_rate": 0.60}
    cand = {"n": 100, "mae": 1.10, "hit_rate": 0.63}
    c = backtest.compare(base, cand)
    assert c["by_metric"]["mae"]["better"] is True     # lower MAE is better
    assert c["by_metric"]["hit_rate"]["better"] is True  # higher hit-rate is better
    assert c["by_metric"]["mae"]["delta"] == -0.15


def test_compare_coverage_toward_target():
    # candidate coverage 0.78 is closer to 0.80 target than baseline 0.56 → better
    base = {"n": 100, "coverage": 0.56}
    cand = {"n": 100, "coverage": 0.78}
    c = backtest.compare(base, cand)
    assert c["by_metric"]["coverage"]["better"] is True
