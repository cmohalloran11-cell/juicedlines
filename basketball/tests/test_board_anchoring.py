"""
Regression test for the mean-vs-median market-anchoring bug (see nfl/board.py's 2026-08 fix
for the full mechanism, ported here). basketball/board.py used to blend the model's MEAN
toward the market line and recenter the simulated array so its MEAN sat on the line — for a
right-skewed stat (WNBA rebounds/points/assists, all Gamma-shaped low counts), that leaves the
MEDIAN (what model_prob is actually computed from) below the line, producing a systematic
Under bias whenever trust is low. Fixed by blending/shifting on the MEDIAN instead.

Monkeypatches P.project_player/P.market_dist directly (no live data source needed) so this
is a pure test of the blending math in attach_basketball, independent of whichever real
data source (balldontlie.io, ESPN, ...) is wired at the time.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest

from basketball import board as B
from basketball import projections as P


def _right_skewed_array(n=20000, seed=7):
    # Gamma(shape=2, scale=6) — mean=12, but right-skewed so median < mean, exactly the
    # shape family every yardage/count stat this codebase models looks like.
    rng = np.random.default_rng(seed)
    return rng.gamma(2.0, 6.0, n)


def test_zero_trust_wnba_line_reads_as_a_coinflip_not_a_systematic_under(monkeypatch):
    arr = _right_skewed_array()
    raw_mean, raw_median = float(arr.mean()), float(np.median(arr))
    assert raw_mean > raw_median, "sanity check: the synthetic array is genuinely right-skewed"

    monkeypatch.setattr(B.P, "project_player",
                        lambda league, player: {"sample_weight": 0.0, "n_games": 0,
                                                "confidence": "low"})
    monkeypatch.setattr(B.P, "market_dist", lambda proj, stat_type: arr)
    monkeypatch.setattr(B.P, "gamelog_source",
                        lambda: type("S", (), {"injuries": lambda self, lg: {}})())

    line_val = round(raw_median, 1)   # the market posts a line AT the true median (fair line)
    lines = [{"sport": "WNBA", "player": "Nobody Yet", "line": line_val,
             "stat_type": "Points", "odds_type": "standard"}]

    done = B.attach_basketball(lines)
    assert done == 1
    l = lines[0]
    assert l["trust_weight"] == 0.0
    assert 0.40 <= l["model_prob"] <= 0.60, (
        f"zero-trust line at the true median should read near a coinflip, got "
        f"model_prob={l['model_prob']}")
    assert l["model_proj"] > l["line"], (
        "a right-skewed stat's honest mean sits above a line set at the median")
