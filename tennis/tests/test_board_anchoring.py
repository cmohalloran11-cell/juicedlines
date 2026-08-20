"""
Regression test for the mean-vs-median market-anchoring bug (see nfl/board.py's 2026-08 fix
for the full mechanism, ported here). tennis/board.py used to blend the model's MEAN toward
the market line and recenter the simulated array so its MEAN sat on the line — for a
right-skewed count stat (games won, aces, ...), that leaves the MEDIAN (what model_prob is
actually computed from) below the line, producing a systematic Under bias whenever trust is
low. Fixed by blending/shifting on the MEDIAN instead — and since tennis is currently fully
anchored on EVERY line (see board.py's own proj_kind comment: the mirror history is too
stale for real model trust), this bug affected the entire tennis board, not an edge case.

Monkeypatches P.project_match/P.market_dist and the ESPN-derived helpers directly (no live
data/model lookups) so this is a pure test of the blending math in attach_tennis.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from tennis import board as B


def _right_skewed_array(n=20000, seed=11):
    # Gamma(shape=2, scale=3) — mean=6, right-skewed, the same family every count stat this
    # codebase models (games won, aces) looks like.
    rng = np.random.default_rng(seed)
    return rng.gamma(2.0, 3.0, n)


def test_zero_trust_tennis_line_reads_as_a_coinflip_not_a_systematic_under(monkeypatch):
    arr = _right_skewed_array()
    raw_mean, raw_median = float(arr.mean()), float(np.median(arr))
    assert raw_mean > raw_median, "sanity check: the synthetic array is genuinely right-skewed"

    monkeypatch.setattr(B, "_pick_tour", lambda a, b: "ATP")
    monkeypatch.setattr(B, "_best_of", lambda lines: 3)
    monkeypatch.setattr(B, "_live_surface_lookup", lambda tour: {})
    monkeypatch.setattr(B, "_recent_match_counts", lambda tour, days=7: {})
    monkeypatch.setattr(B.P, "project_match",
                        lambda tour, a, b, surface, **kw: {
                            "confidence": "low", "model_agreement": 1.0,
                            "eff_matches": 0,
                        })
    monkeypatch.setattr(B.P, "market_dist", lambda res, player, label: arr)

    line_val = round(raw_median, 1)   # the market posts a line AT the true median (fair line)
    lines = [{"sport": "Tennis", "player": "Player A", "matchup": "Player B",
             "line": line_val, "stat_type": "Games Won", "odds_type": "standard"}]

    done = B.attach_tennis(lines)
    assert done == 1
    l = lines[0]
    assert l["proj_kind"] == "market", "confidence='low' -> fully anchored, as documented"
    assert 0.40 <= l["model_prob"] <= 0.60, (
        f"zero-trust line at the true median should read near a coinflip, got "
        f"model_prob={l['model_prob']}")
    assert l["model_proj"] > l["line"], (
        "a right-skewed stat's honest mean sits above a line set at the median")
