"""Unit tests for the value-finder math."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wc.data.base import OddsLine
from wc.value.finder import implied_prob, to_decimal, model_prob_for, find_value


def test_implied_prob():
    assert abs(implied_prob(-110) - 0.5238) < 0.001
    assert abs(implied_prob(+100) - 0.5) < 1e-9
    assert abs(implied_prob(+150) - 0.4) < 0.001
    assert abs(implied_prob(-200) - 0.6667) < 0.001


def test_to_decimal():
    assert abs(to_decimal(+100) - 2.0) < 1e-9
    assert abs(to_decimal(-110) - 1.9091) < 0.001
    assert abs(to_decimal(+150) - 2.5) < 1e-9


def test_model_prob_for_all_markets():
    e = {"goal": {"anytime": 0.30}, "sot": {"over": {1.5: 0.40}},
         "card": {"yes": 0.20}, "saves": {"over": {2.5: 0.60}}}
    assert model_prob_for(e, "goal", None, "yes") == 0.30
    assert model_prob_for(e, "sot", 1.5, "over") == 0.40
    assert model_prob_for(e, "sot", 1.5, "under") == 0.60          # 1 - over
    assert model_prob_for(e, "saves", 2.5, "over") == 0.60
    assert model_prob_for(e, "sot", 9.5, "over") is None           # unmodeled line


def test_find_value_edge_and_ev():
    # model 50% at +100 (implied 50%, decimal 2.0) → zero edge, zero EV
    sim = {"players": {"X": {"team": "T", "confidence": "confirmed", "goal": {"anytime": 0.50}}}}
    r = find_value(sim, [OddsLine("m", "X", "goal", None, "yes", 100)])[0]
    assert abs(r["implied_prob"] - 0.50) < 1e-9
    assert abs(r["edge"]) < 1e-9
    assert abs(r["ev"]) < 1e-9

    # model 60% at +100 → edge +10pts, EV = 0.6*2 - 1 = +0.20
    sim2 = {"players": {"X": {"team": "T", "confidence": "c", "goal": {"anytime": 0.60}}}}
    r2 = find_value(sim2, [OddsLine("m", "X", "goal", None, "yes", 100)])[0]
    assert abs(r2["edge"] - 0.10) < 1e-9
    assert abs(r2["ev"] - 0.20) < 1e-9


def test_find_value_ranks_by_edge():
    sim = {"players": {
        "A": {"team": "T", "confidence": "c", "goal": {"anytime": 0.65}},
        "B": {"team": "T", "confidence": "c", "goal": {"anytime": 0.40}}}}
    rows = find_value(sim, [OddsLine("m", "A", "goal", None, "yes", 100),
                            OddsLine("m", "B", "goal", None, "yes", 100)])
    assert rows[0]["player"] == "A"                    # bigger edge first
    assert rows[0]["edge"] >= rows[1]["edge"]


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn(); print("ok", fn.__name__)
