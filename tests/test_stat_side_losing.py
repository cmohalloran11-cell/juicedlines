"""
Tests for db.stat_side_losing — the confirmed-losing-side finding, more severe than
stat_gammas' "no proven edge". stat_gammas' gamma~=0 means the model's signal is
UNCORRELATED with outcomes; stat_side_losing means the model's actual recommended side has
measured hit rate BELOW breakeven — a wrong-direction signal. Live audit (2026-08-05): MLB
"Pitching Outs" was recommended Under 91.4% of the time on the live board, hitting only
33.8% (n=139) while the rarely-recommended Over hit 53.8%.
"""
from __future__ import annotations

import os
import tempfile

import pytest

import provenance


def _insert(db, rows):
    cols = ("line_id", "game_date", "sport", "stat_type", "close_line",
           "model_raw_prob", "close_prob", "actual", "odds_type", "model_version")
    with db._lock, db._conn() as c:
        for i, r in enumerate(rows):
            r = {"line_id": f"L{i}", "game_date": "2026-07-01", "sport": "MLB",
                 "odds_type": "standard", **r}
            r.setdefault("model_version", provenance.model_version(r["sport"]))
            c.execute(
                f"INSERT INTO prop_clv ({','.join(cols)}) "
                f"VALUES ({','.join('?' for _ in cols)})",
                [r.get(k) for k in cols])
        c.commit()


def _with_temp_db(fn):
    import db as _db
    tmp = tempfile.mkdtemp()
    orig = _db.DB_PATH
    try:
        _db.DB_PATH = os.path.join(tmp, "t.db")
        _db.init_db()
        fn(_db)
    finally:
        _db.DB_PATH = orig


def test_confirmed_losing_side_is_flagged():
    def run(_db):
        # model_raw_prob=0.4 -> recommends "under" every time. actual > line (over wins) on
        # 100/150 rows, so the "under" recommendation itself only wins 50/150 = 33%.
        rows = [{"stat_type": "Pitching Outs", "close_line": 14.5, "model_raw_prob": 0.4,
                "actual": 15.5 if i < 100 else 13.5} for i in range(150)]
        _insert(_db, rows)
        losing = _db.stat_side_losing("MLB", min_n=60)
        assert ("pitching outs", "under") in losing
        d = losing[("pitching outs", "under")]
        assert d["n"] == 150
        assert d["hit_rate"] == pytest.approx(50 / 150, abs=0.01)
    _with_temp_db(run)


def test_winning_side_is_not_flagged():
    def run(_db):
        # "under" recommended (prob=0.3); wins 110/150 = 73%, well above breakeven.
        rows = [{"stat_type": "Walks", "close_line": 0.5, "model_raw_prob": 0.3,
                "actual": 0.0 if i < 110 else 1.0} for i in range(150)]
        _insert(_db, rows)
        losing = _db.stat_side_losing("MLB", min_n=60)
        assert ("walks", "under") not in losing
    _with_temp_db(run)


def test_thin_sample_is_not_flagged_even_if_losing():
    def run(_db):
        # Same losing shape as the confirmed case, but only 20 rows (below min_n=60) —
        # a thin sample can't trigger this on its own.
        rows = [{"stat_type": "Rare Stat", "close_line": 5.5, "model_raw_prob": 0.4,
                "actual": 6.5 if i < 15 else 4.5} for i in range(20)]
        _insert(_db, rows)
        losing = _db.stat_side_losing("MLB", min_n=60)
        assert ("rare stat", "under") not in losing
    _with_temp_db(run)


def test_shrinkage_pulls_a_small_sample_toward_breakeven_not_the_raw_rate():
    # Same raw hit rate (~33%), one sample much smaller than the other — the smaller sample's
    # shrunk rate should sit meaningfully closer to breakeven (0.5238) than the larger
    # sample's, since k=50 is a bigger fraction of a small n. Demonstrates real shrinkage
    # behavior rather than asserting the mathematically-impossible "shrinks past breakeven"
    # (a weighted average of a below-breakeven raw rate and breakeven itself can never
    # exceed breakeven, for any sample size — only get arbitrarily close as n grows).
    def small(_db):
        rows = [{"stat_type": "Small N Stat", "close_line": 2.5, "model_raw_prob": 0.4,
                "actual": 3.5 if i < 40 else 1.5} for i in range(60)]   # 20/60 = 33% hit
        _insert(_db, rows)
        return _db.stat_side_losing("MLB", min_n=60)[("small n stat", "under")]

    def large(_db):
        rows = [{"stat_type": "Large N Stat", "close_line": 2.5, "model_raw_prob": 0.4,
                "actual": 3.5 if i < 400 else 1.5} for i in range(600)]  # 200/600 = 33% hit
        _insert(_db, rows)
        return _db.stat_side_losing("MLB", min_n=60)[("large n stat", "under")]

    result = {}
    _with_temp_db(lambda db: result.update(small=small(db)))
    _with_temp_db(lambda db: result.update(large=large(db)))
    assert result["small"]["hit_rate"] == pytest.approx(1 / 3, abs=0.01)
    assert result["large"]["hit_rate"] == pytest.approx(1 / 3, abs=0.01)
    gap_small = 0.5238 - result["small"]["shrunk_hit_rate"]
    gap_large = 0.5238 - result["large"]["shrunk_hit_rate"]
    assert gap_small < gap_large   # small sample pulled closer to breakeven than the large one
