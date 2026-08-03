"""
Tests for backtest.py's self-diagnostic layer: does the model's own stated confidence,
method, and per-book pricing actually correlate with real accuracy? Runs against a
temporary sqlite file (db.DB_PATH redirected) so the real graded ledger is never touched —
same pattern as tests/test_calibration_characterization.py.
"""
from __future__ import annotations

import os
import tempfile

import db as _db
import backtest
import provenance


def _insert(rows):
    """rows: list of dicts; missing columns → NULL. Each row gets a distinct synthetic
    player so backtest._ledger_rows' (player, stat, game_date) dedup never collides them.
    model_version defaults to the sport's CURRENT version so these rows survive
    backtest._ledger_rows' default model_version scoping (2026-08) — same pattern as
    tests/test_calibration_characterization.py's _insert."""
    cols = ("line_id", "game_date", "sport", "player", "stat_type", "close_line",
            "close_proj", "actual", "close_prob", "model_raw_prob", "model_floor",
            "model_ceiling", "proj_kind", "source", "odds_type", "model_version")
    with _db._lock, _db._conn() as c:
        for i, r in enumerate(rows):
            r = {"line_id": f"L{i}", "game_date": "2026-07-01", "sport": "MLB",
                 "player": f"P{i}", "stat_type": "Hits", "odds_type": "standard", **r}
            r.setdefault("model_version", provenance.model_version(r["sport"]))
            c.execute(f"INSERT INTO prop_clv ({','.join(cols)}) "
                      f"VALUES ({','.join('?' for _ in cols)})", [r.get(k) for k in cols])
        c.commit()


def _with_temp_db(fn):
    orig = _db.DB_PATH
    tmp = tempfile.mkdtemp()
    try:
        _db.DB_PATH = os.path.join(tmp, "t.db")
        _db.init_db()
        fn()
    finally:
        _db.DB_PATH = orig


def test_calibration_by_confidence_flags_an_untrustworthy_signal():
    def run():
        rows = []
        # NARROW band (confident) rows: WRONG most of the time (line=1.5, actual far off).
        for i in range(40):
            rows.append({"close_line": 1.5, "close_proj": 1.6, "actual": 0.0,
                         "model_floor": 1.5, "model_ceiling": 1.7})   # tiny band
        # WIDE band (uncertain) rows: RIGHT most of the time.
        for i in range(40, 80):
            rows.append({"close_line": 1.5, "close_proj": 1.6, "actual": 3.0,
                         "model_floor": 0.0, "model_ceiling": 5.0})   # huge band
        _insert(rows)
        out = backtest.calibration_by_confidence("MLB", n_buckets=2)
        assert out["confidence_signal_trustworthy"] is False
        narrow, wide = out["buckets"]
        assert narrow["hit_rate"] < wide["hit_rate"]
    _with_temp_db(run)


def test_calibration_by_confidence_insufficient_data_is_honest():
    def run():
        _insert([{"close_line": 1.5, "close_proj": 1.6, "actual": 2.0,
                  "model_floor": 1.0, "model_ceiling": 2.0}])
        out = backtest.calibration_by_confidence("MLB", n_buckets=3)
        assert out["status"] == "insufficient_data"
    _with_temp_db(run)


def test_calibration_by_method_splits_engine_vs_other():
    def run():
        rows = []
        for i in range(25):
            rows.append({"close_line": 1.5, "close_proj": 2.0, "actual": 2.0,
                         "proj_kind": "engine"})               # always right
        for i in range(25, 50):
            rows.append({"close_line": 1.5, "close_proj": 2.0, "actual": 0.0,
                         "proj_kind": "empirical"})             # always wrong
        _insert(rows)
        out = backtest.calibration_by_method("MLB", min_n=20)
        assert out["by_method"]["engine"]["hit_rate"] == 1.0
        assert out["by_method"]["empirical"]["hit_rate"] == 0.0
    _with_temp_db(run)


def test_calibration_by_book_splits_by_source():
    def run():
        rows = []
        for i in range(25):
            rows.append({"close_line": 1.5, "close_proj": 2.0, "actual": 2.0, "source": "bookA"})
        for i in range(25, 50):
            rows.append({"close_line": 1.5, "close_proj": 2.0, "actual": 0.0, "source": "bookB"})
        _insert(rows)
        out = backtest.calibration_by_book("MLB", min_n=20)
        assert out["by_book"]["bookA"]["hit_rate"] == 1.0
        assert out["by_book"]["bookB"]["hit_rate"] == 0.0
    _with_temp_db(run)


def test_backtest_never_pools_accuracy_across_a_model_version_change():
    # 2026-08 fix: _ledger_rows (and everything built on it -- current_accuracy, drift,
    # the calibration_by_* diagnostics) never filtered by model_version at all, so a
    # deliberate engine change (e.g. this session's two-stage uncertainty propagation)
    # would have its measured accuracy silently averaged with the era BEFORE the change --
    # masking a regression or fabricating an improvement that was really just old data.
    def run():
        rows = []
        for i in range(60):     # old version: always WRONG
            rows.append({"close_line": 1.5, "close_proj": 2.0, "actual": 0.0,
                         "model_version": "mlb-OLD"})
        for i in range(60, 120):  # current version: always RIGHT
            rows.append({"close_line": 1.5, "close_proj": 2.0, "actual": 2.0,
                         "model_version": provenance.model_version("MLB")})
        _insert(rows)
        acc = backtest.current_accuracy("MLB", min_n=10)
        assert acc["overall"]["hit_rate"] == 1.0, "must reflect ONLY the current version"
        assert acc["model_version"] == provenance.model_version("MLB")

        by_version = backtest.version_history("MLB", min_n=10)
        assert by_version["by_version"]["mlb-OLD"]["hit_rate"] == 0.0
        assert by_version["by_version"][provenance.model_version("MLB")]["hit_rate"] == 1.0
    _with_temp_db(run)


def test_diagnostics_bundles_all_three():
    def run():
        _insert([{"close_line": 1.5, "close_proj": 2.0, "actual": 2.0,
                  "proj_kind": "engine", "source": "bookA"} for _ in range(1)])
        out = backtest.diagnostics("MLB")
        assert set(out.keys()) == {"sport", "confidence_calibration", "method_comparison",
                                    "book_comparison"}
    _with_temp_db(run)
