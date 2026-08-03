"""
Characterization tests for the calibration/CLV math in db.py.

These pin the behavior of the honesty stack — per-stat trust (stat_gammas), P(over)
Platt calibration (prob_calibration), and floor/ceiling widening (interval_width) —
which are the numbers that make projections truthful. They run against a TEMPORARY
sqlite file (db.DB_PATH is redirected) so the real graded ledger is never touched.
"""
from __future__ import annotations

import numpy as np
import pytest

import provenance


def _insert(db, rows):
    """rows: list of dicts with the columns each test needs; missing → NULL.
    model_version defaults to the CURRENT provenance version for the row's sport, matching
    what db.py's calibration functions now filter to by default (2026-08 model-version-scoped
    calibration) — so these fixtures stay valid across future version bumps too."""
    cols = ("line_id", "game_date", "sport", "stat_type", "close_line", "close_proj",
            "model_raw", "close_prob", "model_raw_prob", "model_floor", "model_ceiling",
            "actual", "odds_type", "open_line", "open_proj", "model_version")
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


# ── stat_gammas: recovers the edge-regression slope (shrunk toward prior) ──────

def test_stat_gammas_recovers_known_slope():
    import db as _db
    # skip if env can't host the temp db; handled by fixture normally
    # Build data where (actual - line) = 0.8 * (model - line) exactly, large n so the
    # shrinkage toward prior (0.20, k=300) barely moves the estimate.
    rng = np.random.default_rng(3)
    n = 3000
    rows = []
    line = 1.5
    for m_minus_l in rng.uniform(-1.5, 1.5, n):
        m = line + m_minus_l
        y = line + 0.8 * m_minus_l
        rows.append({"stat_type": "Hits", "close_line": line, "close_proj": m,
                     "model_raw": m, "actual": y})
    import tempfile, os
    tmp = tempfile.mkdtemp()
    orig = _db.DB_PATH
    try:
        _db.DB_PATH = os.path.join(tmp, "t.db")
        _db.init_db()
        _insert(_db, rows)
        g = _db.stat_gammas(min_n=120)
        assert "hits" in g
        # true 0.8, shrunk toward 0.20 by k/(n+k)=300/3300 ≈ 9% → ~0.75
        assert 0.68 < g["hits"] < 0.82
    finally:
        _db.DB_PATH = orig


def test_calibration_never_pools_rows_across_a_model_version_change():
    # 2026-08: the whole point of scoping every calibration query to model_version. Insert
    # two cleanly-separable populations under two different versions -- an OLD-model era where
    # the model was uniformly overconfident, and a NEW-model era where it's honestly
    # calibrated -- and confirm stat_gammas/prob_calibration/interval_width/stat_biases each
    # read ONLY the current version's rows, not a blend of both eras.
    import db as _db
    import tempfile, os
    rng = np.random.default_rng(11)
    n = 2000
    old_rows, new_rows = [], []
    for _ in range(n):
        # OLD model: (actual-line) tracks only 20% of (model-line) -- mostly noise, defer to line
        m_minus_l = rng.uniform(-1.5, 1.5)
        old_rows.append({"stat_type": "Hits", "close_line": 1.5, "close_proj": 1.5 + m_minus_l,
                         "model_raw": 1.5 + m_minus_l, "actual": 1.5 + 0.2 * m_minus_l,
                         "model_version": "mlb-1.0.0"})
        # NEW model: (actual-line) tracks 90% of (model-line) -- the model is genuinely good now
        m_minus_l2 = rng.uniform(-1.5, 1.5)
        new_rows.append({"stat_type": "Hits", "close_line": 1.5, "close_proj": 1.5 + m_minus_l2,
                         "model_raw": 1.5 + m_minus_l2, "actual": 1.5 + 0.9 * m_minus_l2,
                         "model_version": "mlb-1.1.0"})
    tmp = tempfile.mkdtemp()
    orig = _db.DB_PATH
    try:
        _db.DB_PATH = os.path.join(tmp, "t.db")
        _db.init_db()
        _insert(_db, old_rows + new_rows)
        g_old = _db.stat_gammas(min_n=120, model_version="mlb-1.0.0")
        g_new = _db.stat_gammas(min_n=120, model_version="mlb-1.1.0")
        # old era's slope should land near its true 0.2, new era's near its true 0.9 -- if rows
        # were pooled across versions, both calls would return the same blended number instead
        assert g_old["hits"] < 0.4
        assert g_new["hits"] > 0.8
        assert g_old["hits"] != g_new["hits"]
    finally:
        _db.DB_PATH = orig


def test_stat_gammas_defers_when_model_equals_line():
    import db as _db, tempfile, os
    rows = [{"stat_type": "Runs", "close_line": 0.5, "close_proj": 0.5,
             "model_raw": 0.5, "actual": float(a % 2)} for a in range(400)]
    orig = _db.DB_PATH
    tmp = tempfile.mkdtemp()
    try:
        _db.DB_PATH = os.path.join(tmp, "t.db")
        _db.init_db()
        _insert(_db, rows)
        g = _db.stat_gammas(min_n=120)
        # zero model-vs-line variance → stat is skipped (can't estimate a slope)
        assert "runs" not in g
    finally:
        _db.DB_PATH = orig


# ── prob_calibration: overconfident probs → shrink slope a < 1 ────────────────

def test_prob_calibration_shrinks_overconfident_probs():
    import db as _db, tempfile, os
    rng = np.random.default_rng(5)
    n = 4000
    rows = []
    for _ in range(n):
        p = float(rng.uniform(0.05, 0.95))          # stated model prob
        true_p = 0.5 + (p - 0.5) * 0.5              # model is 2x overconfident
        y = 1.0 if rng.uniform() < true_p else 0.0
        # encode outcome relative to the line: over ⇔ actual > line
        line = 1.5
        actual = line + (0.5 if y > 0 else -0.5)
        rows.append({"stat_type": "Hits", "close_line": line, "actual": actual,
                     "close_prob": p, "model_raw_prob": p})
    orig = _db.DB_PATH
    tmp = tempfile.mkdtemp()
    try:
        _db.DB_PATH = os.path.join(tmp, "t.db")
        _db.init_db()
        _insert(_db, rows)
        cal = _db.prob_calibration(min_n=400)
        assert cal, "expected a calibration fit"
        assert 0.2 < cal["a"] < 1.0, "overconfident model should get a<1 (shrink to 0.5)"
    finally:
        _db.DB_PATH = orig


# ── interval_width: undercovering band → stretch factor > 1 ───────────────────

def test_interval_width_stretches_when_band_undercovers():
    import db as _db, tempfile, os
    # Build a band that covers only ~56% of outcomes (docs' measured MLB value); the
    # p10-p90 band should cover 80%, so the returned stretch must be > 1.
    rng = np.random.default_rng(9)
    n = 1000
    rows = []
    for _ in range(n):
        lo, hi = -1.0, 1.0
        # 56% inside [-1,1]: pick actual inside with prob .56, outside otherwise
        if rng.uniform() < 0.56:
            actual = float(rng.uniform(lo, hi))
        else:
            actual = float(hi + rng.uniform(0.1, 2.0)) if rng.uniform() < 0.5 \
                else float(lo - rng.uniform(0.1, 2.0))
        rows.append({"stat_type": "Hits", "close_line": 0.0, "actual": actual,
                     "model_floor": lo, "model_ceiling": hi})
    orig = _db.DB_PATH
    tmp = tempfile.mkdtemp()
    try:
        _db.DB_PATH = os.path.join(tmp, "t.db")
        _db.init_db()
        _insert(_db, rows)
        w = _db.interval_width(min_n=300)
        assert w > 1.0, "an undercovering band must be widened"
        assert w <= 3.0, "width is capped at 3.0"
    finally:
        _db.DB_PATH = orig
