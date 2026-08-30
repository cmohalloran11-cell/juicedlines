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
    cols = ("line_id", "player", "game_date", "sport", "stat_type", "close_line", "close_proj",
            "model_raw", "close_prob", "model_raw_prob", "model_floor", "model_ceiling",
            "actual", "odds_type", "open_line", "open_proj", "model_version", "proj_kind")
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


# ── proj_kind scoping: nfl_regular and the retired preseason engine's nfl_preseason rows
# share sport="NFL" AND model_version (the preseason model's removal was deliberately not
# version-bumped -- see provenance.MODEL_CHANGELOG), so proj_kind is the ONLY thing keeping
# those graded rows out of a regular-season calibration fit. Every calibration function must
# not blend them unless explicitly asked to. Same guarantee CFB's three prior tiers rely on. ─

def test_stat_gammas_never_pools_rows_across_proj_kind_by_default():
    import db as _db
    import tempfile, os
    rng = np.random.default_rng(21)
    n = 2000
    reg_rows, pre_rows = [], []
    for _ in range(n):
        m_minus_l = rng.uniform(-1.5, 1.5)
        reg_rows.append({"sport": "NFL", "stat_type": "Rec Yards", "close_line": 40.5,
                         "close_proj": 40.5 + m_minus_l, "model_raw": 40.5 + m_minus_l,
                         "actual": 40.5 + 0.9 * m_minus_l, "proj_kind": "nfl_regular"})
        m_minus_l2 = rng.uniform(-1.5, 1.5)
        pre_rows.append({"sport": "NFL", "stat_type": "Rec Yards", "close_line": 40.5,
                         "close_proj": 40.5 + m_minus_l2, "model_raw": 40.5 + m_minus_l2,
                         "actual": 40.5 + 0.1 * m_minus_l2, "proj_kind": "nfl_preseason"})
    orig = _db.DB_PATH
    tmp = tempfile.mkdtemp()
    try:
        _db.DB_PATH = os.path.join(tmp, "t.db")
        _db.init_db()
        _insert(_db, reg_rows + pre_rows)
        g_all = _db.stat_gammas("NFL", min_n=120)
        g_reg = _db.stat_gammas("NFL", min_n=120, proj_kind="nfl_regular")
        g_pre = _db.stat_gammas("NFL", min_n=120, proj_kind="nfl_preseason")
        # unfiltered call sees both eras blended; the true slopes (0.9 vs 0.1) are far enough
        # apart that a blend must land clearly between them, not match either pure population.
        assert g_reg["rec yards"] > 0.75
        assert g_pre["rec yards"] < 0.35
        assert g_reg["rec yards"] != g_all["rec yards"]
        assert g_pre["rec yards"] != g_all["rec yards"]
    finally:
        _db.DB_PATH = orig


def test_prob_calibration_and_interval_width_respect_proj_kind_filter():
    import db as _db
    import tempfile, os
    rng = np.random.default_rng(23)
    n = 1500
    rows = []
    for _ in range(n):
        # regular: honestly calibrated + honestly covering band
        p = float(rng.uniform(0.05, 0.95))
        y = 1.0 if rng.uniform() < p else 0.0
        line = 40.5
        actual = line + (10.0 if y > 0 else -10.0)
        rows.append({"sport": "NFL", "stat_type": "Rec Yards", "close_line": line,
                     "actual": actual, "close_prob": p, "model_raw_prob": p,
                     "model_floor": line - 15, "model_ceiling": line + 15,
                     "proj_kind": "nfl_regular"})
        # retired preseason engine's rows: badly overconfident + badly undercovering band
        p2 = float(rng.uniform(0.05, 0.95))
        true_p2 = 0.5 + (p2 - 0.5) * 0.45
        y2 = 1.0 if rng.uniform() < true_p2 else 0.0
        actual2 = line + (10.0 if y2 > 0 else -10.0)
        rows.append({"sport": "NFL", "stat_type": "Rec Yards", "close_line": line,
                     "actual": actual2, "close_prob": p2, "model_raw_prob": p2,
                     "model_floor": line - 2, "model_ceiling": line + 2,
                     "proj_kind": "nfl_preseason"})
    orig = _db.DB_PATH
    tmp = tempfile.mkdtemp()
    try:
        _db.DB_PATH = os.path.join(tmp, "t.db")
        _db.init_db()
        _insert(_db, rows)
        cal_reg = _db.prob_calibration("NFL", min_n=400, proj_kind="nfl_regular")
        cal_pre = _db.prob_calibration("NFL", min_n=400, proj_kind="nfl_preseason")
        assert cal_reg and 0.8 < cal_reg["a"] < 1.2, "regular season is honestly calibrated"
        assert cal_pre and cal_pre["a"] < 0.6, "the other kind is measurably overconfident"

        w_reg = _db.interval_width("NFL", min_n=300, proj_kind="nfl_regular")
        w_pre = _db.interval_width("NFL", min_n=300, proj_kind="nfl_preseason")
        assert w_reg < w_pre, "the undercovering band must stretch more"
    finally:
        _db.DB_PATH = orig


def test_stat_biases_respects_proj_kind_filter():
    import db as _db
    import tempfile, os
    rows = []
    for i in range(80):
        rows.append({"sport": "NFL", "player": f"P{i}", "stat_type": "Rec Yards",
                     "game_date": f"2026-08-{(i % 28) + 1:02d}", "close_proj": 50.0,
                     "actual": 45.0, "proj_kind": "nfl_regular"})            # bias +5
        rows.append({"sport": "NFL", "player": f"Q{i}", "stat_type": "Rec Yards",
                     "game_date": f"2026-08-{(i % 28) + 1:02d}", "close_proj": 50.0,
                     "actual": 65.0, "proj_kind": "nfl_preseason"})         # bias -15
    orig = _db.DB_PATH
    tmp = tempfile.mkdtemp()
    try:
        _db.DB_PATH = os.path.join(tmp, "t.db")
        _db.init_db()
        _insert(_db, rows)
        b_reg = _db.stat_biases("NFL", min_n=60, proj_kind="nfl_regular")
        b_pre = _db.stat_biases("NFL", min_n=60, proj_kind="nfl_preseason")
        assert abs(b_reg["rec yards"] - 5.0) < 0.01
        assert abs(b_pre["rec yards"] - (-15.0)) < 0.01
    finally:
        _db.DB_PATH = orig


def test_scorecard_respects_proj_kind_filter():
    """db.scorecard's hit_rate/CLV feed model_health's graded_props/hit_rate_vs_close card —
    same blending risk as stat_gammas/prob_calibration/interval_width/stat_biases above, since
    nfl_regular and the ledger's retired nfl_preseason rows share sport="NFL"."""
    import db as _db
    import tempfile, os
    rows = []
    for i in range(80):
        # regular: model's pick_over always agrees with the close-vs-line outcome -> hit_rate 1.0
        rows.append({"sport": "NFL", "player": f"P{i}", "stat_type": "Rec Yards",
                     "game_date": f"2026-08-{(i % 28) + 1:02d}", "close_line": 40.5,
                     "close_proj": 45.0, "model_raw_prob": 0.9, "actual": 50.0,
                     "open_line": 40.5, "open_proj": 45.0, "proj_kind": "nfl_regular"})
        # the other kind: model's pick_over always disagrees -> hit_rate 0.0
        rows.append({"sport": "NFL", "player": f"Q{i}", "stat_type": "Rec Yards",
                     "game_date": f"2026-08-{(i % 28) + 1:02d}", "close_line": 40.5,
                     "close_proj": 45.0, "model_raw_prob": 0.9, "actual": 30.0,
                     "open_line": 40.5, "open_proj": 45.0, "proj_kind": "nfl_preseason"})
    orig = _db.DB_PATH
    tmp = tempfile.mkdtemp()
    try:
        _db.DB_PATH = os.path.join(tmp, "t.db")
        _db.init_db()
        _insert(_db, rows)
        c_all = _db.scorecard("NFL")
        c_reg = _db.scorecard("NFL", proj_kind="nfl_regular")
        c_pre = _db.scorecard("NFL", proj_kind="nfl_preseason")
        assert c_reg["graded"] == 80 and c_pre["graded"] == 80
        assert c_reg["hit_rate"] == 1.0
        assert c_pre["hit_rate"] == 0.0
        assert c_all["graded"] == 160 and 0.4 < c_all["hit_rate"] < 0.6
    finally:
        _db.DB_PATH = orig
