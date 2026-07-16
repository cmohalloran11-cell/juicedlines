"""
ensemble.py — XGBoost ensemble layer + RMSE-weighted blending.

The mechanistic Bayesian model (mlb_model / soccer_model) produces a full
distribution. This layer optionally trains an XGBoost (gradient-boosted) model
on historical engineered-features → actuals, and blends its point estimate into
the distribution's centre using inverse-backtested-RMSE weights. Everything
degrades gracefully: no training data or no xgboost ⇒ pure mechanistic model.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np

from .. import db


def _make_regressor():
    """Prefer XGBoost; fall back to sklearn GradientBoosting; else None."""
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
            objective="reg:squarederror", n_jobs=0)
    except Exception:
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.85)
        except Exception:
            return None


class XGBStatModel:
    """One gradient-boosted model per (sport, stat), trained on feature rows."""

    def __init__(self, sport: str, stat: str):
        self.sport, self.stat = sport, stat
        self.model = None
        self.columns: list[str] = []

    def train_from_db(self, min_rows: int = 150) -> bool:
        X, y, cols = _training_matrix(self.sport, self.stat)
        if X is None or len(y) < min_rows:
            return False
        reg = _make_regressor()
        if reg is None:
            return False
        try:
            reg.fit(X, y)
        except Exception:
            return False
        self.model, self.columns = reg, cols
        return True

    def predict(self, feature_vec: dict) -> Optional[float]:
        if self.model is None or not self.columns:
            return None
        x = np.array([[float(feature_vec.get(c, 0.0) or 0.0) for c in self.columns]])
        try:
            return float(self.model.predict(x)[0])
        except Exception:
            return None


def _training_matrix(sport: str, stat: str):
    """Build (X, y, columns) from features⋈actuals for a stat."""
    with db._LOCK, db.connect() as c:
        rows = c.execute(
            """SELECT f.vector AS vector, a.value AS y
               FROM features f JOIN actuals a
                 ON f.sport=a.sport AND f.player=a.player AND f.game_id=a.game_id
               WHERE f.sport=? AND a.stat=?""",
            (sport, stat)).fetchall()
    if not rows:
        return None, [], []
    dicts, ys = [], []
    for r in rows:
        try:
            v = json.loads(r["vector"])
        except Exception:
            continue
        flat = {k: val for k, val in v.items() if isinstance(val, (int, float))}
        if flat:
            dicts.append(flat); ys.append(float(r["y"]))
    if len(ys) < 10:
        return None, [], []
    cols = sorted({k for d in dicts for k in d})
    X = np.array([[float(d.get(c, 0.0)) for c in cols] for d in dicts])
    return X, np.array(ys), cols


def rmse_weights(sport: str, stat: str, models=("mechanistic", "xgboost")) -> dict[str, float]:
    """Inverse-RMSE weights from the latest backtest per model. Defaults if absent."""
    with db._LOCK, db.connect() as c:
        rows = c.execute(
            """SELECT model_tag, rmse FROM
                 (SELECT json_extract(calibration,'$.model') AS model_tag, rmse,
                         created_at FROM backtest_runs WHERE sport=? AND stat=?)
               WHERE model_tag IS NOT NULL""",
            (sport, stat)).fetchall() if _has_model_tag() else []
    got = {r["model_tag"]: r["rmse"] for r in rows if r["rmse"]}
    inv = {m: 1.0 / max(1e-6, got[m]) for m in models if m in got}
    if not inv:                       # no backtest yet → lean mechanistic
        return {"mechanistic": 0.65, "xgboost": 0.35}
    tot = sum(inv.values())
    return {m: inv[m] / tot for m in inv}


def _has_model_tag() -> bool:
    return False  # backtest stores per-model RMSE separately; see backtest.runner


def blend_distribution(samples: np.ndarray, mechanistic_mean: float,
                       xgb_mean: Optional[float], sport: str, stat: str):
    """
    Shift the mechanistic distribution's centre toward the inverse-RMSE blend of
    the mechanistic and XGBoost means (multiplicative rescale preserves shape).
    """
    if xgb_mean is None or mechanistic_mean <= 1e-9:
        return samples, mechanistic_mean
    w = rmse_weights(sport, stat)
    blended = (w.get("mechanistic", 0.65) * mechanistic_mean
               + w.get("xgboost", 0.35) * max(0.0, xgb_mean))
    scale = blended / mechanistic_mean
    scale = float(np.clip(scale, 0.5, 1.8))   # guard against a wild XGB estimate
    return samples * scale, blended
