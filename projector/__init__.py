"""stat-projector — single-stat player projection system for MLB + World Cup.

VENDORED SUBSET. The full project lives beside this repo (../stat-projector) and is the
source of truth; this is the minimal slice JUICED's board needs so the DEPLOY build — which
only checks out juicedlines — can run the real engine instead of silently falling back to the
empirical median. Before this was vendored, 100% of deployed MLB props were `proj_kind:"model"`
(the fallback), never `"engine"`.

Contents = exactly what `projector_bridge` imports, transitively:
    config.py                     (stdlib only; YAML is lazy + optional)
    db.py                         (stdlib sqlite3; only touched by ensemble.rmse_weights,
                                   which is never reached when xgboost is absent)
    models/{base,montecarlo,ensemble,mlb_model}.py
    features/mlb_features.py

Deps: numpy + scipy only. xgboost/sklearn are LAZY imports inside ensemble — without them
`blend_distribution` returns early, so the board runs the pure mechanistic Bayesian →
Monte-Carlo core (the validated path). data/, backtest/, cli, output and the soccer models
are intentionally NOT vendored (they're what pull in xgboost/pybaseball/pandas).

To re-sync after changing the real project, re-copy those files.
"""
__version__ = "0.1.0"
