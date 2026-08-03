"""
model_health.py — public transparency + backtesting surface (spec Vol V Part 4, Ch 186-187).

Aggregates the graded CLV ledger's calibration stack (already computed in db.py) into a
single honest report: how the model has actually performed vs the close, how calibrated its
probabilities and intervals are, and where it earns trust per stat. This is the "historical
accuracy tracking" pillar — the thing that separates Juiced from a pick generator.

Nothing here fabricates a metric; every number is read from graded outcomes. When the ledger
is too thin for a metric, that field is null with a plain-language note rather than a guess.
"""
from __future__ import annotations

from typing import Any, Optional

import db
import provenance
import backtest

_SPORTS = ("MLB", "WNBA", "Tennis")

# Rough attribution of what drives a projection — the "Model Confidence Breakdown" donut.
# These reflect the engine's factor emphasis (recent form is the backbone; matchup/usage
# adjust it; injuries/weather are situational). Illustrative weights, shown transparently.
FACTOR_WEIGHTS = [
    {"factor": "Recent Form", "weight": 30},
    {"factor": "Matchup", "weight": 25},
    {"factor": "Usage / Role", "weight": 20},
    {"factor": "Injuries", "weight": 10},
    {"factor": "Weather / Park", "weight": 5},
    {"factor": "Other", "weight": 10},
]


def _interpret_prob_cal(cal: dict) -> Optional[str]:
    a = cal.get("a")
    if a is None:
        return None
    if a < 0.9:
        return f"overconfident (Platt slope {a}<1 → probabilities shrunk toward 50%)"
    if a > 1.1:
        return f"underconfident (Platt slope {a}>1)"
    return f"well-calibrated (Platt slope {a}≈1)"


def _sport_health(sport: str) -> dict:
    card = db.scorecard(sport)
    gammas = db.stat_gammas(sport)
    prob_cal = db.prob_calibration(sport)
    width = db.interval_width(sport)
    biases = db.stat_biases(sport)

    # Highest- and lowest-trust stats (γ = how much the model beats the line, per stat).
    ranked = sorted(gammas.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "sport": sport,
        "graded_props": card.get("graded"),
        "decided": card.get("decided"),
        "hit_rate_vs_close": card.get("hit_rate"),
        "breakeven": card.get("breakeven"),
        "beats_breakeven": (
            None if card.get("hit_rate") is None
            else card["hit_rate"] > card["breakeven"]),
        "plays": card.get("plays"),
        "play_hit_rate": card.get("play_hit_rate"),
        "clv": {
            "n": card.get("clv_n"),
            "positive_pct": card.get("clv_positive_pct"),
            "avg_move": card.get("clv_avg_move"),
        },
        "ab_test_enhanced_vs_plain": {
            "n": card.get("ab_n"),
            "plain": card.get("ab_plain"),
            "enhanced": card.get("ab_enhanced"),
        },
        "probability_calibration": {
            "platt": prob_cal or None,
            "interpretation": _interpret_prob_cal(prob_cal) if prob_cal else
            "not enough graded rows to calibrate — probabilities ship as-is",
        },
        "interval_widening_factor": width,
        "interval_note": (
            "1.0 = intervals ship un-stretched" if width == 1.0 else
            f"p10–p90 band stretched ×{width} to reach honest 80% coverage"),
        "top_trust_stats": [{"stat": s, "gamma": g} for s, g in ranked[:5]],
        "low_trust_stats": [{"stat": s, "gamma": g} for s, g in ranked[-5:]] if ranked else [],
        "systematic_biases": biases or {},
    }


def health(sport: Optional[str] = None) -> dict:
    """Public model-health report. One sport, or all when sport is None."""
    sports = [sport] if sport else list(_SPORTS)
    reports = {}
    for s in sports:
        try:
            reports[s] = _sport_health(s)
        except Exception as exc:  # a sport with no ledger rows shouldn't break the whole report
            reports[s] = {"sport": s, "error": str(exc)}
    return {
        "versions": provenance.versions(),
        "generated_at": provenance.now_iso(),
        "sports": reports,
        "disclaimer": (
            "Projections are transparent statistical estimates, not guarantees. Metrics are "
            "measured from graded historical outcomes vs the closing line."),
    }


def calibration_detail(sport: str = "MLB") -> dict:
    """Full calibration stack for one sport — the AdminOS calibration dashboard (Ch 342-343).
    Same underlying data as the public report but with every per-stat number, unsummarized."""
    return {
        "sport": sport,
        "per_stat_trust_gamma": db.stat_gammas(sport),
        "per_stat_bias": db.stat_biases(sport),
        "probability_platt": db.prob_calibration(sport),
        "interval_widening_factor": db.interval_width(sport),
        "scorecard": db.scorecard(sport),
        "diagnostics": backtest.diagnostics(sport),
        "drift": backtest.drift(sport),
        "accuracy_by_model_version": backtest.version_history(sport),
    }


# ── Model Health dashboard (self-monitoring surface) ───────────────────────────
# Everything the dashboard needs to answer, per sport: is it improving, is it getting
# worse, which markets/archetypes perform best/worst, which stats are drifting, and what
# changed between model versions. One bundle call per view so the UI isn't stitching
# together 8 separate requests.

def dashboard_summary() -> dict:
    """Lightweight overview across all sports — the landing view of the Model Health
    dashboard. Top-line accuracy + drift status only; call dashboard_detail(sport) for
    the full per-sport breakdown."""
    out = {}
    for sp in _SPORTS:
        try:
            acc = backtest.current_accuracy(sp)
            drift = backtest.drift(sp)
            out[sp] = {
                "sport": sp,
                "model_version": provenance.model_version(sp),
                "n": acc["overall"].get("n", 0),
                "mae": acc["overall"].get("mae"),
                "rmse": acc["overall"].get("rmse"),
                "bias": acc["overall"].get("bias"),
                "hit_rate": acc["overall"].get("hit_rate"),
                "brier": acc["overall"].get("brier"),
                "ece": acc["overall"].get("ece"),
                "coverage": acc["overall"].get("coverage"),
                "drift_status": drift.get("status"),
            }
        except Exception as exc:
            out[sp] = {"sport": sp, "error": str(exc)}
    return {"generated_at": provenance.now_iso(), "sports": out}


def dashboard_detail(sport: str = "MLB") -> dict:
    """The full self-monitoring bundle for one sport: current accuracy (overall +
    per-stat), reliability diagram, interval coverage, drift (sport-wide + per-stat),
    accuracy by player-sample-depth (archetype), accuracy by model version, the version
    changelog (what changed, in plain language), and the recorded daily-snapshot history
    (backtest.record_run, once/day per sport — see main.py's maintenance loop) for
    trend charts. Every field answers one of: is it improving? is it getting worse?
    which markets perform best/worst? which archetypes perform worst? which stats are
    drifting? what changed, and did it help?"""
    try:
        accuracy = backtest.current_accuracy(sport)
    except Exception as exc:
        return {"sport": sport, "error": str(exc)}
    return {
        "sport": sport,
        "model_version": provenance.model_version(sport),
        "generated_at": provenance.now_iso(),
        "accuracy": accuracy,
        "reliability_diagram": backtest.reliability_diagram(sport),
        "drift": backtest.drift(sport),
        "drift_by_stat": backtest.drift_by_stat(sport),
        "by_sample_depth": backtest.by_sample_depth(sport),
        "accuracy_by_model_version": backtest.version_history(sport),
        "changelog": provenance.changelog(sport).get(sport, []),
        "snapshot_history": backtest.registry(sport, limit=90),
        "diagnostics": backtest.diagnostics(sport),
    }
