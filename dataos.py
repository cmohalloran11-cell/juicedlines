"""
dataos.py — data freshness & quality scoring (spec Vol IV Ch 74 / Vol VI Ch 226-227).

The spec's DataOS requires that every number carry a data timestamp and a quality signal.
This module scores the CURRENT board snapshot: how fresh it is, how many sources and sports
are represented, and how much of it actually carries a model projection. It reads the
in-memory cache passed in — no I/O of its own — so it's cheap enough to call per request.

The composite quality score (0–100) is a transparent weighted blend, documented inline.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional


def _age_seconds(updated_at: Optional[str]) -> Optional[float]:
    if not updated_at:
        return None
    try:
        ts = datetime.fromisoformat(updated_at)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - ts).total_seconds())
    except Exception:
        return None


def _freshness_score(age_s: Optional[float], target_s: float = 300.0) -> float:
    """1.0 when just refreshed, decaying to 0 by ~4× the target refresh interval."""
    if age_s is None:
        return 0.0
    if age_s <= target_s:
        return 1.0
    return max(0.0, 1.0 - (age_s - target_s) / (3.0 * target_s))


def health(lines: list[dict[str, Any]], updated_at: Optional[str],
           errors: Optional[dict] = None) -> dict:
    """
    Score the current snapshot. `lines` is the board cache, `updated_at` its timestamp,
    `errors` the per-source error map (so a silently-failing source drags quality down).
    """
    total = len(lines)
    by_source: dict[str, int] = {}
    by_sport: dict[str, int] = {}
    projected = 0
    for l in lines:
        by_source[l.get("source", "unknown")] = by_source.get(l.get("source", "unknown"), 0) + 1
        by_sport[l.get("sport", "other")] = by_sport.get(l.get("sport", "other"), 0) + 1
        if l.get("model_proj") is not None:
            projected += 1

    age_s = _age_seconds(updated_at)
    coverage_ratio = (projected / total) if total else 0.0
    fresh = _freshness_score(age_s)
    n_sources = len([s for s, c in by_source.items() if c > 0 and s != "unknown"])
    source_factor = min(1.0, n_sources / 2.0)          # 2+ live sources = full marks
    errs = errors or {}
    error_penalty = 1.0 if not errs else max(0.3, 1.0 - 0.35 * len(errs))

    # Composite: freshness and projection-coverage matter most; source breadth and a clean
    # error map round it out. Weights are explicit on purpose (transparency).
    quality = 100.0 * (
        0.40 * fresh
        + 0.30 * coverage_ratio
        + 0.15 * source_factor
        + 0.15 * error_penalty
    )

    return {
        "quality_score": int(round(max(0.0, min(100.0, quality)))),
        "updated_at": updated_at,
        "age_seconds": None if age_s is None else int(age_s),
        "freshness": round(fresh, 3),
        "total_lines": total,
        "projected_lines": projected,
        "projection_coverage": round(coverage_ratio, 3),
        "by_source": by_source,
        "by_sport": by_sport,
        "live_sources": n_sources,
        "source_errors": errs,
        "components": {  # show the math behind the score
            "freshness_w0.40": round(fresh, 3),
            "coverage_w0.30": round(coverage_ratio, 3),
            "source_breadth_w0.15": round(source_factor, 3),
            "error_clean_w0.15": round(error_penalty, 3),
        },
    }


# ─────────────────────────── direction-invariant validation ──────────────────
# 2026-07-29 Over/Under bias audit, revised 2026-07-29 projection-realism pass:
#   Median > Line  ⇒  P(Over) > 50%
#   Median < Line  ⇒  P(Over) < 50%
# Checked against MEDIAN, not the displayed (mean-based) Projection. The mean is
# deliberately informative and CAN legitimately diverge from the line-vs-probability
# direction on a skewed stat (e.g. a bench hitter's TB mean is 0.6 while the median — and
# therefore P(over 0.5)'s direction — is 0). Median is reported by every engine from the
# EXACT SAME sample array model_prob is computed from (projector_bridge.py, basketball/
# board.py, tennis/board.py, analytics.py's MLB fallback), which mathematically guarantees
# this holds — see tests/test_engine_characterization.py's
# test_projection_direction_always_agrees_with_prob_over. Runtime safety net: should find
# ~0 violations; any it finds are real bugs to investigate, not a tolerance to widen.

def validate_direction(lines: list[dict[str, Any]], reject: bool = False) -> list[dict[str, Any]]:
    """Return every line that violates the Median/Probability direction invariant, with
    enough context (player/stat/line/projection/median/prob/source) to investigate without
    re-deriving anything. Empty list = a clean board. Falls back to `model_proj` for any
    line that has no `model_median` (shouldn't happen on a current build; defensive only).

    reject=True mutates the offending line dicts in place — clears model_proj/model_prob/
    model_edge and sets direction_rejected=True — so downstream filters (dashboard._projected(),
    which already requires model_proj/model_prob to be non-None) exclude them from the board
    automatically. "Reject any projection that violates this rule" — this IS the rejection."""
    bad = []
    for l in lines:
        proj, line, prob = l.get("model_proj"), l.get("line"), l.get("model_prob")
        med = l.get("model_median")
        check = med if med is not None else proj
        if check is None or line is None or prob is None:
            continue
        violated = (check > line and prob < 0.5) or (check < line and prob > 0.5)
        if violated:
            bad.append({
                "id": l.get("id"), "player": l.get("player"), "sport": l.get("sport"),
                "stat": l.get("stat_type"), "source": l.get("source"),
                "proj_kind": l.get("proj_kind"), "line": line, "projection": proj,
                "median": med, "prob_over": prob,
            })
            if reject:
                l["model_proj"] = None
                l["model_prob"] = None
                l["model_edge"] = None
                l["model_median"] = None
                l["direction_rejected"] = True
    return bad


def direction_report(lines: list[dict[str, Any]], reject: bool = False) -> dict[str, Any]:
    """Diagnostics deliverable for the Over/Under bias audit: violation count, a root-cause
    breakdown by proj_kind/sport (so a regression shows up as a specific engine, not a vague
    number), and the current Over/Under distribution of the exported board. Computes the
    distribution BEFORE any rejection so it reflects what the board actually contained."""
    checked = [l for l in lines
               if l.get("model_proj") is not None and l.get("line") is not None
               and l.get("model_prob") is not None]
    over = sum(1 for l in checked if (l.get("model_prob") or 0) >= 0.5)
    under = len(checked) - over
    bad = validate_direction(lines, reject=reject)
    by_kind: dict[str, int] = {}
    for b in bad:
        k = f"{b.get('sport')}/{b.get('proj_kind')}"
        by_kind[k] = by_kind.get(k, 0) + 1
    return {
        "checked": len(checked),
        "violations": len(bad),
        "violation_rate": round(len(bad) / len(checked), 4) if checked else None,
        "violations_by_sport_and_kind": by_kind,
        "sample_violations": bad[:15],
        "distribution": {
            "over": over, "under": under,
            "pct_over": round(100.0 * over / len(checked), 1) if checked else None,
            "pct_under": round(100.0 * under / len(checked), 1) if checked else None,
        },
    }
