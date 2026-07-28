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
