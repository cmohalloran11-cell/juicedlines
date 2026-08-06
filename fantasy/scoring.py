"""
fantasy.scoring — pure scoring engine: turns a raw stat projection into fantasy points under
one specific league's actual scoring rules. No hardcoded PPR/half-PPR/standard branch anywhere
-- every point value comes from the league's own scoring_settings dict (Sleeper's native
format, stat-key -> points-per-unit), matching valuation.py's "pure function of already-
computed fields, no I/O" boundary.
"""
from __future__ import annotations

from typing import Mapping


def score_stats(stats: Mapping[str, float], scoring_settings: Mapping[str, float]) -> float:
    """Fantasy points for one stat line under one league's scoring rules. A stat key the
    league doesn't score is ignored, not an error -- adapters may produce stat keys a given
    league's scoring_settings doesn't mention at all (e.g. pass_cmp in a non-completion
    league)."""
    return round(sum(float(stats.get(k, 0.0)) * float(v) for k, v in scoring_settings.items()
                     if k in stats), 2)


def score_players(projections: list[dict], scoring_settings: Mapping[str, float]) -> list[dict]:
    """Attach `fantasy_points` to a copy of each projection dict (each must already carry
    `stats`), scored under this league's rules. Does not mutate the input list/dicts."""
    scored = []
    for p in projections:
        row = dict(p)
        row["fantasy_points"] = score_stats(p.get("stats", {}), scoring_settings)
        scored.append(row)
    return scored
