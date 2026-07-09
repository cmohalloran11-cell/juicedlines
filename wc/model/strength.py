"""
Team attack/defense ratings, normalized to the league average.

Blends goals with xG (xG is the more stable, predictive signal) and expresses each
team relative to the tournament baseline, so a fixture's goal expectancy is just
`league_avg × attack × opponent_defense`. Recency-weighting of recent vs older form
is applied upstream in the data adapter (the source returns already-weighted rates;
`recency_decay` documents the intended split); here we combine goals and xG.
"""

from __future__ import annotations

from ..config import load
from ..data.base import TeamStrength

_XG_WEIGHT = 0.6      # weight on xG vs actual goals when both exist


def _att_basis(s: TeamStrength) -> float:
    return _XG_WEIGHT * s.xg_pg + (1 - _XG_WEIGHT) * s.gf_pg if s.xg_pg > 0 else s.gf_pg


def _def_basis(s: TeamStrength) -> float:
    return _XG_WEIGHT * s.xga_pg + (1 - _XG_WEIGHT) * s.ga_pg if s.xga_pg > 0 else s.ga_pg


def ratings(strengths: list[TeamStrength]) -> dict[str, dict]:
    """team → {attack, defense, shots_pg, possession}. attack>1 = above-average
    offense; defense>1 = leakier-than-average (concedes more)."""
    lg = load()["model"]["league_avg_goals"]
    out: dict[str, dict] = {}
    for s in strengths:
        out[s.team] = {
            "attack": round(_att_basis(s) / lg, 3),
            "defense": round(_def_basis(s) / lg, 3),
            "shots_pg": s.shots_pg,
            "possession": s.possession,
        }
    return out
