"""
Opponent adjustment — scale a rate by the opponent's defense relative to league.

The core supports a per-stat multiplier (ideally the opponent's rate ALLOWED at the
player's position / the league rate). ESPN doesn't expose positional defense cheaply,
so v1 uses a neutral 1.0 on the board and an overall-defense multiplier when a team
defensive rating is supplied (backtest / refinement). Kept as an explicit hook so the
positional version drops in without touching the core.
"""

from __future__ import annotations

# Stats meaningfully swung by opponent defense (scoring/creation); others ~neutral.
_DEFENSE_SENSITIVE = {"pts", "ast", "3pm"}


def opponent_adjust(stat: str, opp_def_rtg: float | None = None,
                    league_avg_def: float | None = None) -> float:
    if not opp_def_rtg or not league_avg_def or stat not in _DEFENSE_SENSITIVE:
        return 1.0
    # higher def_rtg = more points allowed = weaker D → boost; clamp to ±15%.
    return max(0.85, min(1.15, opp_def_rtg / league_avg_def))
