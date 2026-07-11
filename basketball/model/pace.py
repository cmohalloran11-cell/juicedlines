"""
Pace / possessions for the matchup. Pace is a game-level factor (same for both
teams in a game), so it shifts everyone's counting stats together.

v1 board path uses the league baseline (fast, no extra fetches). When team paces
are supplied (backtest / refinement) the matchup pace is their average. Pace
uncertainty is injected as variance in the simulator (`pace_sd_frac`), which is
what keeps Summer-League totals from coming in systematically low.
"""

from __future__ import annotations


def matchup_pace(league_pace: float, team_a_pace: float | None = None,
                 team_b_pace: float | None = None) -> float:
    ps = [p for p in (team_a_pace, team_b_pace) if p]
    return round(sum(ps) / len(ps), 1) if ps else league_pace
