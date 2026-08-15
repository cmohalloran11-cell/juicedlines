"""
Opponent matchup — real per-play defensive efficiency, reliability-shrunk.

Not "fantasy points allowed": that number is dominated by how many plays a defense has faced
and by which offenses happened to be on its schedule, so a slow, good defense and a fast, bad
one can post the same figure. This uses per-play rates the defense actually controls — yards
allowed per pass attempt, per carry, completion rate allowed, sack rate — each scaled against
the league mean measured over the same window.

The critical part is that the adjustment is SHRUNK BY MEASURED RELIABILITY rather than applied
at face value. Split-half within a season (weeks 1-9 vs 10-18, 128 team-seasons, 2022-2025)
gives r=0.168 for yards allowed per pass attempt and r=0.318 for yards per carry, i.e.
Spearman-Brown full-season reliabilities of 0.287 and 0.483. Most of a defense's apparent
per-play profile is noise, so a defense sitting 10% above league average moves the projection
by ~2.9% on passing and ~4.8% on rushing, not by 10%. A pass-rush edge is carried by the sack
rate the same way rather than by a separate hand-set "pass-rush matchup" factor.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from ..config import cfg

_RATE_TERMS = {
    "yards_per_pass_att": ("pass_yards", "pass_attempts"),
    "yards_per_carry": ("rush_yards", "carries"),
    "completion_rate": ("completions", "pass_attempts"),
    "sack_rate": ("sacks", "dropbacks"),
}

# Which of the model's own efficiency/opportunity rates each defensive rate scales.
_APPLIES_TO = {
    "yards_per_pass_att": ("yards_per_attempt", "yards_per_target"),
    "yards_per_carry": ("yards_per_carry",),
    "completion_rate": ("completion_rate", "catch_rate"),
}


@dataclass
class DefenseProfile:
    team: str
    rates: dict                 # defensive rate -> allowed value
    plays: dict                 # defensive rate -> denominator behind it
    games: int = 0


def fit_defense(weeks) -> dict[str, DefenseProfile]:
    """{defending team: profile} from every offensive player-week, attributed to the OPPONENT
    of the player who produced it. Uses whatever `weeks` it is given, so a caller can scope it
    to a season, a window, or (for a date-strict backtest) to games before a cutoff."""
    acc: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    games: dict[str, set] = defaultdict(set)
    for w in weeks:
        if not w.opponent:
            continue
        d = acc[w.opponent]
        d["pass_yards"] += w.pass_yards
        d["pass_attempts"] += w.pass_attempts
        d["rush_yards"] += w.rush_yards
        d["carries"] += w.carries
        d["completions"] += w.completions
        d["sacks"] += w.sacks
        d["dropbacks"] += w.pass_attempts + w.sacks
        games[w.opponent].add(w.game_id)
    out = {}
    for team, d in acc.items():
        rates, plays = {}, {}
        for rate, (num, den) in _RATE_TERMS.items():
            if d[den] > 0:
                rates[rate] = d[num] / d[den]
                plays[rate] = d[den]
        out[team] = DefenseProfile(team=team, rates=rates, plays=plays, games=len(games[team]))
    return out


def matchup_multipliers(profile: Optional[DefenseProfile]) -> dict[str, float]:
    """{model rate: multiplier} for one opponent. {} — a completely neutral matchup — when the
    opponent is unknown or has too little history to say anything, which is exactly the state
    every team is in during preseason and week 1."""
    if profile is None:
        return {}
    league = cfg("defense", "league", default={})
    reliability = cfg("defense", "reliability", default={})
    clamp = cfg("defense", "clamp", default={})
    min_plays = float(cfg("defense", "min_plays", default=100))

    out: dict[str, float] = {}
    for rate, targets in _APPLIES_TO.items():
        observed = profile.rates.get(rate)
        lg = league.get(rate)
        if observed is None or not lg or profile.plays.get(rate, 0) < min_plays:
            continue
        raw = observed / lg
        rel = float(reliability.get(rate, 0.3))
        adjusted = 1.0 + rel * (raw - 1.0)
        lo, hi = clamp.get(rate, (0.8, 1.25))
        adjusted = max(lo, min(hi, adjusted))
        for t in targets:
            out[t] = round(adjusted, 4)
    return out


def pass_rush_pressure(profile: Optional[DefenseProfile]) -> Optional[float]:
    """The opponent's sack rate relative to league, reliability-shrunk and clamped — the real
    pass-rush matchup signal. None when unmeasurable. Reported for reasoning/confidence; it is
    not multiplied into passing yards separately, because the yards-per-attempt multiplier
    above already carries the yardage consequence of pressure and applying both would
    double-count the same effect."""
    if profile is None:
        return None
    observed = profile.rates.get("sack_rate")
    lg = cfg("defense", "league", "sack_rate")
    if observed is None or not lg or profile.plays.get("sack_rate", 0) < float(cfg("defense", "min_plays", default=100)):
        return None
    rel = float(cfg("defense", "reliability", "sack_rate", default=0.287))
    lo, hi = cfg("defense", "clamp", "sack_rate", default=(0.48, 1.62))
    return round(max(lo, min(hi, 1.0 + rel * (observed / lg - 1.0))), 4)
