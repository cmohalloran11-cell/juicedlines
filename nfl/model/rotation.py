"""
Preseason rotation model — a DIFFERENT model, not the regular-season one turned down.

The regular-season playing-time engine answers "what share of his team's snaps does this
player normally take". In August that question has no answer, because preseason snaps aren't
distributed by role at all — they're distributed by a coach's rotation plan. A starter who
takes 85% of regular-season snaps might take a series and a half; a third-string receiver
who would never see the field in September plays the entire second half. So the model is
rebuilt around ROTATION TIERS rather than around the player's own snap history:

  confirmed_starter  depth-chart rank 1 AND a rank-1 workload actually observed last season
  expected_starter   depth-chart rank 1, no observed workload to confirm it
  second_team        depth-chart rank 2
  third_team         depth-chart rank 3+
  fringe             on the roster, no depth-chart entry
  unknown            not resolvable in any feed

Each tier gets its own playing-time distribution SHAPE from config.preseason_tiers — tight
for a confirmed starter (short, predictable outing), very wide for fringe/unknown (anywhere
from a cameo to a full half). That mapping is config, not code, so it is tunable without
touching this module.

HONESTY NOTE, and it is the important one: the tier -> snap-share LEVELS in config are
ASSUMPTIONS, not measurements. No free data source publishes preseason snap counts — verified
live 2026-08-15, the nflverse snap_counts release has zero rows with game_type PRE and the
schedules release has zero preseason games. `measure_tier_shares` below is the real
measurement pipeline for that table; the day a preseason snap source is wired, run it and
replace the assumed block with its output. Until then every preseason projection carries
playing_time_confidence "low", which is what makes board.py defer heavily to the market line.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import cfg
from .playing_time import PlayingTime, _concentration_from_moments

TIERS = ("confirmed_starter", "expected_starter", "second_team", "third_team",
         "fringe", "unknown")

_TIER_ROLE = {
    "confirmed_starter": "starter", "expected_starter": "starter",
    "second_team": "rotation", "third_team": "backup",
    "fringe": "fringe", "unknown": "unknown",
}


@dataclass
class TeamTendency:
    """A team's preseason usage pattern.

    Only `team`, `coach` and the two pass/rush rates can be filled from data this repo has
    access to; the rest are None with a stated reason rather than a plausible-looking number
    (see `reasons`). `basis` records where each filled number actually came from — in
    particular the pass/rush rates are that team's REGULAR-SEASON rates, an explicit,
    labelled substitute for a preseason rate nobody publishes, not a preseason measurement.
    """
    team: str
    coach: Optional[str] = None
    starter_snap_rate: Optional[float] = None
    avg_first_team_drives: Optional[float] = None
    qb_rotation_pattern: Optional[str] = None
    second_team_usage: Optional[float] = None
    preseason_pass_rate: Optional[float] = None
    preseason_rush_rate: Optional[float] = None
    basis: dict = None
    reasons: dict = None

    def __post_init__(self):
        if self.basis is None:
            self.basis = {}
        if self.reasons is None:
            self.reasons = {}


_UNAVAILABLE_REASON = ("no free source publishes preseason snap counts or drive charts — "
                       "nflverse snap_counts has 0 PRE rows and schedules has 0 preseason "
                       "games (verified 2026-08-15)")


def _pass_rush_rates(weeks) -> tuple[Optional[float], Optional[float], int]:
    att = sum(w.pass_attempts + w.sacks for w in weeks)
    rush = sum(w.carries for w in weeks)
    plays = att + rush
    if plays < 100:
        return None, None, int(plays)
    return round(att / plays, 4), round(rush / plays, 4), int(plays)


def league_tendency_prior(weeks) -> TeamTendency:
    """League-level fallback, aggregated from REAL data across every team in `weeks`. Used
    whenever a team's own sample is too thin (a team with a new coach and no games yet)."""
    p, r, plays = _pass_rush_rates(weeks)
    t = TeamTendency(team="LEAGUE", preseason_pass_rate=p, preseason_rush_rate=r)
    for f in cfg("team_tendency", "unavailable", default=()):
        t.reasons[f] = _UNAVAILABLE_REASON
    if p is not None:
        t.basis["preseason_pass_rate"] = f"regular_season_measured (n={plays} plays)"
        t.basis["preseason_rush_rate"] = f"regular_season_measured (n={plays} plays)"
    else:
        t.reasons["preseason_pass_rate"] = "fewer than 100 offensive plays of history"
    return t


def team_tendency(team: str, weeks, schedule=None) -> TeamTendency:
    """One team's tendency, falling back per-field to the league prior when its own sample is
    thin. `coach` is real — it comes straight off the schedules release's home_coach/away_coach
    for a game this team plays in."""
    own = [w for w in weeks if w.team == team]
    p, r, plays = _pass_rush_rates(own)
    t = TeamTendency(team=team, preseason_pass_rate=p, preseason_rush_rate=r)
    if p is not None:
        t.basis["preseason_pass_rate"] = f"regular_season_measured (n={plays} plays, own team)"
        t.basis["preseason_rush_rate"] = f"regular_season_measured (n={plays} plays, own team)"
    else:
        lg = league_tendency_prior(weeks)
        t.preseason_pass_rate, t.preseason_rush_rate = lg.preseason_pass_rate, lg.preseason_rush_rate
        t.basis["preseason_pass_rate"] = "league_prior (own sample under 100 plays)"
        t.basis["preseason_rush_rate"] = "league_prior (own sample under 100 plays)"
    for g in (schedule or []):
        if g.home_team == team and g.home_coach:
            t.coach = g.home_coach
            break
        if g.away_team == team and g.away_coach:
            t.coach = g.away_coach
            break
    if t.coach:
        t.basis["coach"] = "schedules release"
    for f in cfg("team_tendency", "unavailable", default=()):
        t.reasons[f] = _UNAVAILABLE_REASON
    return t


def classify_tier(depth_rank: Optional[int], prior_snap_share: Optional[float],
                  n_prior_games: int = 0, on_roster: bool = False) -> tuple[str, str]:
    """(tier, plain-language reason). `prior_snap_share` is the player's REGULAR-season snap
    share from the most recent completed season — the only real workload evidence that exists
    in August — and only counts as confirmation with enough games behind it."""
    conf_share = float(cfg("preseason_tiers", "confirmed_snap_share", default=0.60))
    min_games = int(cfg("preseason_tiers", "min_games_for_confirmation", default=4))
    if depth_rank == 1:
        if (prior_snap_share is not None and prior_snap_share >= conf_share
                and n_prior_games >= min_games):
            return ("confirmed_starter",
                    f"listed first on the depth chart and took a {prior_snap_share*100:.0f}% "
                    f"snap share over his last {n_prior_games} regular-season games")
        return ("expected_starter",
                "listed first on the depth chart, but no confirming snap-share history")
    if depth_rank == 2:
        return ("second_team", "listed second on the depth chart")
    if depth_rank is not None and depth_rank >= 3:
        return ("third_team", f"listed {depth_rank}th on the depth chart")
    if on_roster:
        return ("fringe", "on the roster with no depth-chart entry")
    return ("unknown", "not found on a depth chart or roster")


def tier_playing_time(tier: str, expected_team_snaps: Optional[float] = None,
                      depth_rank: Optional[int] = None) -> PlayingTime:
    """The tier's playing-time distribution. basis is always "preseason_tier" and confidence
    always "low": see the module docstring — the tier LEVELS are assumptions, and a projection
    built on an assumption must not advertise itself as well-evidenced."""
    shape = cfg("preseason_tiers", "snap_share", default={}).get(tier)
    if shape is None:
        shape = cfg("preseason_tiers", "snap_share", default={}).get("unknown", (0.25, 0.25))
    mean, sd = float(shape[0]), float(shape[1])
    conc = _concentration_from_moments(mean, sd)
    return PlayingTime(
        expected_snap_share=round(mean, 4),
        expected_snaps=round(mean * expected_team_snaps, 1) if expected_team_snaps else None,
        concentration=round(conc, 3), n_games=0, basis="preseason_tier",
        role=_TIER_ROLE.get(tier, "unknown"), depth_rank=depth_rank,
        playing_time_probability=None, confidence="low")


def preseason_risk(tier: str, playing_time: PlayingTime) -> float:
    """0-1: how much of this projection is rotation guesswork rather than known role. Built
    from the two things that actually drive it — the tier's own certainty and the width of its
    playing-time distribution (a wide Beta is literally 'we don't know how long he plays')."""
    tier_risk = {"confirmed_starter": 0.35, "expected_starter": 0.5, "second_team": 0.65,
                 "third_team": 0.8, "fringe": 0.9, "unknown": 1.0}.get(tier, 1.0)
    a, b = playing_time.beta_params()
    total = a + b
    spread = float(np.sqrt(a * b / (total * total * (total + 1.0)))) if total > 0 else 0.5
    # A snap-share sd of 0.25 is the widest tier in config; scale against that.
    return round(min(1.0, 0.6 * tier_risk + 0.4 * min(1.0, spread / 0.25)), 3)


def measure_tier_shares(snap_rows, depth_entries, season_type: str = "PRE") -> dict:
    """THE measurement pipeline for config.preseason_tiers.snap_share.

    Give it snap rows whose season_type is `season_type` plus the depth-chart entries for the
    same season and it returns {tier: (mean, sd, n)} measured off real games. Run against
    REG rows it reproduces config.depth_rank_snap_share (that block WAS produced this way);
    run against PRE rows it produces the preseason table that does not exist yet, at which
    point the assumed block in config.py can be replaced with a measured one.

    Returns {} when no rows of that season_type exist — which is exactly what happens today
    for PRE, and is the honest answer rather than a fabricated table.
    """
    rank: dict[tuple, int] = {}
    for d in depth_entries:
        if d.rank is None:
            continue
        key = (d.team, d.player.strip().lower())
        if key not in rank or d.rank < rank[key]:
            rank[key] = d.rank
    buckets: dict[str, list] = defaultdict(list)
    for s in snap_rows:
        if s.season_type != season_type or s.offense_pct is None:
            continue
        r = rank.get((s.team, s.player.strip().lower()))
        if r is None:
            buckets["fringe"].append(float(s.offense_pct))
        elif r == 1:
            buckets["expected_starter"].append(float(s.offense_pct))
        elif r == 2:
            buckets["second_team"].append(float(s.offense_pct))
        else:
            buckets["third_team"].append(float(s.offense_pct))
    return {t: (round(float(np.mean(v)), 4), round(float(np.std(v)), 4), len(v))
            for t, v in buckets.items() if len(v) >= 30}
