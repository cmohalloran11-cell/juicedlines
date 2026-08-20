"""
Preseason rotation model — a DIFFERENT model, not the regular-season one turned down.

The regular-season playing-time engine answers "what share of his team's snaps does this
player normally take". In August that question has no answer, because preseason snaps aren't
distributed by role at all — they're distributed by a coach's rotation plan. A starter who
takes 85% of regular-season snaps might take a series and a half; a third-string receiver
who would never see the field in September plays the entire second half. So the model is
rebuilt around ROTATION TIERS rather than around the player's own snap history:

  TIER 1  confirmed_starter    depth-chart rank 1 AND a rank-1 workload actually observed
  TIER 2  likely_starter       depth-chart rank 1, no confirming workload history
  TIER 3  first_team_rotation  depth-chart rank 2 WITH a real rotational workload observed
  TIER 4  second_team          depth-chart rank 2, no confirming workload history
  TIER 5  third_team           depth-chart rank 3+
  TIER 6  fringe               on the roster, no depth-chart entry
  TIER 7  unknown              not resolvable in any feed

DRIVES FIRST, SNAPS SECOND (the actual fix for the clustering the old model produced): a
player's expected snaps is no longer "tier_mean_share x whatever this game's team-snaps
number happens to be" — the old single-factor version, which meant EVERY preseason game
multiplied by the exact same constant (no preseason schedule/spread ever exists to vary it),
so two tiers landing on the same mean share (confirmed_starter and third_team both sat at
0.30) produced the literal identical expected_snaps for every player in either tier,
regardless of position. Two independent, position/tier-varying factors instead —
`expected_snaps = tier_drives_mean(tier) * position_snaps_per_drive(position)` — structurally
cannot collide that way: a QB and a WR in the same tier get different numbers because they
play different roles within a drive, and two different tiers get different numbers even at
the same position because the model no longer round-trips through one shared table.

HIERARCHY (highest-quality information overrides lower-quality priors):

    player's own PRESEASON history (this season or last)      <- historical_preseason_snap_share
            |  (falls back when absent -- always today, see below)
    depth-chart role -> rotation tier                          <- classify_tier
            |
    team rotation tendency (real signal: team's REGULAR-season pass/rush rate, the one
    thing about preseason offensive identity this data CAN measure)
            |  (falls back when absent -- see team_tendency's own reasons dict)
    position-specific preseason prior (position_snaps_per_drive)
            |
    league preseason prior (tier_drives, position="" fallback)

HONESTY NOTE, and it is the important one: the tier_drives/position_snaps_per_drive LEVELS in
config are ASSUMPTIONS, not measurements. No free data source publishes preseason snap counts
or drive-level data — verified live 2026-08-15, the nflverse snap_counts release has zero rows
with game_type PRE and the schedules release has zero preseason games. `measure_tier_shares`
below is the real measurement pipeline for the old share-based table; the day a preseason
snap/drive source is wired, both tables in config get replaced with measured ones. Until then
every preseason projection carries playing_time_confidence "low", which is what makes
board.py defer heavily to the market line.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import cfg
from .playing_time import PlayingTime, _concentration_from_moments

TIERS = ("confirmed_starter", "likely_starter", "first_team_rotation", "second_team",
        "third_team", "fringe", "unknown")

_TIER_ROLE = {
    "confirmed_starter": "starter", "likely_starter": "starter",
    "first_team_rotation": "rotation", "second_team": "rotation",
    "third_team": "backup", "fringe": "fringe", "unknown": "unknown",
}

# Coarser certainty ranking used by preseason_risk — how much of a tier's uncertainty is
# "we don't know the role" vs "we know the role but not exactly how long he plays". Ordered by
# real role-certainty (a confirmed starter's ROLE is nearly certain even though his exact
# snap count isn't; "unknown" is uncertain on both axes).
_TIER_RISK = {
    "confirmed_starter": 0.30, "likely_starter": 0.42, "first_team_rotation": 0.55,
    "second_team": 0.65, "third_team": 0.80, "fringe": 0.90, "unknown": 1.0,
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
    in August — and only counts as confirmation with enough games behind it.

    Depth-chart rank 2 now splits into two tiers on the SAME real-evidence test rank 1 already
    used (a qualifying prior snap share over enough games): a rank-2 player who was ALSO a
    genuine rotational piece last season (first_team_rotation) is a materially different
    projection than one who is a rank-2 in name only (second_team) — collapsing both into one
    "second_team" tier the way the old 6-tier model did was throwing away real information the
    depth-chart-plus-history join already carries."""
    conf_share = float(cfg("preseason_tiers", "confirmed_snap_share", default=0.60))
    rot_share = float(cfg("preseason_tiers", "first_team_rotation_snap_share", default=0.35))
    min_games_conf = int(cfg("preseason_tiers", "min_games_for_confirmation", default=4))
    min_games_rot = int(cfg("preseason_tiers", "min_games_for_rotation", default=3))
    if depth_rank == 1:
        if (prior_snap_share is not None and prior_snap_share >= conf_share
                and n_prior_games >= min_games_conf):
            return ("confirmed_starter",
                    f"listed first on the depth chart and took a {prior_snap_share*100:.0f}% "
                    f"snap share over his last {n_prior_games} regular-season games")
        return ("likely_starter",
                "listed first on the depth chart, but no confirming snap-share history")
    if depth_rank == 2:
        if (prior_snap_share is not None and prior_snap_share >= rot_share
                and n_prior_games >= min_games_rot):
            return ("first_team_rotation",
                    f"listed second on the depth chart with a real {prior_snap_share*100:.0f}% "
                    f"snap share over his last {n_prior_games} regular-season games — a "
                    f"genuine rotational role, not just a nominal backup slot")
        return ("second_team", "listed second on the depth chart")
    if depth_rank is not None and depth_rank >= 3:
        return ("third_team", f"listed {depth_rank}th on the depth chart")
    if on_roster:
        return ("fringe", "on the roster with no depth-chart entry")
    return ("unknown", "not found on a depth chart or roster")


def historical_preseason_snap_share(pre_snap_sample: Optional[list[float]],
                                    halflife: float = 1.0) -> Optional[tuple[float, int]]:
    """(recency-weighted mean offense_pct, n) from the player's OWN preseason games —
    `pre_snap_sample` is offense_pct per PRESEASON game, most-recent-first, across however many
    prior preseasons are available. Real, tested code (see nfl/tests/test_core.py) — but with
    today's data source it is ALWAYS called with an empty/None sample, because nflverse
    publishes no preseason snap rows at all (season_type is "REG"|"POST" only — see
    data/base.py's PlayerWeek/SnapWeek docstrings). This is the top of the hierarchy: the
    single best signal there is about how a specific coach plays a specific player in August,
    the day any source actually publishes it.
    """
    if not pre_snap_sample:
        return None
    wsum = wval = 0.0
    for i, pct in enumerate(pre_snap_sample):
        w = 0.5 ** (i / halflife) if halflife > 0 else 1.0
        wsum += w
        wval += w * pct
    if wsum <= 0:
        return None
    return (wval / wsum, len(pre_snap_sample))


def team_rotation_nudge(position: str, team_pass_rate: Optional[float]) -> float:
    """Signed fraction to scale a skill position's expected drives by, from the team's REAL
    (regular-season-measured, preseason-labelled-as-substitute — see TeamTendency.basis)
    pass rate. Capped so it is a nudge, never a second global multiplier: at the observed
    range of team pass rates (~0.42-0.62), this moves a projection by at most
    team_pass_rate_nudge_cap in either direction. QB/DL/etc are unaffected — a team throwing
    more doesn't change how long ITS OWN quarterback plays, only how involved its receivers
    and backs are relative to each other."""
    pos = (position or "").upper()
    if team_pass_rate is None or pos not in ("WR", "TE", "RB"):
        return 0.0
    cap = float(cfg("preseason_tiers", "team_pass_rate_nudge_cap", default=0.10))
    lg = float(cfg("preseason_tiers", "team_pass_rate_league_mean", default=0.5673))
    if lg <= 0:
        return 0.0
    delta = (float(team_pass_rate) - lg) / lg
    sign = 1.0 if pos in ("WR", "TE") else -1.0
    return max(-cap, min(cap, sign * delta))


def tier_playing_time(tier: str, expected_team_snaps: Optional[float] = None,
                      depth_rank: Optional[int] = None, position: str = "",
                      team_pass_rate: Optional[float] = None,
                      own_preseason_snap_sample: Optional[list[float]] = None) -> PlayingTime:
    """The tier's playing-time distribution, via `expected_snaps = tier_drives_mean(tier) *
    position_snaps_per_drive(position)`, optionally overridden by the player's own preseason
    history (see historical_preseason_snap_share) when it exists — the hierarchy in this
    module's docstring, implemented. `confidence` is always "low" for a tier-only estimate
    (see the module docstring — the tier LEVELS are assumptions, and a projection built on an
    assumption must not advertise itself as well-evidenced); a player-specific historical
    override reports its own confidence from real sample size instead.
    """
    pos = (position or "").upper()
    spd_table = cfg("preseason_tiers", "position_snaps_per_drive", default={})
    spd = float(spd_table.get(pos, spd_table.get("", 4.0)))
    drives_table = cfg("preseason_tiers", "tier_drives", default={})
    d_mean, d_sd = drives_table.get(tier, drives_table.get("unknown", (3.8, 3.0)))
    d_mean, d_sd = float(d_mean), float(d_sd)

    nudge = team_rotation_nudge(pos, team_pass_rate)
    d_mean *= (1.0 + nudge)

    mean_snaps = d_mean * spd
    # Delta method: Var(snaps) ~= position_snaps_per_drive^2 * Var(drives) — spd is treated as
    # fixed (it's a real per-snap-of-a-drive mechanic, not itself uncertain per player).
    sd_snaps = d_sd * spd

    basis, confidence, n_games = "preseason_tier", "low", 0
    hist = historical_preseason_snap_share(own_preseason_snap_sample)
    if hist is not None:
        hist_mean, n = hist
        denom = expected_team_snaps or 65.0
        prior_share = mean_snaps / denom
        k = 2.0   # pseudo-games of tier-prior weight — same shrinkage form as every other
                  # rate in this codebase; see model/usage.py's module docstring.
        blended_share = (hist_mean * n + prior_share * k) / (n + k)
        mean_snaps = blended_share * denom
        sd_snaps = sd_snaps * (k / (n + k))   # own history narrows the band, not just the mean
        basis, n_games = "preseason_own_history", n
        confidence = "high" if n >= 6 else "medium" if n >= 3 else "low"

    denom = expected_team_snaps or 65.0
    share = min(max(mean_snaps / denom, 1e-3), 1 - 1e-3)
    sd_share = max(sd_snaps / denom, 1e-3)
    conc = _concentration_from_moments(share, sd_share)

    return PlayingTime(
        expected_snap_share=round(share, 4),
        expected_snaps=round(mean_snaps, 1) if expected_team_snaps else None,
        concentration=round(conc, 3), n_games=n_games, basis=basis,
        role=_TIER_ROLE.get(tier, "unknown"), depth_rank=depth_rank,
        playing_time_probability=None, confidence=confidence)


def preseason_risk(tier: str, playing_time: PlayingTime) -> float:
    """0-1: how much of this projection is rotation guesswork rather than known role. Built
    from the two things that actually drive it — the tier's own certainty (_TIER_RISK) and the
    width of its playing-time distribution (a wide Beta is literally 'we don't know how long he
    plays')."""
    tier_risk = _TIER_RISK.get(tier, 1.0)
    a, b = playing_time.beta_params()
    total = a + b
    spread = float(np.sqrt(a * b / (total * total * (total + 1.0)))) if total > 0 else 0.5
    # A snap-share sd of 0.25 is the widest tier in the old config; scale against that.
    return round(min(1.0, 0.6 * tier_risk + 0.4 * min(1.0, spread / 0.25)), 3)


def prior_influence_weights(depth_rank: Optional[int], own_preseason_n: int,
                            team_rotation_nudge_magnitude: float) -> dict:
    """How much each hierarchy layer actually shaped this projection — a transparent
    decomposition (weights sum to ~1), not a formal Bayesian posterior. Built specifically so
    a spot-check across many players can catch the model quietly reverting everyone to the
    same generic prior — pair with `snap_clustering_report` below, which catches the same
    failure mode from the OUTPUT side instead of the input side."""
    historical_w = own_preseason_n / (own_preseason_n + 2.0) if own_preseason_n else 0.0
    remaining = 1.0 - historical_w
    depth_chart_w = remaining * (0.55 if depth_rank is not None else 0.0)
    team_rotation_w = remaining * min(0.15, team_rotation_nudge_magnitude * 1.5)
    position_w = remaining * 0.20
    league_w = max(0.0, remaining - depth_chart_w - team_rotation_w - position_w)
    return {
        "historical_usage_weight": round(historical_w, 3),
        "depth_chart_weight": round(depth_chart_w, 3),
        "team_rotation_weight": round(team_rotation_w, 3),
        "position_prior_weight": round(position_w, 3),
        "league_prior_weight": round(league_w, 3),
    }


def snap_diagnostic(tier: str, tier_reason: str, position: str,
                    playing_time: PlayingTime, team_tendency_obj: Optional[TeamTendency],
                    snap_percentiles: Optional[dict], role_confidence: str,
                    depth_rank: Optional[int], own_preseason_n: int,
                    team_rotation_nudge_magnitude: float) -> dict:
    """The "Snap Model Breakdown" a debugger (or the drawer UI) reads to see WHY a projection
    landed where it did — every field traces to a real input above, nothing here is generated
    independently of the actual computation."""
    weights = prior_influence_weights(depth_rank, own_preseason_n, team_rotation_nudge_magnitude)
    team_rotation_note = "No real preseason team-rotation signal available"
    if team_tendency_obj is not None:
        basis = team_tendency_obj.basis.get("preseason_pass_rate", "")
        if "own team" in basis:
            team_rotation_note = (f"Team pass rate {team_tendency_obj.preseason_pass_rate:.1%} "
                                  f"(regular-season measured, used as a preseason proxy)")
        elif team_tendency_obj.preseason_pass_rate is not None:
            team_rotation_note = (f"League-average pass rate used "
                                  f"({team_tendency_obj.preseason_pass_rate:.1%}) — this "
                                  f"team's own sample was too thin")
    return {
        "depth_chart_tier": tier,
        "depth_chart_reason": tier_reason,
        "position_prior": (position or "").upper() or "unknown",
        "team_rotation": team_rotation_note,
        "historical_usage": ("None — no free source publishes preseason snap counts"
                             if own_preseason_n == 0 else
                             f"{own_preseason_n} of the player's own preseason games"),
        "expected_snaps": playing_time.expected_snaps,
        "snap_percentiles": snap_percentiles,
        "role_confidence": role_confidence,
        "snap_confidence": playing_time.confidence,
        "prior_influence": weights,
    }


def snap_clustering_report(expected_snaps: list[float], target: float = 20.0) -> dict:
    """Detects whether preseason snap projections are degenerately clustered around one value
    — the actual bug this module was rebuilt to fix. NOT a target to optimize toward: the goal
    is to DETECT excessive prior dependence, not to force apart players who genuinely should
    be close together (e.g. two backups on the same depth-chart rank legitimately can land
    near the same number — that's fine; a stat this returns is a symptom to investigate, not a
    thing to minimize for its own sake). Returns {} on an empty input rather than dividing by
    zero."""
    if not expected_snaps:
        return {}
    arr = np.asarray(expected_snaps, dtype=float)
    q = np.percentile(arr, [10, 25, 50, 75, 90])
    n = len(arr)
    return {
        "n": n,
        "mean": round(float(arr.mean()), 2),
        "median": round(float(q[2]), 2),
        "std": round(float(arr.std()), 2),
        "p10": round(float(q[0]), 2), "p25": round(float(q[1]), 2),
        "p50": round(float(q[2]), 2), "p75": round(float(q[3]), 2),
        "p90": round(float(q[4]), 2),
        "pct_within_2_of_target": round(100.0 * float(np.mean(np.abs(arr - target) <= 2.0)), 1),
        "pct_within_3_of_target": round(100.0 * float(np.mean(np.abs(arr - target) <= 3.0)), 1),
        "pct_within_5_of_target": round(100.0 * float(np.mean(np.abs(arr - target) <= 5.0)), 1),
        "n_distinct_values": len(set(round(float(x), 1) for x in arr)),
    }


def measure_tier_shares(snap_rows, depth_entries, season_type: str = "PRE") -> dict:
    """THE measurement pipeline for a share-based preseason table (the mechanism the old model
    used; kept as the honest self-test that nothing is fabricated, and as the pipeline that
    would need re-deriving into a drives-based table the day a real preseason snap/drive
    source exists — see the module docstring).

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
            buckets["likely_starter"].append(float(s.offense_pct))
        elif r == 2:
            buckets["second_team"].append(float(s.offense_pct))
        else:
            buckets["third_team"].append(float(s.offense_pct))
    return {t: (round(float(np.mean(v)), 4), round(float(np.std(v)), 4), len(v))
            for t, v in buckets.items() if len(v) >= 30}
