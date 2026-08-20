"""
Board integration — attach model projections to live PrizePicks/Underdog/Sleeper NFL lines.

Same contract as basketball/board.py's attach_basketball: take the board's line dicts, filter
to this sport, group by (player, market), project each player once, and write
model_proj / model_prob / model_edge / model_floor / model_ceiling / model_median / model_n /
proj_kind — plus the NFL-specific fields listed in `NFL_FIELDS` below.

Market anchoring: identical mechanism to basketball and tennis. The projected mean is blended
toward the market's standard line in proportion to how much real evidence backs it, and a
projection with almost none defers fully to the market rather than dragging the board with
noise. Preseason is capped well below full trust by construction — the preseason playing-time
tiers are assumptions (see model/rotation.py), so the market gets the LEVEL and the model
contributes the SHAPE that prices the over/under around it.

SEASON TYPE is resolved per line, preferring the book's own classification (PrizePicks'
distinct "NFLP" league code) over schedule-absence inference, and never from the calendar —
see `resolve_season_type`, and read its docstring before assuming the nflverse schedules
release carries a "PRE" game_type, because it does not.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from . import projections as P
from .config import cfg
from .confidence import nfl_confidence
from .data import espn as ESPN
from .data.base import ScheduleGame

_SPORTS = ("NFL",)

# Every NFL-specific field this module writes onto a line, for the integration layer.
NFL_FIELDS = (
    "season_type", "expected_snaps", "snap_range", "expected_routes",
    "expected_routes_basis", "expected_targets", "expected_carries",
    "playing_time_confidence", "playing_time_probability", "role_confidence",
    "preseason_risk", "rotation_tier", "game_total", "team_total", "spread", "weather",
    "role", "depth_chart_position", "red_zone_opportunities", "pass_rush_matchup",
    "nfl_confidence", "nfl_confidence_factors", "trust_weight", "model_raw",
    "model_raw_prob", "p25", "p75", "model_std_dev", "opponent", "is_home",
    "snap_p10", "snap_p90", "snap_std_dev", "prior_influence",
)


def _opponent(team: Optional[str], game: Optional[ScheduleGame]) -> tuple[Optional[str], Optional[bool]]:
    """(opponent abbr, is_home) from the matched schedule game, or (None, None) when there
    isn't one — true for every preseason line by construction (see resolve_season_type's
    docstring: no PRE game is ever in the schedule to match), not a gap specific to this
    function. Never guessed from anything but the real schedule row."""
    if not game or not team:
        return None, None
    home, away = P.norm_team(game.home_team), P.norm_team(game.away_team)
    if team == home:
        return game.away_team, True
    if team == away:
        return game.home_team, False
    return None, None


def _espn_assets() -> tuple[dict, dict]:
    """(team_assets, headshots) from ESPN, each independently try/excepted so a network
    failure or ESPN outage degrades to 'no logo/no override', never breaks projection."""
    try:
        assets = ESPN.team_assets()
    except Exception:
        assets = {}
    try:
        heads = ESPN.all_headshots()
    except Exception:
        heads = {}
    return assets, heads


def _schedule_index(schedule: list[ScheduleGame]) -> dict[tuple, ScheduleGame]:
    """{(gameday, team): game} for every team in every scheduled game, and the same for the
    day either side. A line's start_time is UTC, so a Sunday-night kickoff lands on the
    following calendar day in UTC; matching a +/-1 day window is what makes the lookup
    correct rather than the timezone being guessed at."""
    idx: dict[tuple, ScheduleGame] = {}
    for g in schedule:
        if not g.gameday:
            continue
        try:
            day = datetime.fromisoformat(g.gameday).date()
        except ValueError:
            continue
        for offset in (-1, 0, 1):
            d = (day + timedelta(days=offset)).isoformat()
            for team in (g.home_team, g.away_team):
                idx.setdefault((d, team), g)
    return idx


def resolve_season_type(line: dict, sched_idx: dict) -> tuple[str, Optional[ScheduleGame], bool]:
    """(season_type, matched game, confirmed) for one line.

    Preference order:
    1. `line["book_season_type"]` — a real, book-published classification (PrizePicks tags
       preseason props under a DISTINCT league code, "NFLP", separate from "NFL" — see
       pullers._pp_nfl_season_type; confirmed live 2026-08). When present this is trusted
       outright (`confirmed=True`), including overriding a schedule match/non-match, because
       it's the book's own real data, not something derived here.
    2. Schedule-absence inference, for lines with no book hint (Underdog, or any PrizePicks
       league string _pp_nfl_season_type doesn't recognize). The nflverse `schedules` release
       publishes REGULAR SEASON AND PLAYOFF games only — 7,548 rows across every season it
       covers, game_type in {REG, WC, DIV, CON, SB}, zero preseason games (verified live
       2026-08-15). So there is no "PRE" value to read, and the determination is made the
       only way real data supports: a game between two NFL teams that does NOT appear in the
       authoritative regular-and-postseason schedule is by construction not a regular-season
       game. That is an inference from the schedule, not from the calendar.

    `confirmed` is False only for that absence-based inference; True whenever the season type
    came from either a book classification or an outright schedule match, so a caller can
    tell a real answer from a deduced one. When the line carries no usable date or team at
    all and no book hint, the caller gets ("preseason", None, False) — the wider, lower-
    confidence of the two models, which is the right way to be wrong about an unknown game.
    """
    team = P.norm_team(line.get("team"))
    start = line.get("start_time") or ""
    day = start[:10] if len(start) >= 10 else None
    game = sched_idx.get((day, team)) if (team and day) else None

    book = line.get("book_season_type")
    if book == "preseason":
        return "preseason", None, True
    if book == "regular":
        return "regular", game, True

    if not team or not day:
        return "preseason", None, False
    if game is not None:
        return "regular", game, True
    return "preseason", None, False


def attach_nfl(lines: list[dict]) -> int:
    """Attach projections to live NFL lines. Returns the number of lines projected."""
    nlines = [l for l in lines if l.get("sport") in _SPORTS
              and l.get("player") and l.get("line") is not None]
    if not nlines:
        return 0

    try:
        data = P.league_data()
    except Exception:
        return 0
    sched_idx = _schedule_index(data["schedule"])

    full_trust = float(cfg("board", "full_trust_at", default=0.6))
    pre_cap = float(cfg("board", "preseason_max_trust", default=0.35))
    min_trust = float(cfg("board", "min_trust", default=0.2))
    espn_assets, espn_heads = _espn_assets()

    # The GAME is part of the grouping key, not just the player and market: a board carrying
    # both a preseason and a regular-season line for the same player+market must not resolve
    # one season type for both and project them together — they are two different models.
    groups: dict = defaultdict(list)
    resolved: dict = {}
    for l in nlines:
        market = P._resolve_market(l.get("stat_type") or "")
        if market is None:
            continue
        season_type, game, confirmed = resolve_season_type(l, sched_idx)
        key = (P._norm(l["player"]), P.norm_team(l.get("team")), market, season_type,
               game.game_id if game else None)
        resolved[key] = (season_type, game, confirmed)
        groups[key].append(l)

    proj_cache: dict = {}
    done = 0
    for key, glines in groups.items():
        pnorm, team, market = key[0], key[1], key[2]
        first = glines[0]
        season_type, game, confirmed = resolved[key]
        ck = (pnorm, team, season_type, game.game_id if game else None)
        if ck not in proj_cache:
            try:
                proj_cache[ck] = P.project_player(
                    first["player"], team=team, position=first.get("position"),
                    season_type=season_type, game=game)
            except Exception:
                proj_cache[ck] = None
        proj = proj_cache[ck]
        if not proj:
            continue
        arr = P.market_dist(proj, first.get("stat_type") or "")
        if arr is None:
            continue

        model_mean = float(np.mean(arr))
        std = [float(l["line"]) for l in glines
               if (l.get("odds_type") or "standard") == "standard"]
        anchor = float(np.median(std)) if std else float(np.median([float(l["line"]) for l in glines]))

        trust = min(1.0, proj["sample_weight"] / full_trust)
        if season_type == "preseason":
            trust = min(trust, pre_cap)
        blended = trust * model_mean + (1.0 - trust) * anchor
        if trust < min_trust:
            blended = anchor

        pt = proj["playing_time"]
        env = proj["environment"]
        conf = nfl_confidence(proj, {"model_raw": round(model_mean, 2),
                                     "model_proj": round(blended, 2),
                                     "trust_weight": round(trust, 3),
                                     "lineup_status": first.get("lineup_status")})

        opp_abbr, is_home = _opponent(team, game)
        team_asset = espn_assets.get(P._norm(team)) if team else None
        # ESPN's real photo is authoritative when a name match exists — the book's own
        # image_url is sometimes a team crest instead of a face (observed live 2026-08 on a
        # recently-signed player) and is kept ONLY as the fallback for a player ESPN's active
        # roster doesn't (yet) carry, never blanked for an unmatched name.
        espn_headshot = espn_heads.get(P._norm(first["player"]))

        for l in glines:
            line_val = float(l["line"])
            center = blended
            shifted = arr + (center - model_mean)
            q = np.percentile(shifted, [10, 25, 50, 75, 90])
            l["model_prob"] = round(float((shifted > line_val).mean()), 4)
            l["model_proj"] = round(center, 1)
            l["model_median"] = round(float(q[2]), 1)
            l["model_floor"] = round(max(0.0, float(q[0])), 1)
            l["model_ceiling"] = round(max(float(q[4]), float(q[0])), 1)
            l["model_edge"] = round(center - line_val, 1)
            l["model_n"] = proj["n_games"]
            # Both are genuine full Monte Carlo runs (see valuation._FULL_ENGINE_KINDS), and
            # they're distinct so a calibration query can separate a preseason projection —
            # built on assumed rotation tiers — from a regular-season one.
            l["proj_kind"] = "nfl_preseason" if season_type == "preseason" else "nfl_regular"
            l["p25"] = round(float(q[1]), 1)
            l["p75"] = round(float(q[3]), 1)
            l["model_std_dev"] = round(float(shifted.std()), 2)
            l["model_raw"] = round(model_mean, 2)
            l["model_raw_prob"] = round(float((arr > line_val).mean()), 4)
            l["trust_weight"] = round(float(trust), 3)

            l["season_type"] = season_type
            l["season_type_confirmed"] = confirmed
            l["expected_snaps"] = proj["expected_snaps"]
            l["snap_range"] = proj["snap_range"]
            l["expected_routes"] = proj["expected_routes"]
            l["expected_routes_basis"] = proj["expected_routes_basis"]
            l["expected_targets"] = proj["expected_targets"]
            l["expected_carries"] = proj["expected_carries"]
            l["red_zone_opportunities"] = proj["red_zone_opportunities"]
            l["playing_time_confidence"] = pt.confidence
            l["playing_time_probability"] = pt.playing_time_probability
            l["role_confidence"] = proj["confidence"]
            l["role"] = pt.role
            l["depth_chart_position"] = proj["depth_rank"]
            l["rotation_tier"] = proj["rotation_tier"]
            l["preseason_risk"] = proj["preseason_risk"]
            sp = proj.get("snap_percentiles")
            if sp:
                l["snap_p10"] = sp["p10"]
                l["snap_p90"] = sp["p90"]
                l["snap_std_dev"] = sp["std_dev"]
            if proj.get("prior_influence"):
                l["prior_influence"] = proj["prior_influence"]
            l["pass_rush_matchup"] = proj["pass_rush_matchup"]
            l["game_total"] = env.game_total
            l["team_total"] = env.team_total
            l["spread"] = env.spread
            l["weather"] = env.weather
            l["nfl_confidence"] = conf["score"]
            l["nfl_confidence_factors"] = conf["factors"]

            l["opponent"] = opp_abbr
            l["is_home"] = is_home
            if team_asset and team_asset.get("logo"):
                l["team_logo"] = team_asset["logo"]
            if espn_headshot:
                l["headshot"] = espn_headshot
            done += 1
    return done
