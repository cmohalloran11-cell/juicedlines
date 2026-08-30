"""
Board integration — attach model projections to live PrizePicks/Underdog/Sleeper NFL lines.

Same contract as basketball/board.py's attach_basketball: take the board's line dicts, filter
to this sport, group by (player, market), project each player once, and write
model_proj / model_prob / model_edge / model_floor / model_ceiling / model_median / model_n /
proj_kind — plus the NFL-specific fields listed in `NFL_FIELDS` below.

Market anchoring: identical mechanism to basketball and tennis. The projected mean is blended
toward the market's standard line in proportion to how much real evidence backs it, and a
projection with almost none defers fully to the market rather than dragging the board with
noise.
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
    "expected_snaps", "snap_range", "expected_routes",
    "expected_routes_basis", "expected_targets", "expected_carries",
    "playing_time_confidence", "playing_time_probability", "role_confidence",
    "game_total", "team_total", "spread", "weather",
    "role", "depth_chart_position", "red_zone_opportunities", "pass_rush_matchup",
    "nfl_confidence", "nfl_confidence_factors", "trust_weight", "model_raw",
    "model_raw_prob", "p25", "p75", "model_std_dev", "opponent", "is_home",
    "snap_p10", "snap_p90", "snap_std_dev",
    "model_pre_mean", "model_pre_median", "model_pre_sd", "model_pre_prob", "model_anchor_t",
)


def _opponent(team: Optional[str], game: Optional[ScheduleGame]) -> tuple[Optional[str], Optional[bool]]:
    """(opponent abbr, is_home) from the matched schedule game, or (None, None) when there
    isn't one. Never guessed from anything but the real schedule row."""
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


def match_game(line: dict, sched_idx: dict) -> Optional[ScheduleGame]:
    """The scheduled game this line belongs to, or None when the nflverse `schedules` release
    carries no row for (this line's date, this line's team) — which is not a failure state:
    the environment then falls back to the measured league baseline and every game-context
    field is reported as unknown rather than defaulted (see model/environment.py). A line is
    still projected without one, because a schedules release that hasn't published or parsed
    must not silently empty the board."""
    team = P.norm_team(line.get("team"))
    start = line.get("start_time") or ""
    day = start[:10] if len(start) >= 10 else None
    return sched_idx.get((day, team)) if (team and day) else None


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
    min_trust = float(cfg("board", "min_trust", default=0.2))
    espn_assets, espn_heads = _espn_assets()

    # The GAME is part of the grouping key, not just the player and market: a board carrying
    # two of the same player's games (a doubleheader-shaped feed, or a line whose date the
    # schedule can't match sitting next to one it can) must not project both off whichever
    # game the first line happened to resolve to.
    groups: dict = defaultdict(list)
    resolved: dict = {}
    for l in nlines:
        market = P._resolve_market(l.get("stat_type") or "")
        if market is None:
            continue
        game = match_game(l, sched_idx)
        key = (P._norm(l["player"]), P.norm_team(l.get("team")), market,
               game.game_id if game else None)
        resolved[key] = game
        groups[key].append(l)

    proj_cache: dict = {}
    done = 0
    for key, glines in groups.items():
        pnorm, team, market = key[0], key[1], key[2]
        first = glines[0]
        game = resolved[key]
        ck = (pnorm, team, game.game_id if game else None)
        if ck not in proj_cache:
            try:
                proj_cache[ck] = P.project_player(
                    first["player"], team=team, position=first.get("position"), game=game)
            except Exception:
                proj_cache[ck] = None
        proj = proj_cache[ck]
        if not proj:
            continue
        arr = P.market_dist(proj, first.get("stat_type") or "")
        if arr is None:
            continue

        model_mean = float(np.mean(arr))
        model_median_raw = float(np.median(arr))
        std = [float(l["line"]) for l in glines
               if (l.get("odds_type") or "standard") == "standard"]
        anchor = float(np.median(std)) if std else float(np.median([float(l["line"]) for l in glines]))

        trust = min(1.0, proj["sample_weight"] / full_trust)
        # Blend on the MEDIAN, not the mean. A market line IS, definitionally, the book's
        # implied 50/50 threshold, so the model's own median — not its mean — is the
        # apples-to-apples quantity to blend it with. The old code blended the MEAN toward
        # the anchor and then shifted the whole array by (blended - mean) to match, which
        # recenters a right-skewed array (Gamma yards) so its MEAN sits on the line while its
        # MEDIAN — the value model_prob is actually computed from, see
        # dataos.validate_direction's own docstring — sits BELOW the line by the
        # distribution's own skew. At low trust (blended ~= anchor = the market line exactly)
        # that produced model_prob < 50% ("Under") on a line showing NO visible edge at all.
        # Found live 2026-08: 94% of the live board recommending Under; verified
        # the mechanism directly on the simulated array (mean 134.7, median 122.4, P(>mean)
        # 44.1%, P(>median) 50.0% exactly). Shifting by median instead makes a zero-trust
        # line correctly read as a 50/50 coinflip, matching what "the line" means.
        blended_median = trust * model_median_raw + (1.0 - trust) * anchor
        if trust < min_trust:
            blended_median = anchor
        shifted = arr + (blended_median - model_median_raw)
        q = np.percentile(shifted, [10, 25, 50, 75, 90])
        # Displayed "Proj" is the MEAN of the correctly-recentered array — still an
        # informative expected value (see WNBA board.py's identical design choice), but now
        # honestly computed: at zero trust it will sit ABOVE a right-skewed stat's market
        # line by the shape's own skew, not exactly on it, because the book's line is their
        # implied median and this engine's own Gamma shape says the mean runs higher than
        # that for this kind of stat.
        center = float(shifted.mean())

        pt = proj["playing_time"]
        env = proj["environment"]
        conf = nfl_confidence(proj, {"model_raw": round(model_mean, 2),
                                     "model_proj": round(center, 2),
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
            l["model_prob"] = round(float((shifted > line_val).mean()), 4)
            l["model_proj"] = round(center, 1)
            l["model_median"] = round(float(q[2]), 1)
            l["model_floor"] = round(max(0.0, float(q[0])), 1)
            l["model_ceiling"] = round(max(float(q[4]), float(q[0])), 1)
            l["model_edge"] = round(center - line_val, 1)
            l["model_n"] = proj["n_games"]
            # A genuine full Monte Carlo run (see valuation._FULL_ENGINE_KINDS).
            l["proj_kind"] = "nfl_regular"
            l["p25"] = round(float(q[1]), 1)
            l["p75"] = round(float(q[3]), 1)
            l["model_std_dev"] = round(float(shifted.std()), 2)
            l["model_raw"] = round(model_mean, 2)
            l["model_raw_prob"] = round(float((arr > line_val).mean()), 4)
            l["trust_weight"] = round(float(trust), 3)
            # PRE-anchor moments of the untouched array + the weight actually left on the
            # model, for the Juice Score. Zero on the snap-to-market branch above, which is
            # the state juice must report as "no model signal" rather than score.
            l["model_pre_mean"] = round(model_mean, 2)
            l["model_pre_median"] = round(model_median_raw, 2)
            l["model_pre_sd"] = round(float(arr.std()), 4)
            l["model_pre_prob"] = l["model_raw_prob"]
            l["model_anchor_t"] = 0.0 if trust < min_trust else round(float(trust), 3)

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
            sp = proj.get("snap_percentiles")
            if sp:
                l["snap_p10"] = sp["p10"]
                l["snap_p90"] = sp["p90"]
                l["snap_std_dev"] = sp["std_dev"]
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
