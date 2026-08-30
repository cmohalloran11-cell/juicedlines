"""
Live nflverse-data adapter — free, no auth, no key.

Releases used, every URL and column name confirmed live 2026-08-15 by streaming the header
row before anything here was written:

  stats_player   stats_player_week_{season}.csv — weekly per-player stats. NOT the older
                 `player_stats/player_stats.csv` single file, which stops at the 2024 season
                 and carries no game_id (fantasy/projections/nflverse_adapter.py still points
                 at it). Columns used: player_id, player_display_name, position, season, week,
                 season_type, game_id, team, opponent_team, completions, attempts,
                 passing_yards, passing_tds, passing_interceptions, sacks_suffered, carries,
                 rushing_yards, rushing_tds, receptions, targets, receiving_yards,
                 receiving_tds.
  snap_counts    snap_counts_{season}.csv — game_id, season, game_type, week, player,
                 position, team, opponent, offense_snaps, offense_pct, defense_snaps,
                 defense_pct, st_snaps, st_pct.
  schedules      games.csv (one file, all seasons) — game_id, season, game_type, week,
                 gameday, home_team, away_team, spread_line, total_line, roof, surface, temp,
                 wind, home_coach, away_coach.
  depth_charts   depth_charts_{season}.csv. TWO schemas exist and both are handled: seasons
                 <=2024 carry (season, club_code, week, game_type, depth_team, position,
                 depth_position, full_name, gsis_id); seasons >=2025 carry a rolling snapshot
                 (dt, team, player_name, gsis_id, pos_grp, pos_name, pos_abb, pos_slot,
                 pos_rank) with no week column at all, so the latest `dt` per player wins.
  weekly_rosters roster_weekly_{season}.csv — season, team, position, depth_chart_position,
                 status, full_name, gsis_id, week, game_type.

WHAT THIS SOURCE DOES NOT HAVE (so the model marks these unknown rather than inventing them):
  * PRESEASON anything. snap_counts has 0 rows with game_type PRE in every published season;
    schedules has 0 preseason games (7,548 rows, game_type in {REG,WC,DIV,CON,SB}). This
    engine projects regular and postseason games only.
  * ROUTES RUN. Not published in any free nflverse release (it's an NGS/PFF field). The usage
    model reports pass-play snaps instead and labels it as such — see model/usage.py.
  * RED-ZONE opportunity splits. Only derivable from the play-by-play release, which this
    adapter does not pull; reported as None.
"""

from __future__ import annotations

import csv
import io
import os

from .base import (NflDataSource, PlayerWeek, SnapWeek, ScheduleGame, DepthChartEntry,
                   RosterWeek)
from . import cache
from ..config import cfg

_BASE = os.getenv("NFL_NFLVERSE_BASE",
                  "https://github.com/nflverse/nflverse-data/releases/download")

_STATS_URL = _BASE + "/stats_player/stats_player_week_{season}.csv"
_SNAPS_URL = _BASE + "/snap_counts/snap_counts_{season}.csv"
_SCHEDULE_URL = _BASE + "/schedules/games.csv"
_DEPTH_URL = _BASE + "/depth_charts/depth_charts_{season}.csv"
_ROSTER_URL = _BASE + "/weekly_rosters/roster_weekly_{season}.csv"

_SKILL = {"QB", "RB", "WR", "TE", "FB"}


def _f(v) -> float:
    try:
        return float(v) if v not in (None, "", "NA") else 0.0
    except (TypeError, ValueError):
        return 0.0


def _of(v):
    try:
        return float(v) if v not in (None, "", "NA") else None
    except (TypeError, ValueError):
        return None


def _oi(v):
    f = _of(v)
    return int(f) if f is not None else None


def fetch_rows(url: str, ttl: float | None = None) -> list[dict]:
    """Parse one release CSV into raw dicts. Isolated from every mapper below so tests can
    feed synthetic rows with no HTTP at all (the same split nflverse_adapter.fetch_rows uses).
    Returns [] for a release that isn't published — a season that hasn't started yet."""
    text = cache.fetch_text(url, ttl if ttl is not None else cfg("data", "cache_ttl", default=21600))
    if not text:
        return []
    return list(csv.DictReader(io.StringIO(text)))


def to_player_weeks(rows: list[dict]) -> list[PlayerWeek]:
    out = []
    for r in rows:
        pos = (r.get("position") or "").upper()
        if pos not in _SKILL:
            continue
        season, week = _oi(r.get("season")), _oi(r.get("week"))
        if season is None or week is None:
            continue
        out.append(PlayerWeek(
            season=season, week=week, season_type=r.get("season_type") or "",
            game_id=r.get("game_id") or "", player_id=r.get("player_id") or "",
            player=r.get("player_display_name") or r.get("player_name") or "",
            position="RB" if pos == "FB" else pos,
            team=r.get("team") or r.get("recent_team") or "",
            opponent=r.get("opponent_team") or "",
            completions=_f(r.get("completions")), pass_attempts=_f(r.get("attempts")),
            pass_yards=_f(r.get("passing_yards")), pass_tds=_f(r.get("passing_tds")),
            interceptions=_f(r.get("passing_interceptions") or r.get("interceptions")),
            sacks=_f(r.get("sacks_suffered") or r.get("sacks")),
            carries=_f(r.get("carries")), rush_yards=_f(r.get("rushing_yards")),
            rush_tds=_f(r.get("rushing_tds")),
            receptions=_f(r.get("receptions")), targets=_f(r.get("targets")),
            rec_yards=_f(r.get("receiving_yards")), rec_tds=_f(r.get("receiving_tds")),
        ))
    return out


def to_snap_weeks(rows: list[dict]) -> list[SnapWeek]:
    out = []
    for r in rows:
        season, week = _oi(r.get("season")), _oi(r.get("week"))
        if season is None or week is None:
            continue
        out.append(SnapWeek(
            season=season, week=week, season_type=r.get("game_type") or "",
            game_id=r.get("game_id") or "", player=r.get("player") or "",
            position=(r.get("position") or "").upper(), team=r.get("team") or "",
            opponent=r.get("opponent") or "",
            offense_snaps=_of(r.get("offense_snaps")), offense_pct=_of(r.get("offense_pct")),
            defense_snaps=_of(r.get("defense_snaps")), defense_pct=_of(r.get("defense_pct")),
        ))
    return out


def to_schedule(rows: list[dict], season: int) -> list[ScheduleGame]:
    out = []
    for r in rows:
        s = _oi(r.get("season"))
        if s != season:
            continue
        out.append(ScheduleGame(
            game_id=r.get("game_id") or "", season=s, game_type=r.get("game_type") or "",
            week=_oi(r.get("week")) or 0, gameday=r.get("gameday") or "",
            home_team=r.get("home_team") or "", away_team=r.get("away_team") or "",
            spread_line=_of(r.get("spread_line")), total_line=_of(r.get("total_line")),
            roof=r.get("roof") or None, surface=r.get("surface") or None,
            temp=_of(r.get("temp")), wind=_of(r.get("wind")),
            home_coach=r.get("home_coach") or None, away_coach=r.get("away_coach") or None,
        ))
    return out


def to_depth_charts(rows: list[dict], season: int) -> list[DepthChartEntry]:
    """Latest depth-chart entry per (team, position, player). Handles both published schemas
    (see the module docstring); the post-2024 snapshot format has many timestamps per season,
    so the newest `dt` wins."""
    if not rows:
        return []
    modern = "pos_rank" in rows[0]
    best: dict[tuple, tuple] = {}
    for r in rows:
        if modern:
            team, player = r.get("team") or "", r.get("player_name") or ""
            position = (r.get("pos_abb") or "").upper()
            rank, order, week, slot = _oi(r.get("pos_rank")), r.get("dt") or "", None, r.get("pos_slot")
            player_id = r.get("gsis_id") or None
            group = (r.get("pos_grp") or "").lower()
            if group and "off" not in group and position not in ("QB", "RB", "WR", "TE", "FB"):
                continue
        else:
            if (r.get("formation") or "Offense") != "Offense":
                continue
            team, player = r.get("club_code") or "", r.get("full_name") or ""
            position = (r.get("position") or "").upper()
            rank, week = _oi(r.get("depth_team")), _oi(r.get("week"))
            order, slot = f"{_oi(r.get('week')) or 0:03d}", r.get("depth_position")
            player_id = r.get("gsis_id") or None
        if not team or not player:
            continue
        key = (team, position, player.strip().lower())
        prev = best.get(key)
        if prev is None or order > prev[0]:
            best[key] = (order, DepthChartEntry(
                season=season, team=team, position=position, rank=rank, player=player,
                player_id=player_id, week=week, slot=slot))
    return [v[1] for v in best.values()]


def to_rosters(rows: list[dict], season: int) -> list[RosterWeek]:
    out = []
    for r in rows:
        week = _oi(r.get("week"))
        if week is None:
            continue
        out.append(RosterWeek(
            season=season, week=week, team=r.get("team") or "",
            player=r.get("full_name") or "", position=(r.get("position") or "").upper(),
            status=r.get("status") or None,
            depth_chart_position=r.get("depth_chart_position") or None,
            player_id=r.get("gsis_id") or None,
        ))
    return out


class NflverseSource(NflDataSource):
    """The live adapter. Each accessor is memoized per (release, season) — see data/cache.py
    for why both an in-process and an on-disk layer exist."""

    def _ttl(self) -> float:
        return float(cfg("data", "cache_ttl", default=21600))

    def player_weeks(self, season: int) -> list[PlayerWeek]:
        return cache.memoize(
            ("NFL", "stats_player", season), self._ttl(),
            lambda: to_player_weeks(fetch_rows(_STATS_URL.format(season=season))))

    def snap_counts(self, season: int) -> list[SnapWeek]:
        return cache.memoize(
            ("NFL", "snap_counts", season), self._ttl(),
            lambda: to_snap_weeks(fetch_rows(_SNAPS_URL.format(season=season))))

    def schedule(self, season: int) -> list[ScheduleGame]:
        return cache.memoize(
            ("NFL", "schedules", season), self._ttl(),
            lambda: to_schedule(fetch_rows(_SCHEDULE_URL), season))

    def depth_charts(self, season: int) -> list[DepthChartEntry]:
        return cache.memoize(
            ("NFL", "depth_charts", season), self._ttl(),
            lambda: to_depth_charts(fetch_rows(_DEPTH_URL.format(season=season)), season))

    def rosters(self, season: int) -> list[RosterWeek]:
        return cache.memoize(
            ("NFL", "weekly_rosters", season), self._ttl(),
            lambda: to_rosters(fetch_rows(_ROSTER_URL.format(season=season)), season))
