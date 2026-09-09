"""Synthetic NFL data source + line builders shared by the test modules. No network."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nfl.data.base import (DepthChartEntry, NflDataSource, PlayerWeek, RosterWeek, ScheduleGame,
                           SnapWeek)

SEASON = 2026


def week(player="Star Receiver", position="WR", team="CIN", opponent="CLE", season=SEASON,
         wk=1, season_type="REG", **stats) -> PlayerWeek:
    return PlayerWeek(
        season=season, week=wk, season_type=season_type,
        game_id=f"{season}_{wk:02d}_{opponent}_{team}",
        player_id=player.lower().replace(" ", "_"), player=player, position=position,
        team=team, opponent=opponent, **stats)


def snap(player="Star Receiver", position="WR", team="CIN", opponent="CLE", season=SEASON,
         wk=1, season_type="REG", offense_snaps=55.0, offense_pct=0.85) -> SnapWeek:
    return SnapWeek(season=season, week=wk, season_type=season_type,
                    game_id=f"{season}_{wk:02d}_{opponent}_{team}", player=player,
                    position=position, team=team, opponent=opponent,
                    offense_snaps=offense_snaps, offense_pct=offense_pct)


def game(home="CIN", away="CLE", wk=1, gameday="2026-09-13", spread=-3.5, total=45.5,
         season=SEASON, **kw) -> ScheduleGame:
    return ScheduleGame(game_id=f"{season}_{wk:02d}_{away}_{home}", season=season,
                        game_type="REG", week=wk, gameday=gameday, home_team=home,
                        away_team=away, spread_line=spread, total_line=total,
                        home_coach="Home Coach", away_coach="Away Coach", **kw)


def line(player="Star Receiver", stat="Receiving Yards", value=72.5, team="CIN",
         start="2026-09-13T17:00:00Z", **kw) -> dict:
    d = {"id": f"pp_{player}_{stat}", "source": "prizepicks", "sport": "NFL",
         "player": player, "team": team, "position": None, "stat_type": stat,
         "line": value, "odds_type": "standard", "matchup": None, "start_time": start,
         "status": "pre_game", "meta": {}}
    d.update(kw)
    return d


class FakeSource(NflDataSource):
    """Every accessor returns exactly what it was constructed with — an empty list means the
    release genuinely isn't published, which is what the degradation tests exercise."""

    def __init__(self, weeks=None, snaps=None, schedule=None, depth=None, rosters=None):
        self._weeks = weeks or []
        self._snaps = snaps or []
        self._schedule = schedule or []
        self._depth = depth or []
        self._rosters = rosters or []

    def player_weeks(self, season):
        return [w for w in self._weeks if w.season == season]

    def snap_counts(self, season):
        return [s for s in self._snaps if s.season == season]

    def schedule(self, season):
        return [g for g in self._schedule if g.season == season]

    def depth_charts(self, season):
        return [d for d in self._depth if d.season == season]

    def rosters(self, season):
        return [r for r in self._rosters if r.season == season]


def full_season(player="Star Receiver", position="WR", team="CIN", n=10, season=SEASON,
                offense_pct=0.85, **stats):
    """A player with `n` weeks of identical production plus matching snap rows."""
    weeks = [week(player, position, team, wk=w, season=season, **stats) for w in range(1, n + 1)]
    snaps = [snap(player, position, team, wk=w, season=season,
                  offense_snaps=round(offense_pct * 62), offense_pct=offense_pct)
             for w in range(1, n + 1)]
    return weeks, snaps


# A whole offensive skill group at roughly the per-game production and snap share the measured
# positional priors imply for each role — the shape a real team's board actually takes, so a
# team-level check run against it is being tested on a HEALTHY board, not a strawman.
STARTER_OFFENSE = (
    dict(player="Gun Slinger", position="QB", offense_pct=0.99, depth_rank=1,
         pass_attempts=34.0, completions=22.0, pass_yards=245.0, pass_tds=1.6,
         carries=4.0, rush_yards=18.0),
    dict(player="Bell Cow", position="RB", offense_pct=0.62, depth_rank=1,
         carries=15.0, rush_yards=65.0, targets=4.0, receptions=3.0, rec_yards=24.0),
    dict(player="Change Up", position="RB", offense_pct=0.33, depth_rank=2,
         carries=6.0, rush_yards=25.0, targets=2.0, receptions=1.5, rec_yards=12.0),
    dict(player="Alpha Receiver", position="WR", offense_pct=0.88, depth_rank=1,
         targets=9.0, receptions=6.0, rec_yards=84.0),
    dict(player="Second Receiver", position="WR", offense_pct=0.74, depth_rank=2,
         targets=6.0, receptions=4.0, rec_yards=52.0),
    dict(player="Slot Receiver", position="WR", offense_pct=0.55, depth_rank=3,
         targets=4.0, receptions=3.0, rec_yards=31.0),
    dict(player="Seam Tight End", position="TE", offense_pct=0.71, depth_rank=1,
         targets=5.0, receptions=3.5, rec_yards=40.0),
)


def offense(specs=STARTER_OFFENSE, team="CIN", season=SEASON, n=10):
    """(weeks, snaps, depth, rosters) for a whole offensive skill group. Each spec is a
    `player`/`position`/`offense_pct`/`depth_rank` plus that player's per-game stat line, which
    is repeated over `n` identical games — the same construction full_season uses for one
    player, so a fit against it recovers exactly the planted rates."""
    weeks, snaps, depth, rosters = [], [], [], []
    for spec in specs:
        s = dict(spec)
        name, pos = s.pop("player"), s.pop("position")
        pct, rank = s.pop("offense_pct"), s.pop("depth_rank", None)
        w, sn = full_season(name, pos, team, n=n, season=season, offense_pct=pct, **s)
        weeks += w
        snaps += sn
        depth.append(DepthChartEntry(season=season, team=team, position=pos, rank=rank,
                                     player=name))
        rosters.append(RosterWeek(season=season, week=1, team=team, player=name,
                                  position=pos, status="ACT"))
    return weeks, snaps, depth, rosters


def league_filler(n_players=40, season=SEASON, weeks_per=6):
    """Enough distinct players for usage.positional_priors' 200-row floor to be exercised."""
    weeks, snaps = [], []
    for i in range(n_players):
        w, s = full_season(f"Filler {i}", "WR", "BUF", n=weeks_per, season=season,
                           targets=6.0, receptions=4.0, rec_yards=48.0)
        weeks += w
        snaps += s
    return weeks, snaps
