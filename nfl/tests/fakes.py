"""Synthetic NFL data source + line builders shared by the test modules. No network."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nfl.data.base import NflDataSource, PlayerWeek, SnapWeek, ScheduleGame

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


def league_filler(n_players=40, season=SEASON, weeks_per=6):
    """Enough distinct players for usage.positional_priors' 200-row floor to be exercised."""
    weeks, snaps = [], []
    for i in range(n_players):
        w, s = full_season(f"Filler {i}", "WR", "BUF", n=weeks_per, season=season,
                           targets=6.0, receptions=4.0, rec_yards=48.0)
        weeks += w
        snaps += s
    return weeks, snaps
