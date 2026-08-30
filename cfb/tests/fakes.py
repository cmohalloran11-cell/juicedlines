"""
A synthetic FBS league, big enough to exercise every gate in cfb/model/config.py.

This is a FIXTURE, not data: no CFBD_API_KEY exists in any environment this repo has run in
(see cfb/data/cfbd_client.py's module docstring), so the engine's fits have never been run
against a live response. What these rows are for is proving the MECHANISM does what the code
claims -- that a known between-player spread comes back out of the empirical-Bayes estimator,
that a known FCS production inflation comes back out of the opponent fit, that a known
blowout usage drop comes back out of the garbage-time fit. Nothing measured here is a claim
about real college football, and no number fitted from it is ever written into the engine.

The generator plants exactly four effects, so a test can assert the fits recover them:

    FCS_INFLATION          yards per opportunity vs an FCS opponent
    DEFENSE_SENSITIVITY    d log(yards per opportunity) / d (opponent defensive PPA)
    BLOWOUT_STARTER_FACTOR starter usage multiplier in a game that ends up a blowout
    per-team base play counts, so team tempo has a real between-team spread to shrink toward
"""
from __future__ import annotations

import math
import random
import zlib
from typing import Optional

from ..data.base import (CfbDataSource, PlayerGameStats, PlayerRef, RecruitRating,
                         ScheduleGame, TeamGameEfficiency, TeamRef)

SEASON = 2026

POWER_TEAMS = [f"Power {i}" for i in range(20)]
GROUP_TEAMS = [f"Group {i}" for i in range(20)]
FBS_TEAMS = POWER_TEAMS + GROUP_TEAMS
FCS_TEAMS = [f"Lower {i}" for i in range(8)]

ROLE_POSITION = {"QB": "QB", "RB1": "RB", "RB2": "RB", "WR1": "WR", "WR2": "WR", "TE": "TE"}
ROLES = tuple(ROLE_POSITION)

# Planted effects (see module docstring).
FCS_INFLATION = 1.25
DEFENSE_SENSITIVITY = 0.60
BLOWOUT_STARTER_FACTOR = 0.80
BLOWOUT_BACKUP_FACTOR = 1.18
BLOWOUT_MARGIN = 21.0

# Planted per-play usage and per-attempt efficiency, before any of the effects above.
_USAGE = {
    "QB":  {"pass_att_per_play": 0.42, "rush_att_per_play": 0.06, "rec_per_play": 0.0},
    "RB1": {"pass_att_per_play": 0.0, "rush_att_per_play": 0.26, "rec_per_play": 0.03},
    "RB2": {"pass_att_per_play": 0.0, "rush_att_per_play": 0.11, "rec_per_play": 0.02},
    "WR1": {"pass_att_per_play": 0.0, "rush_att_per_play": 0.01, "rec_per_play": 0.10},
    "WR2": {"pass_att_per_play": 0.0, "rush_att_per_play": 0.0, "rec_per_play": 0.07},
    "TE":  {"pass_att_per_play": 0.0, "rush_att_per_play": 0.0, "rec_per_play": 0.05},
}
_STARTER_ROLES = ("RB1", "WR1", "QB")
_YPC, _YPR, _YPA, _COMP = 4.6, 12.0, 7.6, 0.62
_RUSH_TD, _REC_TD, _PASS_TD = 0.045, 0.085, 0.070

_LEAGUE_DEF_PPA = 0.22


def _team_index(team: str) -> int:
    return FBS_TEAMS.index(team) if team in FBS_TEAMS else 99


def base_plays(team: str) -> float:
    return 60.0 + (_team_index(team) % 11) * 2.0


def defense_ppa(team: str) -> float:
    return 0.10 + (_team_index(team) % 9) * 0.03


def offense_ppa(team: str) -> float:
    return 0.12 + (_team_index(team) % 7) * 0.035


def team_strength(team: str) -> float:
    return offense_ppa(team) - defense_ppa(team)


def _stable_seed(*parts: object) -> int:
    """A seed that's the same value on every process run, unlike Python's builtin hash() (salted
    per-process by PEP 456 since 3.3 -- see the generator section below for why that matters
    here)."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) % (2 ** 31)


def _skill(player_id: str) -> float:
    """A stable per-player multiplier, so the league has a real between-player spread for the
    empirical-Bayes estimator to find rather than pure sampling noise."""
    return 0.75 + 0.5 * ((_stable_seed(player_id) % 1000) / 1000.0)


def _rounds(teams: list, week: int) -> list:
    n = len(teams)
    shift = (week - 1) % (n - 1)
    rotated = [teams[0]] + teams[1 + shift:] + teams[1:1 + shift]
    return [(rotated[i], rotated[n - 1 - i]) for i in range(n // 2)]


def _games(season: int) -> list[ScheduleGame]:
    rng = random.Random(season)
    out: list[ScheduleGame] = []
    for i, opp in enumerate(FCS_TEAMS):
        home = POWER_TEAMS[i]
        margin = 31.0 + i
        out.append(ScheduleGame(
            id=f"{season}-0-{i}", season=season, week=0, season_type="regular",
            start_date=f"{season}-08-29T23:00:00.000Z", home_team=home, away_team=opp,
            home_classification="fbs", away_classification="fcs",
            spread=-28.0, over_under=58.0, completed=True,
            home_points=int(24 + margin), away_points=24))
    for week in range(1, 13):
        for gi, (home, away) in enumerate(_rounds(FBS_TEAMS, week)):
            edge = 25.0 * (team_strength(home) - team_strength(away)) + 2.5
            margin = edge + rng.gauss(0.0, 15.0)
            total = base_plays(home) + base_plays(away) - 78.0
            out.append(ScheduleGame(
                id=f"{season}-{week}-{gi}", season=season, week=week, season_type="regular",
                start_date=f"{season}-09-{min(28, week + 5):02d}T23:00:00.000Z",
                home_team=home, away_team=away,
                home_classification="fbs", away_classification="fbs",
                spread=round(-edge, 1), over_under=round(total, 1), completed=True,
                home_points=int(round(27 + margin / 2)), away_points=int(round(27 - margin / 2))))
    return out


class FakeSource(CfbDataSource):
    """A complete two-and-a-bit season league. Deterministic: same rows every run."""

    def __init__(self, seasons: Optional[range] = None, completed: bool = True):
        self.seasons = range(SEASON - 2, SEASON + 1) if seasons is None else seasons
        self.completed = completed
        self._cache: dict = {}

    # ── source interface ─────────────────────────────────────────────────────────────────

    def teams(self, season: int, classification: str = "fbs") -> list[TeamRef]:
        out = [TeamRef(id=str(i), school=t, conference="SEC" if t in POWER_TEAMS else "MAC",
                       classification="fbs", abbreviation=t[:3].upper())
               for i, t in enumerate(FBS_TEAMS)]
        return out if classification == "fbs" else []

    def roster(self, team: str, season: int) -> list[PlayerRef]:
        return [PlayerRef(id=f"{team}-{r}", name=f"{team} {r}", team=team,
                          position=ROLE_POSITION[r]) for r in ROLES]

    def schedule(self, season: int, week: Optional[int] = None) -> list[ScheduleGame]:
        if season not in self.seasons:
            return []
        games = self._cache.setdefault(("sched", season), _games(season))
        if not self.completed:
            games = [ScheduleGame(**{**g.__dict__, "completed": False,
                                     "home_points": None, "away_points": None})
                     for g in games]
        return games if week is None else [g for g in games if g.week == week]

    def team_efficiency(self, season: int, week: Optional[int] = None) -> list[TeamGameEfficiency]:
        if season not in self.seasons:
            return []
        hit = self._cache.get(("eff", season))
        if hit is None:
            rng = random.Random(season * 7)
            hit = []
            for g in self.schedule(season):
                for team, opp in ((g.home_team, g.away_team), (g.away_team, g.home_team)):
                    if team not in FBS_TEAMS:
                        continue
                    hit.append(TeamGameEfficiency(
                        season=season, week=g.week, team=team, opponent=opp,
                        offense_ppa=offense_ppa(team) + rng.gauss(0, 0.05),
                        defense_ppa=defense_ppa(team) + rng.gauss(0, 0.05),
                        offense_success_rate=0.43, defense_success_rate=0.42,
                        plays=int(round(self._plays(season, g, team)))))
            self._cache[("eff", season)] = hit
        return hit if week is None else [e for e in hit if e.week == week]

    def player_game_stats(self, season: int, week: Optional[int] = None,
                          team: Optional[str] = None) -> list[PlayerGameStats]:
        if season not in self.seasons:
            return []
        hit = self._cache.get(("box", season))
        if hit is None:
            hit = []
            for g in self.schedule(season):
                for t, opp in ((g.home_team, g.away_team), (g.away_team, g.home_team)):
                    if t not in FBS_TEAMS:
                        continue
                    hit.extend(self._team_box(season, g, t, opp))
            self._cache[("box", season)] = hit
        rows = hit if week is None else [r for r in hit if r.week == week]
        return rows if team is None else [r for r in rows if r.team == team]

    def recruiting(self, season: int) -> list[RecruitRating]:
        """One signing class: the RB2/WR2 of every team, rated in proportion to the usage they
        go on to hold, so the tier-C rating fit has a real relationship to recover."""
        out = []
        for t in FBS_TEAMS:
            for role in ("RB2", "WR2"):
                pid = f"{t}-{role}"
                out.append(RecruitRating(season=season, name=f"{t} {role}", team=t,
                                         position=ROLE_POSITION[role],
                                         rating=round(0.80 + 0.15 * (_skill(pid) - 0.75) / 0.5, 4),
                                         stars=4))
            # A rated true freshman with no college production at all -- the tier-C case, and
            # the only kind of player for whom the recruiting rating IS the whole prior.
            out.append(RecruitRating(season=season, name=f"{t} Freshman", team=t,
                                     position="RB", rating=0.92, stars=4))
        return out

    # ── generator ────────────────────────────────────────────────────────────────────────
    #
    # Seeding uses zlib.crc32 on a stable string, not Python's built-in hash(). Builtin hash()
    # of a str/tuple is salted per-process (PEP 456, on by default since Python 3.3) -- every
    # fresh `pytest` invocation gets a different salt, so a seed derived from hash() produces
    # different "random" fixture data on every real run. Locally that's invisible (one process,
    # one hidden salt, always looks deterministic); in CI it means the exact numbers this
    # fixture generates -- and therefore how close a fitted coefficient lands to the planted
    # value -- silently change from run to run. That's what made
    # test_the_opponent_fit_recovers_the_planted_fcs_and_defense_effects fail intermittently:
    # not test order, the interpreter's hash salt. crc32 has no such randomization.

    def _plays(self, season: int, game: ScheduleGame, team: str) -> float:
        rng = random.Random(_stable_seed(season, game.id, team))
        return max(40.0, base_plays(team) + rng.gauss(0.0, 5.0))

    def _team_box(self, season: int, game: ScheduleGame, team: str,
                  opponent: str) -> list[PlayerGameStats]:
        plays = self._plays(season, game, team)
        margin = game.margin_for(team) or 0.0
        blowout = abs(margin) >= BLOWOUT_MARGIN
        opp_factor = FCS_INFLATION if opponent in FCS_TEAMS else math.exp(
            DEFENSE_SENSITIVITY * (defense_ppa(opponent) - _LEAGUE_DEF_PPA))
        rows = []
        for role in ROLES:
            pid = f"{team}-{role}"
            rng = random.Random(_stable_seed(season, game.id, pid))
            skill = _skill(pid)
            usage = 1.0
            if blowout:
                usage = BLOWOUT_STARTER_FACTOR if role in _STARTER_ROLES else BLOWOUT_BACKUP_FACTOR
            u = _USAGE[role]
            carries = max(0, int(round(u["rush_att_per_play"] * plays * usage * skill
                                       + rng.gauss(0, 1.2))))
            recs = max(0, int(round(u["rec_per_play"] * plays * usage * skill
                                    + rng.gauss(0, 1.0))))
            atts = max(0, int(round(u["pass_att_per_play"] * plays * usage * skill
                                    + rng.gauss(0, 2.5))))
            comps = min(atts, max(0, int(round(atts * _COMP + rng.gauss(0, 1.5)))))
            rows.append(PlayerGameStats(
                season=season, week=game.week, season_type="regular", game_id=game.id,
                player_id=pid, player=f"{team} {role}", team=team, opponent=opponent,
                pass_attempts=float(atts), pass_completions=float(comps),
                pass_yards=round(atts * _YPA * opp_factor * skill + rng.gauss(0, 18), 1),
                pass_tds=float(max(0, round(comps * _PASS_TD + rng.gauss(0, 0.5)))),
                rush_attempts=float(carries),
                rush_yards=round(carries * _YPC * opp_factor * skill + rng.gauss(0, 9), 1),
                rush_tds=float(max(0, round(carries * _RUSH_TD + rng.gauss(0, 0.4)))),
                receptions=float(recs), targets=float(recs),
                rec_yards=round(recs * _YPR * opp_factor * skill + rng.gauss(0, 12), 1),
                rec_tds=float(max(0, round(recs * _REC_TD + rng.gauss(0, 0.3))))))
        return rows


def positions() -> dict:
    """{normalized player name: position} -- what cfb/players_sync.py's canonical table would
    hold for this league, injected via projections.set_positions_override so the engine's fits
    can be exercised without a database."""
    from fantasy.player_matching import normalize_name
    return {normalize_name(f"{t} {r}"): ROLE_POSITION[r] for t in FBS_TEAMS for r in ROLES}


def line(player: str = "Power 0 RB1", team: str = "Power 0", position: str = "RB",
         stat: str = "Rushing Yards", value: float = 82.5, season: int = SEASON,
         week: int = 3) -> dict:
    """One board Line dict in cfb/lines.py's shape, for a real scheduled game."""
    games = FakeSource().schedule(season, week=week)
    game = next(g for g in games if team in (g.home_team, g.away_team))
    return {
        "id": f"cfb_odds_{game.id}_draftkings_{stat}_{player}", "source": "draftkings",
        "sport": "CFB", "player": player, "team": team, "position": position,
        "stat_type": stat, "line": value, "odds_type": "standard",
        "matchup": f"{game.away_team} @ {game.home_team}",
        "start_time": game.start_date, "over_price": "-115", "under_price": "-105",
        "over_implied": 0.535, "under_implied": 0.512,
        "meta": {"event_id": game.id, "cfb_player_id": None}, "game_id": game.id,
    }
