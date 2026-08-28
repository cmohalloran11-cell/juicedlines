"""
Pace -- how many offensive plays the team actually runs, projected explicitly.

This is the single biggest reason a CFB engine can't reuse a per-game average. Play-count
spread across FBS offenses is wider than any professional league's: a huddle-free tempo team
and a ball-control team are both ordinary FBS programs, and the same usage share against the
same defense is worth materially different counting stats at the two. So the projection is
always `projected_plays x usage_share x efficiency` (see priors.py), and this module owns the
first factor.

Three real inputs, in decreasing order of how much they're allowed to move the number:

 1. The two teams' own plays-per-game, each empirical-Bayes shrunk toward the league mean --
    (observed*n + league*k)/(n+k), with k estimated from the league's within-team vs
    between-team variance exactly as opponent.py shrinks a defense. A game's tempo is a
    property of the matchup, not of one side, so the matchup baseline is the average of the
    two shrunk tendencies -- the same construction basketball/model/pace.py::matchup_pace uses
    for a WNBA possession estimate.
 2. The market's own spread and total for the game (CFBD's /lines, aggregated real books),
    through a least-squares fit of real team plays on them. This is the same design decision
    nfl/model/environment.py documents: the market's total already aggregates everything an
    implied-total model would try to reconstruct, so reading it beats rebuilding it. The fit
    is applied as a DELTA from the league mean, which keeps the matchup's own tempo identity
    intact and lets only the game environment's difference from average move it.
 3. The measured residual spread around that fit, carried into the simulator as the per-trial
    play-count draw rather than collapsed into the point estimate.

Nothing here has a fallback constant. No efficiency rows -> no pace model, and projected_plays
returns None, which the caller turns into "not projected" rather than a league-average guess
for a team it knows nothing about.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..data.base import TeamGameEfficiency
from .config import MIN_PRICED_TEAM_GAMES_FOR_MARKET_FIT, MIN_TEAM_GAMES_FOR_PACE_FIT


@dataclass
class PaceModel:
    league_plays: Optional[float] = None
    team_plays: dict = field(default_factory=dict)       # team -> EB-shrunk plays/game
    market_coef: Optional[tuple] = None                  # (intercept, per_spread, per_total)
    plays_sd_frac: Optional[float] = None                # residual sd / league mean
    n_team_games: int = 0
    n_priced_team_games: int = 0
    basis: str = "unmeasured"


@dataclass
class PaceProjection:
    plays: float
    sd: float
    basis: str                       # market_priced | matchup | league
    team_plays: Optional[float] = None
    opponent_plays: Optional[float] = None


def fit_pace_model(efficiency: list[TeamGameEfficiency], context: dict) -> PaceModel:
    """Fit team tempo + the market's play-count response. `context` maps (game_id, team) ->
    GameContext; efficiency rows key on (week, team), so the context is indexed by that here
    to pick up each row's spread/total."""
    by_team: dict[str, list] = defaultdict(list)
    for e in efficiency:
        if e.plays:
            by_team[e.team].append(float(e.plays))
    n_team_games = sum(len(v) for v in by_team.values())
    model = PaceModel(n_team_games=n_team_games)
    if n_team_games < MIN_TEAM_GAMES_FOR_PACE_FIT or len(by_team) < 2:
        return model

    league = float(np.mean([v for vals in by_team.values() for v in vals]))
    model.league_plays = league
    multi = [vals for vals in by_team.values() if len(vals) >= 2]
    within = float(np.mean([np.var(vals, ddof=1) for vals in multi])) if multi else 0.0
    means = np.array([np.mean(vals) for vals in by_team.values()])
    mean_n = float(np.mean([len(vals) for vals in by_team.values()]))
    between = float(np.var(means, ddof=1)) - (within / mean_n if mean_n else 0.0)
    if within > 0 and between > 0:
        k = within / between
        model.team_plays = {t: (len(v) * float(np.mean(v)) + k * league) / (len(v) + k)
                            for t, v in by_team.items()}
    else:
        model.team_plays = {t: league for t in by_team}
    model.basis = "matchup"

    by_week_team = {(e.season, e.week, e.team): e for e in efficiency if e.plays}
    ctx_by_week_team = {(c.season, c.week, c.team): c for c in context.values()}
    x, y = [], []
    for key, e in by_week_team.items():
        c = ctx_by_week_team.get(key)
        if c is None or c.spread is None or c.over_under is None:
            continue
        x.append([1.0, float(c.spread), float(c.over_under)])
        y.append(float(e.plays))
    model.n_priced_team_games = len(x)

    residuals = np.array([float(v) - model.team_plays[t]
                          for t, vals in by_team.items() for v in vals])
    if len(x) >= MIN_PRICED_TEAM_GAMES_FOR_MARKET_FIT:
        a = np.array(x)
        coef = np.linalg.lstsq(a, np.array(y), rcond=None)[0]
        model.market_coef = (float(coef[0]), float(coef[1]), float(coef[2]))
        residuals = np.array(y) - a @ coef
        model.basis = "market_priced"
    model.plays_sd_frac = float(residuals.std() / league) if league else None
    return model


def projected_plays(model: PaceModel, team: Optional[str], opponent: Optional[str],
                    context) -> Optional[PaceProjection]:
    """Expected offensive plays for `team` in this game, with the measured residual spread.

    None when no pace model exists at all -- a team whose tempo has never been observed and a
    game with no market environment is a projection this engine declines to make rather than
    fills in with the league average.
    """
    if model.league_plays is None:
        return None
    league = model.league_plays
    own = model.team_plays.get(team) if team else None
    opp = model.team_plays.get(opponent) if opponent else None
    known = [v for v in (own, opp) if v is not None]
    base = float(np.mean(known)) if known else league
    basis = "matchup" if known else "league"

    if (model.market_coef and context is not None
            and context.spread is not None and context.over_under is not None):
        a, b, c = model.market_coef
        base += (a + b * float(context.spread) + c * float(context.over_under)) - league
        basis = "market_priced"

    sd = (model.plays_sd_frac or 0.0) * league
    return PaceProjection(plays=base, sd=sd, basis=basis, team_plays=own, opponent_plays=opp)
