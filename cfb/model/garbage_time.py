"""
Garbage time -- the dominant error source in a college football player projection.

The NFL rarely produces a 45-point game. College football produces them every week, at both
ends of the same distribution: a 35-point favourite empties its bench in the third quarter,
and a 35-point underdog's starters can come out just as early once the game is gone (and its
offense spends the fourth quarter in a game script nobody projected). A projection that
prices a starter's full workload into a game that is over by halftime is wrong in a way no
amount of usage-rate accuracy fixes.

THREE MEASURED PIECES, none of them assumed:

 1. `margin_sd` -- how far a real final margin lands from the market's own expected margin.
    Measured as the spread of (actual margin - spread) over completed, priced games. This is
    the only thing that turns a point spread into a probability.
 2. `blowout_probability(expected_margin)` -- P(|final margin| >= BLOWOUT_MARGIN) under a
    Normal centred on the expected margin with that measured spread. It is symmetric in the
    sign of the margin BY CONSTRUCTION, which is exactly the both-ends requirement: a 35-point
    favourite and a 35-point underdog are the same game and get the same blowout probability.
    (BLOWOUT_MARGIN itself is a label, not a fitted effect -- see config.py. The discount below
    is fitted ON this probability, so the fitted slope absorbs the threshold choice rather than
    the threshold inventing an effect.)
 3. `share_coef` -- the actual usage discount, from an ordinary least squares of the starter
    group's share of its team's skill opportunities on that game's blowout probability, over
    every real team-game in the window. If starters genuinely lose share in likely blowouts,
    the slope is negative and the size of the discount is whatever the data says it is; if
    they don't, the slope is ~0 and this layer does nothing.

Expected margin comes from the market spread wherever CFBD's /lines has priced the game, and
otherwise from a fitted margin-vs-net-PPA relationship over the same completed games, so a
game the books haven't posted still gets a real (if weaker) blowout estimate from team
efficiency instead of being silently treated as a coin-flip matchup. That efficiency fallback
is fitted on season-average net PPA which includes the game being predicted, so its
coefficient is optimistic; it is only ever consulted for an unpriced game, and
`expected_margin`'s `basis` says which source produced the number.

Below the sample gates in config.py nothing here fires: `basis` stays "unmeasured",
blowout_probability returns None and usage_multiplier returns exactly 1.0.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..data.base import PlayerGameStats, TeamGameEfficiency
from .config import (BLOWOUT_MARGIN, MIN_GAMES_FOR_MARGIN_FIT,
                     MIN_TEAM_GAMES_FOR_GARBAGE_FIT, STARTER_GROUP_SIZE)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass
class GarbageTimeModel:
    margin_sd: Optional[float] = None
    n_margin_games: int = 0
    ppa_margin_coef: Optional[tuple] = None          # (intercept, slope on net-PPA difference)
    net_ppa: dict = field(default_factory=dict)      # team -> season mean (offense - defense)
    share_coef: Optional[tuple] = None               # (intercept, slope) on blowout probability
    mean_blowout_p: Optional[float] = None
    league_starter_share: Optional[float] = None
    n_team_games: int = 0
    basis: str = "unmeasured"

    def starter_share_per_player(self) -> Optional[float]:
        if self.league_starter_share is None:
            return None
        return self.league_starter_share / STARTER_GROUP_SIZE


@dataclass
class BlowoutEstimate:
    probability: float
    expected_margin: float
    basis: str                    # market_spread | team_efficiency


def _fit_margin_sd(context: dict) -> tuple[Optional[float], int]:
    """Spread of (actual margin - market expected margin) over completed, priced games. Games
    are de-duplicated by game_id -- each game appears twice in the context index (once per
    team) and the two residuals are the same number with opposite signs."""
    resid: dict[str, float] = {}
    for c in context.values():
        if c.spread is None or c.margin is None:
            continue
        resid.setdefault(c.game_id, float(c.margin) - float(c.spread))
    if len(resid) < MIN_GAMES_FOR_MARGIN_FIT:
        return None, len(resid)
    sd = float(np.std(list(resid.values())))
    return (sd if sd > 0 else None), len(resid)


def _fit_net_ppa_margin(context: dict, efficiency: list[TeamGameEfficiency]
                        ) -> tuple[dict, Optional[tuple]]:
    """Season-average net PPA per team, plus a least-squares margin-vs-net-PPA-difference fit
    -- the expected-margin source for a game the market hasn't priced."""
    acc: dict[str, list] = defaultdict(list)
    for e in efficiency:
        if e.offense_ppa is not None and e.defense_ppa is not None:
            acc[e.team].append(float(e.offense_ppa) - float(e.defense_ppa))
    net = {t: float(np.mean(v)) for t, v in acc.items() if v}
    if not net:
        return {}, None
    x, y = [], []
    seen: set = set()
    for c in context.values():
        if c.margin is None or c.game_id in seen:
            continue
        a, b = net.get(c.team), net.get(c.opponent)
        if a is None or b is None:
            continue
        seen.add(c.game_id)
        x.append([1.0, a - b])
        y.append(float(c.margin))
    if len(x) < MIN_GAMES_FOR_MARGIN_FIT:
        return net, None
    coef = np.linalg.lstsq(np.array(x), np.array(y), rcond=None)[0]
    return net, (float(coef[0]), float(coef[1]))


def fit_garbage_time_model(context: dict, rows: list[PlayerGameStats],
                           efficiency: list[TeamGameEfficiency]) -> GarbageTimeModel:
    """Fit the whole layer off real completed games. `context` maps (game_id, team) ->
    GameContext; `rows` are the box lines those games produced."""
    model = GarbageTimeModel()
    model.margin_sd, model.n_margin_games = _fit_margin_sd(context)
    model.net_ppa, model.ppa_margin_coef = _fit_net_ppa_margin(context, efficiency)
    if model.margin_sd is None:
        return model

    season_opps: dict[tuple, float] = defaultdict(float)
    game_opps: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        o = float(r.rush_attempts + r.receptions)
        if o <= 0:
            continue
        season_opps[(r.team, r.player_id)] += o
        game_opps[(r.game_id, r.team)][r.player_id] = \
            game_opps[(r.game_id, r.team)].get(r.player_id, 0.0) + o

    starters: dict[str, set] = {}
    by_team: dict[str, list] = defaultdict(list)
    for (team, pid), total in season_opps.items():
        by_team[team].append((total, pid))
    for team, pairs in by_team.items():
        pairs.sort(reverse=True)
        starters[team] = {pid for _, pid in pairs[:STARTER_GROUP_SIZE]}

    x, y, shares = [], [], []
    for (game_id, team), players in game_opps.items():
        ctx = context.get((game_id, team))
        if ctx is None:
            continue
        est = expected_margin(model, ctx)
        if est is None:
            continue
        total = sum(players.values())
        if total <= 0:
            continue
        share = sum(v for pid, v in players.items() if pid in starters.get(team, ())) / total
        p = _blowout_p(est[0], model.margin_sd)
        x.append([1.0, p])
        y.append(share)
        shares.append(share)

    model.n_team_games = len(x)
    if len(x) < MIN_TEAM_GAMES_FOR_GARBAGE_FIT:
        return model
    a = np.array(x)
    coef = np.linalg.lstsq(a, np.array(y), rcond=None)[0]
    model.share_coef = (float(coef[0]), float(coef[1]))
    model.mean_blowout_p = float(a[:, 1].mean())
    model.league_starter_share = float(np.mean(shares))
    model.basis = "measured"
    return model


def _blowout_p(mu: float, sd: float) -> float:
    return (1.0 - _normal_cdf((BLOWOUT_MARGIN - mu) / sd)) + _normal_cdf((-BLOWOUT_MARGIN - mu) / sd)


def expected_margin(model: GarbageTimeModel, context) -> Optional[tuple]:
    """(expected margin in this team's favour, basis), from the market spread when the game is
    priced and from the fitted net-PPA relationship when it isn't. None when neither exists."""
    if context is None:
        return None
    if context.spread is not None:
        return float(context.spread), "market_spread"
    if model.ppa_margin_coef:
        a, b = model.ppa_margin_coef
        own, opp = model.net_ppa.get(context.team), model.net_ppa.get(context.opponent)
        if own is not None and opp is not None:
            return a + b * (own - opp), "team_efficiency"
    return None


def blowout_estimate(model: GarbageTimeModel, context) -> Optional[BlowoutEstimate]:
    """P(|final margin| >= BLOWOUT_MARGIN) for this game. None when the margin spread was
    never measurable or the game has neither a market spread nor two teams with efficiency."""
    if model.margin_sd is None:
        return None
    est = expected_margin(model, context)
    if est is None:
        return None
    mu, basis = est
    return BlowoutEstimate(probability=_blowout_p(mu, model.margin_sd),
                           expected_margin=mu, basis=basis)


def starterness(model: GarbageTimeModel, player_share: Optional[float]) -> float:
    """How much of the measured starter-group discount applies to this player, 0..1: their own
    share of their team's skill opportunities against the measured average share of one member
    of a starter group. A true feature back sits at 1.0, a rotational back partway, a player
    with no measurable share at 0.0 (no discount, because the fit was measured on starters and
    the complementary bench-workload boost was never separately fitted -- applying one would be
    inventing it)."""
    per_player = model.starter_share_per_player()
    if not per_player or not player_share:
        return 0.0
    return float(min(1.0, max(0.0, player_share / per_player)))


def usage_multiplier(model: GarbageTimeModel, blowout_probability: Optional[float],
                     player_starterness: float) -> float:
    """Multiplier on projected usage for this game. Exactly 1.0 for an unmeasured model, an
    unpriced/unknown game, or a player with no measurable starter share."""
    if (model.basis != "measured" or blowout_probability is None
            or model.share_coef is None or model.mean_blowout_p is None):
        return 1.0
    a, b = model.share_coef
    reference = a + b * model.mean_blowout_p
    if reference <= 0:
        return 1.0
    predicted = a + b * float(blowout_probability)
    return 1.0 + (predicted / reference - 1.0) * float(player_starterness)
