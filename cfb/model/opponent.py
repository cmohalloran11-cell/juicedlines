"""
Opponent strength -- how much of a stat line was the player and how much was who they played.

College football's schedule is the least balanced in major American sport: an FBS team plays
an FCS opponent most seasons, and the gap between the best and worst FBS defenses is far wider
than the NFL's. A baseline built by averaging raw box scores therefore rates a back who ran
for 180 on an FCS defense above one who ran for 110 against a top-quartile FBS front, which is
backwards. This module measures the size of that effect instead of asserting it.

TWO STAGES, BOTH MEASURED:

 1. Each defense's own quality is empirical-Bayes shrunk toward the league mean before it is
    used at all -- (observed*n + league*k)/(n+k) over its per-game defensive PPA, with k
    estimated from the league's real within-team vs between-team variance. Most of a defense's
    game-to-game PPA is noise, and NFL's matchup module reached the same conclusion from a
    split-half reliability measurement (r=0.17-0.32); shrinking first is what keeps a
    three-game sample from swinging a projection.

 2. The production effect is then fitted, not assumed: an ordinary least squares of
    log(yards per opportunity) on (shrunk opponent defensive PPA - league mean) and an FCS
    indicator, over every real player-game in the window. The log scale is what makes the
    result a MULTIPLIER, and it is also what makes the fit valid at all -- defensive PPA
    straddles zero, so a ratio of PPA values would be meaningless where a difference is fine.

The FCS coefficient is the "down-weight production vs FCS opponents" requirement expressed as
a measured number rather than a chosen one, and the continuous PPA coefficient covers the
bottom-quartile-opponent case on the same scale (`weak_defense_ppa` exposes that quartile
boundary for display). When the window has fewer than the gate's worth of rows, `basis` is
"unmeasured" and production_factor() returns exactly 1.0 for everything -- no adjustment
rather than a plausible one.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..data.base import PlayerGameStats, TeamGameEfficiency
from .config import MIN_GAMES_FOR_FCS_FACTOR, MIN_TEAM_GAMES_FOR_PACE_FIT


@dataclass
class OpponentModel:
    league_defense_ppa: Optional[float] = None
    defense_ppa: dict = field(default_factory=dict)      # team -> EB-shrunk PPA allowed/play
    weak_defense_ppa: Optional[float] = None             # bottom-quartile boundary (worst 25%)
    ppa_coef: Optional[float] = None                     # d log(yards/opportunity) / d PPA
    fcs_coef: Optional[float] = None                     # log-scale FCS production bump
    n_team_games: int = 0
    n_player_games: int = 0
    basis: str = "unmeasured"

    def is_weak(self, team: Optional[str]) -> Optional[bool]:
        """True when this defense sits in the league's bottom quartile, None when unmeasured
        or the team has no efficiency rows -- never False-by-default for an unknown team."""
        if self.weak_defense_ppa is None or team not in self.defense_ppa:
            return None
        return self.defense_ppa[team] >= self.weak_defense_ppa


def _shrink_defense(efficiency: list[TeamGameEfficiency]) -> tuple[Optional[float], dict, int]:
    """(league mean PPA allowed, {team: shrunk PPA allowed}, team-games used). k comes from the
    league's own variance decomposition: within-team game-to-game variance over between-team
    variance, in games -- the number of games of one defense that is worth as much as the
    league prior."""
    by_team: dict[str, list] = defaultdict(list)
    for e in efficiency:
        if e.defense_ppa is not None:
            by_team[e.team].append(float(e.defense_ppa))
    n_games = sum(len(v) for v in by_team.values())
    if n_games < MIN_TEAM_GAMES_FOR_PACE_FIT or len(by_team) < 2:
        return None, {}, n_games

    league = float(np.mean([v for vals in by_team.values() for v in vals]))
    multi = [vals for vals in by_team.values() if len(vals) >= 2]
    within = float(np.mean([np.var(vals, ddof=1) for vals in multi])) if multi else 0.0
    means = np.array([np.mean(vals) for vals in by_team.values()])
    mean_n = float(np.mean([len(vals) for vals in by_team.values()]))
    between = float(np.var(means, ddof=1)) - (within / mean_n if mean_n else 0.0)

    if within <= 0 or between <= 0:
        # Every apparent difference between defenses is game-to-game noise on this window:
        # the honest estimate for every team is the league mean, not a lightly-shrunk one.
        return league, {t: league for t in by_team}, n_games

    k = within / between
    shrunk = {t: (len(v) * float(np.mean(v)) + k * league) / (len(v) + k)
              for t, v in by_team.items()}
    return league, shrunk, n_games


def fit_opponent_model(efficiency: list[TeamGameEfficiency], rows: list[PlayerGameStats],
                       context: dict) -> OpponentModel:
    """Fit the opponent adjustment off real efficiency rows and real box lines.

    `context` maps (game_id, team) -> GameContext, which is where the opponent and its
    classification come from. Rows whose game isn't in the context (a game the schedule pull
    didn't cover) are skipped rather than treated as league-average opponents.
    """
    league, shrunk, n_team_games = _shrink_defense(efficiency)
    model = OpponentModel(league_defense_ppa=league, defense_ppa=shrunk,
                          n_team_games=n_team_games)
    if shrunk:
        model.weak_defense_ppa = float(np.percentile(list(shrunk.values()), 75))

    x, y = [], []
    for r in rows:
        ctx = context.get((r.game_id, r.team))
        if ctx is None:
            continue
        opportunities = r.rush_attempts + r.receptions
        yards = r.rush_yards + r.rec_yards
        if opportunities <= 0 or yards <= 0:
            continue
        ppa = shrunk.get(ctx.opponent)
        delta = (ppa - league) if (ppa is not None and league is not None) else 0.0
        is_fcs = 1.0 if (ctx.opponent_classification or "").lower() == "fcs" else 0.0
        x.append([1.0, delta, is_fcs])
        y.append(np.log(yards / opportunities))

    model.n_player_games = len(x)
    if len(x) < MIN_GAMES_FOR_FCS_FACTOR:
        return model
    coef = np.linalg.lstsq(np.array(x), np.array(y), rcond=None)[0]
    model.ppa_coef = float(coef[1])
    model.fcs_coef = float(coef[2])
    model.basis = "measured"
    return model


def production_factor(model: OpponentModel, opponent: Optional[str],
                      opponent_classification: Optional[str] = None) -> float:
    """How much this opponent inflates (>1) or suppresses (<1) yards per opportunity. Exactly
    1.0 when the model is unmeasured or the opponent has no efficiency rows and isn't FCS --
    the caller's baseline is then simply un-normalised, which is honest, rather than nudged by
    a number nothing measured."""
    if model.basis != "measured":
        return 1.0
    delta = 0.0
    ppa = model.defense_ppa.get(opponent) if opponent else None
    if ppa is not None and model.league_defense_ppa is not None:
        delta = ppa - model.league_defense_ppa
    is_fcs = 1.0 if (opponent_classification or "").lower() == "fcs" else 0.0
    return float(np.exp((model.ppa_coef or 0.0) * delta + (model.fcs_coef or 0.0) * is_fcs))
