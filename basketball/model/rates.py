"""
Per-possession rate fitting with sample-size shrinkage — the shared core.

For each base stat we estimate a per-possession rate from the player's recent
games (recency-weighted), then regress it toward a league prior using
pseudo-possessions. The prior and the shrinkage strength are the league hooks:
WNBA shrinks lightly toward positional averages (healthy samples → tight); Summer
League shrinks HEAVILY toward a translated pre-NBA prior (tiny sample barely moves it).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import BASE_STATS
from ..data.base import PlayerGame


@dataclass
class PlayerRates:
    player: str
    league: str
    per_poss: dict = field(default_factory=dict)   # stat -> per-possession rate
    eff_games: float = 0.0        # recency-weighted effective games played
    n_games: int = 0
    sample_weight: float = 0.0    # fraction of the estimate from the sample vs prior
    minutes_sample: list = field(default_factory=list)   # (weight, minutes) for minutes.py


def player_possessions(minutes: float, game_len: float, pace: float) -> float:
    """Team possessions a player is on the floor for ≈ minutes-share × team pace."""
    return (minutes / game_len) * pace if game_len else 0.0


def fit_rates(games: list[PlayerGame], league: str, prior_poss: dict,
              game_len: float, league_pace: float, shrink_poss: float,
              halflife: float) -> PlayerRates:
    """`games` most-recent-first. prior_poss: stat -> per-possession prior rate."""
    wsum = wposs = 0.0
    wstat = {s: 0.0 for s in BASE_STATS}
    msample = []
    for i, g in enumerate(games):
        w = 0.5 ** (i / halflife) if halflife > 0 else 1.0
        poss = player_possessions(g.minutes, game_len, league_pace)
        wsum += w
        wposs += w * poss
        msample.append(g.minutes)        # ordered most-recent-first (minutes.py re-weights)
        for s in BASE_STATS:
            wstat[s] += w * g.stat(s)

    per_poss = {}
    for s in BASE_STATS:
        pr = prior_poss.get(s, 0.0)
        denom = wposs + shrink_poss
        per_poss[s] = (wstat[s] + shrink_poss * pr) / denom if denom > 0 else pr

    sample_weight = wposs / (wposs + shrink_poss) if (wposs + shrink_poss) > 0 else 0.0
    return PlayerRates(player=games[0].player if games else "", league=league,
                       per_poss=per_poss, eff_games=round(wsum, 2), n_games=len(games),
                       sample_weight=round(sample_weight, 3), minutes_sample=msample)


def prior_only_rates(league: str, prior_poss: dict) -> PlayerRates:
    """A player with no usable games — the prior IS the estimate (Summer League)."""
    return PlayerRates(player="", league=league, per_poss=dict(prior_poss),
                       eff_games=0.0, n_games=0, sample_weight=0.0)
