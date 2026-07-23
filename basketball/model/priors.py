"""
League priors — the shrinkage targets. This is the main hook a league config overrides.

WNBA: regress rates toward positional per-40 averages (healthy samples make these a
formality).
"""

from __future__ import annotations

from .. import BASE_STATS

# Positional per-40 baselines (shrinkage targets). Rough, league-ish.
_WNBA_PER40 = {
    "G": {"pts": 16.0, "reb": 4.0, "ast": 5.0, "stl": 1.6, "blk": 0.4, "3pm": 2.0, "to": 2.6},
    "F": {"pts": 15.0, "reb": 7.0, "ast": 2.6, "stl": 1.2, "blk": 0.9, "3pm": 1.4, "to": 2.0},
    "C": {"pts": 15.0, "reb": 9.5, "ast": 2.0, "stl": 0.9, "blk": 1.6, "3pm": 0.6, "to": 2.2},
    "":  {"pts": 15.0, "reb": 6.5, "ast": 3.2, "stl": 1.2, "blk": 0.8, "3pm": 1.4, "to": 2.2},
}

# Fraction of a player's rebounds that are OFFENSIVE — the shrinkage target for the
# orb/drb split (see DERIVED_STATS). MEASURED off ESPN box scores, 82 completed WNBA games
# (2026-06-15..07-17); ORB+DRB reconciled to REB exactly, which validates the parse:
#     G 0.94/4.53 = 20.8%   F 2.29/8.21 = 27.9%   C 2.71/10.54 = 25.7%
_ORB_SHARE = {"G": 0.21, "F": 0.28, "C": 0.26, "": 0.24}
# pseudo-rebounds of prior weight — the split is a stable skill, so a modest sample moves it.
_ORB_SHARE_SHRINK = 25.0
# How much of a Summer-League player's COLLEGE offensive share carries into the positional
def orb_share_prior(position: str) -> float:
    """Baseline offensive-rebound share for a position."""
    return _ORB_SHARE.get((position or "")[:1].upper(), _ORB_SHARE[""])


def fit_orb_share(games, position: str) -> float:
    """Player's offensive-rebound share, shrunk toward the positional baseline.

    Uses whatever games carry the split (box scores); a player with none falls back to the
    prior entirely. Shrinking matters: an 8-game sample of a low-rebound guard can read
    0% or 50% on noise alone.
    """
    prior = orb_share_prior(position)
    o = sum(getattr(g, "orb", 0.0) or 0.0 for g in games)
    d = sum(getattr(g, "drb", 0.0) or 0.0 for g in games)
    n = o + d
    if n <= 0:
        return prior
    return float((o + _ORB_SHARE_SHRINK * prior) / (n + _ORB_SHARE_SHRINK))

def _per40_to_poss(per40: dict, league_pace: float) -> dict:
    """A full 40 minutes ≈ league_pace possessions, so per-poss = per-40 / pace."""
    return {s: (per40.get(s, 0.0) / league_pace if league_pace else 0.0) for s in BASE_STATS}


def positional_prior_poss(position: str, league_pace: float, league: str = "WNBA") -> dict:
    table = _WNBA_PER40 if league == "WNBA" else _SL_PER40
    return _per40_to_poss(table.get(position or "", table[""]), league_pace)
