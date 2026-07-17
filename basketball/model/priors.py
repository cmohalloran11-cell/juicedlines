"""
League priors — the shrinkage targets. This is the main hook the two configs override.

WNBA: regress rates toward positional per-40 averages (healthy samples make these a
formality). Summer League: the prior IS the projection — built from translated
pre-NBA production (draft slot + source-league translation factor + archetype),
because the SL sample can't move it.

Translation factors map pre-NBA per-40 → Summer-League per-40 BY SOURCE LEAGUE
(NCAA and G-League translate more reliably than international). These are rough
starting values to be validated/de-biased by the backtest (translation-bias-by-source
check), exactly as the spec calls for.
"""

from __future__ import annotations

from .. import BASE_STATS
from ..data.base import PlayerBackground

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
# baseline when they have little/no pro split yet. Unlike the per-40 rates, the share needs no
# translation factor: orb and drb translate identically (see _TRANSLATION), so the ratio passes
# through untouched — the only open question is how much it REGRESSES, and 0.5 is a deliberate
# "informative but not gospel" stance rather than a calibrated number. Worth revisiting once
# enough players have both a college season and a real pro split to regress one on the other.
_ORB_SHARE_COLLEGE_W = 0.5


def orb_share_prior(position: str, bg: PlayerBackground | None = None) -> float:
    """Baseline offensive share: positional, blended toward the player's college split if known."""
    base = _ORB_SHARE.get((position or "")[:1].upper(), _ORB_SHARE[""])
    if bg is not None:
        o = (bg.rates40 or {}).get("orb", 0.0) or 0.0
        d = (bg.rates40 or {}).get("drb", 0.0) or 0.0
        if o + d > 0:
            return float(_ORB_SHARE_COLLEGE_W * (o / (o + d)) + (1 - _ORB_SHARE_COLLEGE_W) * base)
    return base


def fit_orb_share(games, position: str, bg: PlayerBackground | None = None) -> float:
    """Player's offensive-rebound share, shrunk toward the (possibly college-informed) baseline.

    Uses whatever games carry the split (box scores); a player with none falls back to the
    prior entirely. Shrinking matters: an 8-game sample of a low-rebound guard can read
    0% or 50% on noise alone.
    """
    prior = orb_share_prior(position, bg)
    o = sum(getattr(g, "orb", 0.0) or 0.0 for g in games)
    d = sum(getattr(g, "drb", 0.0) or 0.0 for g in games)
    n = o + d
    if n <= 0:
        return prior
    return float((o + _ORB_SHARE_SHRINK * prior) / (n + _ORB_SHARE_SHRINK))

# Generic Summer-League rookie per-40 baseline (used when no background is found).
# Scoring baselines were raised ~+2/40 (2026-07-16): props list the featured, high-usage
# players — not generic rookies — so the old 13.5-15 pts/40 prior sat well below the board
# population and dragged thin (2-3 game) samples' POINTS ~0.4-0.7 under the line via
# shrinkage (rebounds/assists priors already matched, so only scoring was biased). Reb/ast
# unchanged (calibrated). Full-data players are barely affected (low prior weight).
_SL_PER40 = {
    "G": {"pts": 17.0, "reb": 3.8, "ast": 4.2, "stl": 1.1, "blk": 0.4, "3pm": 1.9, "to": 2.8},
    "F": {"pts": 16.0, "reb": 6.2, "ast": 2.2, "stl": 0.9, "blk": 0.8, "3pm": 1.4, "to": 2.3},
    "C": {"pts": 15.5, "reb": 8.0, "ast": 1.6, "stl": 0.7, "blk": 1.4, "3pm": 0.5, "to": 2.4},
    "":  {"pts": 16.0, "reb": 5.5, "ast": 2.8, "stl": 1.0, "blk": 0.7, "3pm": 1.4, "to": 2.4},
}

# pre-NBA per-40 → Summer-League per-40 translation factor, by source league.
# Calibrated for SUMMER LEAGUE specifically, NOT the NBA: SL is a markedly weaker, faster,
# sloppier league (worse team defense, exhibition intensity), so college/pro production
# translates MUCH more directly than the ~0.75 NCAA→NBA scoring factor would suggest.
# Factors deflate creation the most and inflate turnovers slightly. Validated against the
# opening slate (mean edge ≈ −0.5, i.e. a mild residual under-lean that matches SL props
# being softly padded for hyped rookies) — refine further via backtest as outcomes accrue.
# No orb/drb rows: those are DERIVED from `reb` via an offensive SHARE (see DERIVED_STATS), and
# a share is invariant to a translation factor that scales orb and drb alike.
_TRANSLATION = {
    "NCAA":          {"pts": 0.88, "reb": 0.93, "ast": 0.90, "stl": 0.84, "blk": 0.87, "3pm": 0.83, "to": 1.03},
    "G-League":      {"pts": 0.95, "reb": 0.95, "ast": 0.92, "stl": 0.88, "blk": 0.90, "3pm": 0.90, "to": 1.00},
    "International": {"pts": 0.85, "reb": 0.90, "ast": 0.86, "stl": 0.80, "blk": 0.84, "3pm": 0.80, "to": 1.05},
    "":              {"pts": 0.85, "reb": 0.90, "ast": 0.86, "stl": 0.80, "blk": 0.84, "3pm": 0.80, "to": 1.05},
}


def _per40_to_poss(per40: dict, league_pace: float) -> dict:
    """A full 40 minutes ≈ league_pace possessions, so per-poss = per-40 / pace."""
    return {s: (per40.get(s, 0.0) / league_pace if league_pace else 0.0) for s in BASE_STATS}


def positional_prior_poss(position: str, league_pace: float, league: str = "WNBA") -> dict:
    table = _WNBA_PER40 if league == "WNBA" else _SL_PER40
    return _per40_to_poss(table.get(position or "", table[""]), league_pace)


def draft_minutes_prior(pick: int | None) -> float:
    """Expected SL minutes from draft slot — high picks get showcased."""
    if pick is None:
        return 15.0                 # undrafted / roster hopeful
    if pick <= 5:
        return 30.0
    if pick <= 14:
        return 27.0
    if pick <= 30:
        return 23.0
    if pick <= 45:
        return 20.0
    return 17.0


def translated_prior_poss(bg: PlayerBackground, league_pace: float) -> tuple[dict, dict]:
    """(per-possession prior, translated per-40) from a player's pre-NBA production."""
    fac = _TRANSLATION.get(bg.pre_league or "", _TRANSLATION[""])
    per40 = {s: bg.rates40.get(s, 0.0) * fac.get(s, 0.72) for s in BASE_STATS}
    # showcase bump: top picks get featured → nudge usage-driven stats up
    if bg.draft_pick is not None and bg.draft_pick <= 14:
        bump = 1.12 if bg.draft_pick <= 5 else 1.06
        for s in ("pts", "ast", "3pm"):
            per40[s] *= bump
    return _per40_to_poss(per40, league_pace), per40
