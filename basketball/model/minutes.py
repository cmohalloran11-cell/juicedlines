"""
Minutes projection — modelled as its OWN component (not buried in the rate model),
because minutes swing counting-stat projections more than anything else and, in
Summer League, are distributed almost arbitrarily by coaches.

WNBA: recency-weighted recent minutes, lightly regressed toward a role baseline.
Summer League: the draft-slot prior dominates the tiny sample, and a `news` hook
lets you inject who's being showcased / resting / just signed a two-way.
"""

from __future__ import annotations

from .priors import draft_minutes_prior

_WNBA_ROLE_BASELINE = 24.0        # midpoint starter/bench; light shrink target
_MINUTES_HALFLIFE = 3.0           # games; minutes track recent role → short half-life


def project_minutes(minutes_sample: list, league: str, minutes_shrink_games: float,
                    min_sd_frac: float, background=None,
                    news_minutes: float | None = None) -> tuple[float, float]:
    """Return (projected_minutes, minutes_sd). `minutes_sample` = ordered minutes,
    most-recent-first. Minutes reflect the player's CURRENT role, so we weight recent
    games much more heavily than the rate model does (a short half-life) — otherwise
    a season-long average lags players whose minutes ramped up late in the year."""
    if news_minutes is not None:                       # injected news wins outright
        return news_minutes, max(2.0, min_sd_frac * news_minutes)

    wsum = wmin = 0.0
    for i, m in enumerate(minutes_sample):
        w = 0.5 ** (i / _MINUTES_HALFLIFE)
        wsum += w
        wmin += w * m

    if league == "NBA Summer League":
        baseline = draft_minutes_prior(background.draft_pick if background else None)
    else:
        baseline = _WNBA_ROLE_BASELINE

    k = minutes_shrink_games
    proj = (wmin + k * baseline) / (wsum + k) if (wsum + k) > 0 else baseline
    return round(proj, 1), round(max(2.0, min_sd_frac * proj), 1)
