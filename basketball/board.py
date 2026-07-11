"""
Board integration — attach model projections to live PrizePicks/Underdog basketball
lines (WNBA + NBA Summer League).

Groups the board's basketball props by (league, player, market), projects each player
once with the shared core, and writes model_proj / model_prob / model_edge / confidence.

Market anchoring: the projected MEAN is blended toward the market's standard line in
proportion to how little real sample the model has (`trust` from sample_weight). A
well-sampled WNBA star is ~pure model (edges preserved); a Summer-League player with
no usable history defers to the market line — the same market-consensus fallback the
soccer model uses — instead of clustering at a generic prior. The model's distribution
SHAPE still prices the over/under (and the demon/goblin variants) around that mean.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import projections as P

_LEAGUES = ("WNBA", "NBA Summer League")

# sample_weight at/above which we trust the model fully (edges preserved). Below it,
# the projection is blended toward the market line proportionally.
_FULL_TRUST_AT = 0.6


def attach_basketball(lines: list[dict]) -> int:
    """Attach projections to live WNBA/Summer-League lines. Returns count projected."""
    blines = [l for l in lines if l.get("sport") in _LEAGUES
              and l.get("player") and l.get("line") is not None]
    if not blines:
        return 0

    proj_cache: dict = {}

    def get_proj(league: str, player: str):
        ck = (league, P._norm(player))
        if ck not in proj_cache:
            try:
                proj_cache[ck] = P.project_player(league, player)
            except Exception:
                proj_cache[ck] = None
        return proj_cache[ck]

    # group by player + market so the mean can be anchored to the market's standard line
    groups: dict = defaultdict(list)
    for l in blines:
        mk = P._resolve_market(l.get("stat_type") or "")
        if mk is None:
            continue
        groups[(l["sport"], P._norm(l["player"]), mk)].append(l)

    done = 0
    for (league, _pnorm, _mk), glines in groups.items():
        proj = get_proj(league, glines[0]["player"])
        if not proj:
            continue
        arr = P.market_dist(proj, glines[0].get("stat_type") or "")
        if arr is None:
            continue
        model_mean = float(np.mean(arr))

        # market anchor = the standard line (book's estimate of the mean); fall back to
        # the median posted line for the market if no standard line is present.
        std = [float(l["line"]) for l in glines if (l.get("odds_type") or "standard") == "standard"]
        anchor = float(np.median(std)) if std else float(np.median([float(l["line"]) for l in glines]))

        trust = min(1.0, proj["sample_weight"] / _FULL_TRUST_AT)
        blended = trust * model_mean + (1.0 - trust) * anchor
        if trust < 0.2:                 # ~no reliable model info → defer fully to the
            blended = anchor            # market (edge≈0, symmetric) rather than a noisy drag

        for l in glines:
            line = float(l["line"])
            # per-line guard: if the full-game projection is >1.5x THIS line, the line is a
            # partial-game prop (1H≈2x, 1Q≈4x) sharing the stat label from a source not keyed
            # by a period league → price it off the line itself (edge≈0). Legit goblins sit
            # ≤~1.35x and demons <1x, so they still price off `blended`.
            center = line if (line > 0 and blended > 1.5 * line) else blended
            arr_line = arr + (center - model_mean)
            l["model_prob"] = round(float((arr_line > line).mean()), 4)
            l["model_proj"] = round(center, 1)
            l["model_edge"] = round(center - line, 1)
            l["proj_kind"] = "basketball"
            l["model_n"] = proj["n_games"]
            l["bball_confidence"] = proj["confidence"]
            done += 1
    return done
