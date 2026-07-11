"""
Board integration — attach model projections to live PrizePicks/Underdog tennis lines.

Groups the board's tennis props into matches (player + opponent from the line's
matchup), projects each match once with the serve/return + Elo model, and writes
model_proj / model_prob / confidence onto every line whose market we model.

Market anchoring: the mirror history is stale (~2016–22), so current players are
thin and the raw model over-projects. The projected mean is blended toward the
market's standard line by a `trust` from the projection's confidence — a
well-covered player is ~pure model, a thin/unknown one defers to the market line
(the same fallback the soccer/basketball models use). The model's distribution
still prices the over/under (and demon/goblin variants) around that mean.

Heuristics (flagged refinements): tour is picked by which fitted model (ATP/WTA) has
the players; best-of is inferred from the match's games line (>30 ⇒ BO5); surface
defaults to Hard until a live surface source is wired.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import projections as P
from .projections import _norm, _model

_TRUST = {"high": 1.0, "medium": 0.5, "low": 0.15}


def _pick_tour(a: str, b: str) -> str:
    """ATP if the players are found in the ATP model, else WTA, else ATP."""
    for tour in ("ATP", "WTA"):
        idx = _model(tour)["name"]
        na, nb = _norm(a), _norm(b)
        if na in idx or nb in idx or na.split()[-1:] == nb.split()[-1:]:
            if na in idx or nb in idx:
                return tour
    return "ATP"


def _best_of(lines) -> int:
    """Infer format from a games-total line in the match (BO5 games run high)."""
    for l in lines:
        key, per = P._resolve_market(l.get("stat_type") or "")
        if key == "total_games" and l.get("line"):
            return 5 if float(l["line"]) > 30 else 3
    return 3


def attach_tennis(lines: list[dict], surface: str = "Hard") -> int:
    """Attach model_proj/model_prob to live tennis lines. Returns count projected."""
    tlines = [l for l in lines if l.get("sport") == "Tennis"
              and l.get("player") and l.get("line") is not None and l.get("matchup")]
    if not tlines:
        return 0

    matches: dict = {}
    for l in tlines:
        key = frozenset({_norm(l["player"]), _norm(l["matchup"])})
        matches.setdefault(key, []).append(l)

    done = 0
    cache: dict = {}
    for mlines in matches.values():
        a, b = mlines[0]["player"], mlines[0]["matchup"]
        ck = frozenset({_norm(a), _norm(b)})
        if ck not in cache:
            try:
                cache[ck] = P.project_match(_pick_tour(a, b), a, b, surface, best_of=_best_of(mlines))
            except Exception:
                cache[ck] = None
        res = cache[ck]
        if not res:
            continue
        trust = _TRUST.get(res.get("confidence"), 0.3)

        # group the match's lines by (player, market) so the mean anchors to the
        # market's standard line for that specific prop.
        groups: dict = defaultdict(list)
        for l in mlines:
            key, _pp = P._resolve_market(l.get("stat_type") or "")
            if key is None:
                continue
            groups[(_norm(l["player"]), key)].append(l)

        for _gk, glines in groups.items():
            label = glines[0].get("stat_type") or ""
            player = glines[0]["player"]
            try:
                arr = P.market_dist(res, player, label)
            except Exception:
                arr = None
            if arr is None:
                continue
            model_mean = float(np.mean(arr))
            std = [float(l["line"]) for l in glines if (l.get("odds_type") or "standard") == "standard"]
            anchor = float(np.median(std)) if std else float(np.median([float(l["line"]) for l in glines]))
            blended = trust * model_mean + (1.0 - trust) * anchor
            if trust < 0.2:             # thin/unknown player → defer fully to the market line
                blended = anchor

            for l in glines:
                line = float(l["line"])
                # per-line guard: if the projection is >1.5x THIS line, the line is a
                # partial-game prop (set/period) sharing the label → price it off the line.
                center = line if (line > 0 and blended > 1.5 * line) else blended
                arr_line = arr + (center - model_mean)
                l["model_prob"] = round(float((arr_line > line).mean()), 4)
                l["model_proj"] = round(center, 2)
                l["model_edge"] = round(center - line, 1)
                l["proj_kind"] = "tennis"
                l["model_n"] = res["eff_matches"]
                l["tennis_confidence"] = res["confidence"]
                done += 1
    return done
