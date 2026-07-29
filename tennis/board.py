"""
Board integration — attach model projections to live PrizePicks/Underdog tennis lines.

Groups the board's tennis props into matches (player + opponent from the line's
matchup), projects each match once with the serve/return + Elo model, and writes
model_proj / model_prob / confidence onto every line whose market we model.

Market anchoring: the mirror history is stale (~2016–22), so current players are
thin and the raw model over-projects. The projected mean is blended toward the
market's standard line by a `trust` from the projection's confidence — a
well-covered player is ~pure model, a thin/unknown one defers to the market line
(the same fallback the basketball model uses). The model's distribution
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
from .data import espn as _espn

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


def _live_surface_lookup(tour: str) -> dict[frozenset, str]:
    """{frozenset({normA, normB}): surface} from ESPN's live schedule — real per-match
    surface instead of the old universal "Hard" default (the README's top-flagged gap).
    Best-effort: an ESPN outage just means matches fall back to the `surface` param."""
    out: dict = {}
    try:
        matches = _espn.upcoming(tour, days_ahead=2) + _espn.completed_matches(tour, days_back=2)
        for m in matches:
            if m.get("a_name") and m.get("b_name"):
                out[frozenset({_norm(m["a_name"]), _norm(m["b_name"])})] = m["surface"]
    except Exception:
        pass
    return out


def _recent_match_counts(tour: str, days: int = 7) -> dict[str, int]:
    """{normalized name: matches played in the last `days`} from ESPN — the fatigue
    signal (spec: "matches played in last 7 days"). Best-effort; {} on an ESPN outage
    means no fatigue adjustment is applied, not a crash."""
    out: dict[str, int] = defaultdict(int)
    try:
        for m in _espn.completed_matches(tour, days_back=days):
            out[_norm(m["a_name"])] += 1
            out[_norm(m["b_name"])] += 1
    except Exception:
        pass
    return out


def _fatigue_penalty(n_recent: int) -> float:
    """Small serve-rate (spw) penalty per match beyond the first played in the last 7
    days — capped, since this is a real but modest effect (a AAA travel/recovery signal
    would refine this further; not available from the free ESPN feed)."""
    return -min(0.02, 0.004 * max(0, n_recent - 1))


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
    surf_cache: dict = {}
    fatigue_cache: dict = {}
    for mlines in matches.values():
        a, b = mlines[0]["player"], mlines[0]["matchup"]
        ck = frozenset({_norm(a), _norm(b)})
        tour = _pick_tour(a, b)
        if tour not in surf_cache:
            surf_cache[tour] = _live_surface_lookup(tour)
        if tour not in fatigue_cache:
            fatigue_cache[tour] = _recent_match_counts(tour)
        match_surface = surf_cache[tour].get(ck, surface)
        fatigue_a = _fatigue_penalty(fatigue_cache[tour].get(_norm(a), 0))
        fatigue_b = _fatigue_penalty(fatigue_cache[tour].get(_norm(b), 0))
        if ck not in cache:
            try:
                cache[ck] = P.project_match(tour, a, b, match_surface, best_of=_best_of(mlines),
                                            fatigue_a=fatigue_a, fatigue_b=fatigue_b)
            except Exception:
                cache[ck] = None
        res = cache[ck]
        if not res:
            continue
        # Model agreement (spec: a real confidence factor, not just sample size) dampens
        # trust when the sim/Elo/analytic win-prob estimates genuinely disagree — e.g. a
        # live-Elo-only player (zero point-model history) vs a well-covered veteran.
        trust = _TRUST.get(res.get("confidence"), 0.3) * (0.7 + 0.3 * res.get("model_agreement", 1.0))

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
            anchored = trust < 0.2      # thin/unknown player → defer fully to the market line
            if anchored:
                blended = anchor

            for l in glines:
                line = float(l["line"])
                # Use the blended projection directly — the old per-line guard snapped any
                # projection >1.5x the line down to the line (assuming a partial-game prop),
                # which just flattened legit big edges on low lines. Partial props are handled
                # upstream now, so trust the model.
                center = blended
                arr_line = arr + (center - model_mean)
                # Projection = the MEAN (`center`) — informative and differentiating (a
                # qualifier's games-won mean can be 8.4 while the median sits lower). The
                # direction guarantee (recommendation vs the market) lives in probability
                # (valuation.recommend_side), not in this display field — see the 2026-07-29
                # projection-realism pass (after the mean→median swap made thin-data
                # projections look flatter than they should).
                med = float(np.median(arr_line))
                lo10, hi90 = (float(x) for x in np.percentile(arr_line, [10, 90]))
                l["model_prob"] = round(float((arr_line > line).mean()), 4)
                l["model_proj"] = round(center, 2)
                l["model_median"] = round(med, 2)
                l["model_floor"] = round(max(0.0, lo10), 1)
                l["model_ceiling"] = round(max(hi90, lo10), 1)
                l["model_edge"] = round(center - line, 1)
                # Be honest about what the number IS. Fully anchored → the projection is the
                # market's own median standard line, NOT a model call, and any apparent "edge"
                # is just this variant's distance from that median (demons/goblins sit far off
                # it by design). The data is stale enough that this is currently EVERY tennis
                # line; mislabelling them "tennis" would imply a model that isn't there.
                l["proj_kind"] = "market" if anchored else "tennis"
                l["model_n"] = res["eff_matches"]
                l["tennis_confidence"] = res["confidence"]
                l["surface"] = match_surface
                l["model_agreement"] = res.get("model_agreement")
                l["elo_eff_matches"] = res.get("elo_eff_matches")
                done += 1
    return done
