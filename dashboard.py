"""
dashboard.py — aggregation for the Juiced dashboard home (spec Vol VI Pt3 Ch 299-300).

Assembles everything the dashboard renders in ONE payload from data that already exists:
the live board cache (projections), the valuation engine (juice/EV/confidence), the CLV
ledger (line/projection movers), and the model/data health reports. No new modelling — it
composes the pieces built in earlier volumes.
"""
from __future__ import annotations

from typing import Any, Optional

import time

import db
import valuation
import model_health
import dataos
import backtest


def _edge_pct(line: dict) -> Optional[float]:
    """The play's expected value as a percent — the honest 'edge %'. valuation.expected_value
    already returns None for demon/goblin (no real payout exposed to price EV against) rather
    than a fabricated even-money-fallback number."""
    ev = valuation.expected_value(line.get("model_prob"), line)
    return None if ev is None else round(ev * 100.0, 1)


_OPP_CACHE: dict = {"t": 0.0, "v": {}}


def _opp_by_abbr() -> dict:
    """team-abbr → {opp, is_home} for today's MLB slate, from the schedule (analytics).
    Memoized 5 min; degrades to {} with no network (opponent just shows as unknown)."""
    if time.time() - _OPP_CACHE["t"] < 300 and _OPP_CACHE["v"]:
        return _OPP_CACHE["v"]
    out: dict = {}
    try:
        import analytics
        by_id = analytics._team_map().get("_by_id", {})
        for tid, info in analytics._today_opponents().items():
            abbr = (by_id.get(tid) or {}).get("abbr")
            if abbr:
                out[abbr] = {"opp": info.get("opp_abbr"), "is_home": info.get("is_home")}
    except Exception:
        out = {}
    _OPP_CACHE.update(t=time.time(), v=out)
    return out


_ACCURACY_CACHE: dict = {"t": 0.0, "v": {}}
_ACCURACY_SPORTS = ("MLB", "WNBA", "Tennis")
# NFL shares one sport+model_version across two real engines (nfl_regular/nfl_preseason —
# see db.stat_gammas's proj_kind docstring), so it can't reuse the plain per-sport lookup
# above without blending a preseason rotation-tier hit rate into a regular-season one.
_ACCURACY_NFL_PROJ_KINDS = ("nfl_regular", "nfl_preseason")


def _accuracy_map() -> dict:
    """{sport: {stat_type_lower: hit_rate}} from the real graded-ledger backtest — the
    per-stat "Historical Accuracy" shown on a play card. NFL is additionally keyed
    {(sport, proj_kind): {...}} for its two engines. Memoized 10 min (backtest.
    current_accuracy scans the ledger, not free to call per-request)."""
    import time as _time
    if _time.time() - _ACCURACY_CACHE["t"] < 600 and _ACCURACY_CACHE["v"]:
        return _ACCURACY_CACHE["v"]
    out: dict = {}
    for s in _ACCURACY_SPORTS:
        try:
            acc = backtest.current_accuracy(s)
            out[s] = {st.lower(): m.get("hit_rate") for st, m in (acc.get("per_stat") or {}).items()
                      if m.get("hit_rate") is not None}
        except Exception:
            out[s] = {}
    for pk in _ACCURACY_NFL_PROJ_KINDS:
        try:
            acc = backtest.current_accuracy("NFL", proj_kind=pk)
            out[("NFL", pk)] = {st.lower(): m.get("hit_rate")
                                for st, m in (acc.get("per_stat") or {}).items()
                                if m.get("hit_rate") is not None}
        except Exception:
            out[("NFL", pk)] = {}
    _ACCURACY_CACHE.update(t=_time.time(), v=out)
    return out


def _drop(line: dict, acc_map: Optional[dict] = None) -> dict:
    opp = _opp_by_abbr().get(line.get("team")) if line.get("sport") == "MLB" else None
    rec = valuation.recommend_side(line.get("model_prob"), line)
    acc = None
    if acc_map is not None:
        acc_key = ((line.get("sport"), line.get("proj_kind"))
                  if line.get("sport") == "NFL" else line.get("sport"))
        acc = (acc_map.get(acc_key) or {}).get((line.get("stat_type") or "").lower())
    return {
        "id": line.get("id"),
        "player": line.get("player"),
        "team": line.get("team"),
        "opponent": opp.get("opp") if opp else None,     # real, from today's MLB schedule
        "isHome": opp.get("is_home") if opp else None,
        "sport": line.get("sport"),
        "stat": line.get("stat_type"),
        "line": line.get("line"),
        "projection": line.get("model_proj"),
        # The median of the same distribution — separate from the mean-based projection on
        # purpose (2026-07-29 projection-realism pass): a low-count skewed stat can have a
        # median of 0 while the mean is a real, informative nonzero number, and showing only
        # one or the other either looks like a broken "PROJ 0" or hides why the model leans
        # Under. The UI shows both.
        "median": line.get("model_median"),
        "edge": line.get("model_edge"),
        "edgePct": _edge_pct(line),
        "probability": line.get("model_prob"),
        "floor": line.get("model_floor"),          # for the drawer's distribution (esp. static)
        "ceiling": line.get("model_ceiling"),
        "juiceScore": valuation.juice_score(line),
        "juiceFactors": valuation.juice_factors(line),   # real per-prop breakdown (2026-07-30 rebuild)
        "confidence": valuation.confidence_score(line),
        "confidenceFactors": valuation.confidence_factors(line),  # real per-prop breakdown (not a constant)
        # The side we'd actually recommend: highest-EV AVAILABLE side (valuation.recommend_side),
        # not just whichever side model_prob favors — a multiplier book can make the less-likely
        # side the better bet, and a side the book doesn't offer is never recommended.
        "side": rec["side"] if rec else None,
        "headshot": line.get("headshot"),
        "teamLogo": line.get("team_logo"),
        "position": line.get("position"),
        "startTime": line.get("start_time"),
        "source": line.get("source"),
        "oddsType": (line.get("odds_type") or "standard").lower(),
        "unpriced": valuation.is_unpriced(line),   # demon/goblin — no Edge %/EV, see valuation.py
        # Projection version metadata (spec Principle 4 reproducibility) — stamped by
        # provenance.stamp_lines (see analytics.enrich_lines) but previously dropped here
        # before reaching the API, so a user could never see which model version actually
        # produced their pick. Same fix as build_static.py's _KEEP for the static board.
        "model_version": line.get("model_version"),
        "data_snapshot": line.get("data_snapshot"),
        "kellyPct": (round(k * 100, 1) if (k := valuation.kelly_fraction(line.get("model_prob"), line)) is not None else None),
        "volatility": valuation.volatility_cv(line),
        "historicalAccuracy": acc,
        # NFL-specific fields (nfl.board.NFL_FIELDS) — None for every other sport via .get(),
        # same zero-risk additive pattern as model_version/data_snapshot above.
        "seasonType": line.get("season_type"),
        "seasonTypeConfirmed": line.get("season_type_confirmed"),
        "expectedSnaps": line.get("expected_snaps"),
        "snapRange": line.get("snap_range"),
        "expectedRoutes": line.get("expected_routes"),
        "expectedRoutesBasis": line.get("expected_routes_basis"),
        "expectedTargets": line.get("expected_targets"),
        "expectedCarries": line.get("expected_carries"),
        "playingTimeConfidence": line.get("playing_time_confidence"),
        "playingTimeProbability": line.get("playing_time_probability"),
        "roleConfidence": line.get("role_confidence"),
        "preseasonRisk": line.get("preseason_risk"),
        "rotationTier": line.get("rotation_tier"),
        "role": line.get("role"),
        "depthChartPosition": line.get("depth_chart_position"),
        "redZoneOpportunities": line.get("red_zone_opportunities"),
        "passRushMatchup": line.get("pass_rush_matchup"),
        "gameTotal": line.get("game_total"),
        "teamTotal": line.get("team_total"),
        "spread": line.get("spread"),
        "weather": line.get("weather"),
        "nflConfidence": line.get("nfl_confidence"),
        "nflConfidenceFactors": line.get("nfl_confidence_factors"),
    }


def _projected(lines: list[dict]) -> list[dict]:
    # Demon/goblin ARE included (the engine projects them fine — 96% coverage, actually higher
    # than standard) — they just carry no Edge %/EV (see valuation.is_unpriced) since PrizePicks
    # doesn't expose their boosted payout, and are always "over" (you can't take Under on a
    # boosted leg — valuation.recommend_side). Everything else must have a real, offered,
    # priceable side (valuation.recommend_side returns non-None) — a prop where the book offers
    # NEITHER side priced (rare on Underdog/Sleeper) is dropped rather than shown as a bet that
    # can't actually be placed.
    out = []
    for l in lines:
        if (l.get("model_proj") is None or l.get("model_prob") is None
                or l.get("line") is None or l.get("lineup_status") == "out"):
            continue
        if valuation.recommend_side(l.get("model_prob"), l) is None:
            continue
        out.append(l)
    return out


def _tile(line: Optional[dict], value: Any, label: str) -> Optional[dict]:
    if not line:
        return None
    rec = valuation.recommend_side(line.get("model_prob"), line)
    return {
        "id": line.get("id"),
        "player": line.get("player"), "team": line.get("team"),
        "stat": line.get("stat_type"), "line": line.get("line"),
        "side": rec["side"] if rec else None,
        "headshot": line.get("headshot"), "value": value, "label": label,
    }


def _movers(limit: int = 5) -> dict:
    rows = db.recent_prop_moves(limit=400)
    line_moves, proj_up, proj_down = [], [], []
    for r in rows:
        lm = (r.get("close_line") or 0) - (r.get("open_line") or 0)
        if abs(lm) > 1e-9:
            line_moves.append({"player": r["player"], "stat": r["stat_type"],
                               "sport": r["sport"], "move": round(lm, 1),
                               "book": r.get("source"), "open_line": r.get("open_line"),
                               "close_line": r.get("close_line")})
        op, cp = r.get("open_proj"), r.get("close_proj")
        if op is not None and cp is not None and abs(cp - op) > 1e-9:
            entry = {"player": r["player"], "stat": r["stat_type"],
                     "sport": r["sport"], "move": round(cp - op, 2)}
            (proj_up if cp > op else proj_down).append(entry)
    proj_up.sort(key=lambda x: x["move"], reverse=True)
    proj_down.sort(key=lambda x: x["move"])
    line_moves.sort(key=lambda x: abs(x["move"]), reverse=True)
    return {"proj_up": proj_up[:limit], "proj_down": proj_down[:limit],
            "line_moves": line_moves[:limit]}


def _steam_moves(limit: int = 8) -> list[dict]:
    """Same player+stat line moving the SAME direction on 2+ distinct books — real
    cross-book synchronized movement, computed from today's already-tracked open/close
    lines (db.recent_prop_moves). Unlike a single-book "line move", this is a genuine signal
    multiple books independently reacted to the same way."""
    rows = db.recent_prop_moves(limit=400)
    groups: dict[tuple, dict] = {}
    for r in rows:
        move = (r.get("close_line") or 0) - (r.get("open_line") or 0)
        if abs(move) < 1e-9:
            continue
        key = (r["player"], r["stat_type"], r["sport"])
        g = groups.setdefault(key, {"player": r["player"], "stat": r["stat_type"],
                                    "sport": r["sport"], "books": []})
        g["books"].append({"book": r.get("source"), "move": round(move, 1)})
    out = []
    for g in groups.values():
        # Same source can log >1 row per player+stat (odds_type variants share this key) —
        # "steam" means DISTINCT books moving together, so dedupe by book before counting.
        by_book: dict = {}
        for b in g["books"]:
            by_book[b["book"]] = b["move"]
        signs = {1 if m > 0 else -1 for m in by_book.values()}
        if len(by_book) >= 2 and len(signs) == 1:
            g["books"] = [{"book": bk, "move": mv} for bk, mv in by_book.items()]
            g["direction"] = "up" if signs.pop() > 0 else "down"
            g["bookCount"] = len(by_book)
            g["avgMove"] = round(sum(by_book.values()) / len(by_book), 1)
            out.append(g)
    out.sort(key=lambda x: abs(x["avgMove"]), reverse=True)
    return out[:limit]


def _most_offered(priced: list[dict], limit: int = 8) -> list[dict]:
    """Props listed on the most distinct books — a real, measured proxy for market interest.
    NOT betting volume (no such feed exists here) — deliberately labeled "Most Widely
    Offered" in the UI rather than "Most Bet", so the number shown always means what it says."""
    seen, out = set(), []
    for l in sorted(priced, key=lambda x: x.get("market_book_count") or 0, reverse=True):
        key = (l.get("player"), l.get("stat_type"))
        if key in seen or not (l.get("market_book_count") or 0) >= 2:
            continue
        seen.add(key)
        out.append({"player": l.get("player"), "stat": l.get("stat_type"),
                    "sport": l.get("sport"), "bookCount": l.get("market_book_count"),
                    "headshot": l.get("headshot")})
        if len(out) >= limit:
            break
    return out


def _upcoming_games(lines: list[dict], limit: int = 6) -> list[dict]:
    """Distinct games from the slate, derived from team + start_time on the lines."""
    seen: dict[tuple, dict] = {}
    for l in lines:
        team, st, sport = l.get("team"), l.get("start_time"), l.get("sport")
        if not team or not st:
            continue
        key = (sport, st, team)
        seen.setdefault(key, {"sport": sport, "team": team, "startTime": st})
    games = list(seen.values())
    games.sort(key=lambda g: g["startTime"] or "")
    return games[:limit]


def build(lines: list[dict], updated_at: Optional[str],
          errors: Optional[dict] = None, sport: Optional[str] = None) -> dict:
    pool = _projected(lines)
    if sport and sport.lower() != "all":
        pool = [l for l in pool if (l.get("sport") or "").lower() == sport.lower()]

    # demon/goblin lines are engineered to sit at an extreme threshold (that's how PrizePicks
    # boosts the payout), so their decisiveness — and therefore juice_score — is structurally
    # inflated by design, not earned. Left in the ranked pool they'd swamp every top-N surface
    # (Daily Juice Drops, the Juice Leader tile, Best Value) with boosted lines instead of real
    # picks. The dashboard's "picks" surfaces are scoped to priced (standard/boosted) props;
    # demon/goblin are still fully browsable — with their real projection — on the Projections
    # page (see dashboard.projections()'s odds_types param), just not ranked alongside priced plays.
    # stat_side_losing (analytics.attach_stat_trust) — the model's own recommended side on
    # this stat has measured below breakeven (e.g. MLB Pitching Outs Under, 33.8% hit rate).
    # Excluded from ranked/recommended surfaces the same way demon/goblin are just above,
    # for a different reason — still fully visible, real, and unhidden on Projections.
    priced = [l for l in pool if not valuation.is_unpriced(l) and not l.get("stat_side_losing")]
    unpriced = [l for l in pool if valuation.is_unpriced(l)]
    demons = [l for l in unpriced if (l.get("odds_type") or "").lower() == "demon"]
    goblins = [l for l in unpriced if (l.get("odds_type") or "").lower() == "goblin"]
    standard = [l for l in priced if (l.get("odds_type") or "standard").lower() == "standard"]

    acc_map = _accuracy_map()
    drops = sorted((_drop(l, acc_map) for l in priced), key=lambda d: d["juiceScore"], reverse=True)

    juice_leader = max(priced, key=lambda l: valuation.juice_score(l), default=None)
    top_edge = max(priced, key=lambda l: (_edge_pct(l) or -999), default=None)
    best_value = max(priced, key=lambda l: abs(l.get("model_edge") or 0), default=None)
    highest_confidence = max(priced, key=lambda l: valuation.confidence_score(l), default=None)
    best_standard = max(standard, key=lambda l: valuation.juice_score(l), default=None)
    best_demon = max(demons, key=lambda l: valuation.juice_score(l), default=None)
    best_goblin = max(goblins, key=lambda l: valuation.juice_score(l), default=None)
    # Safest = confident AND stable (tight distribution relative to its own projection) —
    # a real, different question than "Highest Confidence" alone (that tile can be a
    # decisive-but-volatile prop; this one downweights volatility on top of confidence).
    def _safety(l: dict) -> float:
        conf = valuation.confidence_score(l)
        cv = valuation.volatility_cv(l)
        stability = 100.0 if cv is None else max(0.0, 100.0 - min(cv, 2.0) * 50.0)
        return 0.6 * conf + 0.4 * stability
    safest = max(priced, key=_safety, default=None)
    moves = db.recent_prop_moves(limit=1)
    big_move = moves[0] if moves else None

    roi = db.roi_summary(days=7)

    tiles = {
        "todays_roi": {"label": "Model ROI (Trailing 7D)", **roi},
        "juice_leader": _tile(juice_leader,
                              valuation.juice_score(juice_leader) if juice_leader else None,
                              "Juice Score Leader"),
        "top_edge": _tile(top_edge, _edge_pct(top_edge) if top_edge else None,
                          "Top Projected EV %"),
        "biggest_line_move": ({
            "player": big_move["player"], "stat": big_move["stat_type"],
            "value": round((big_move.get("close_line") or 0) - (big_move.get("open_line") or 0), 1),
            "label": "Biggest Line Move",
        } if big_move else None),
        "best_value": _tile(best_value,
                            (round(abs(best_value.get("model_edge") or 0), 1)
                             if best_value else None),
                            "Best Value (PROJ vs LINE)"),
        "highest_confidence": _tile(highest_confidence,
                                    valuation.confidence_score(highest_confidence) if highest_confidence else None,
                                    "Highest Confidence"),
        "safest_play": _tile(safest, round(_safety(safest), 1) if safest else None, "Safest Play"),
        "best_standard": _tile(best_standard, valuation.juice_score(best_standard) if best_standard else None,
                               "Best Standard Play"),
        "best_demon": _tile(best_demon, valuation.juice_score(best_demon) if best_demon else None,
                            "Best Demon"),
        "best_goblin": _tile(best_goblin, valuation.juice_score(best_goblin) if best_goblin else None,
                             "Best Goblin"),
    }

    movers = _movers()
    movers["steam"] = _steam_moves()

    high_juice = sum(1 for d in drops if d["juiceScore"] >= 80)
    med_juice = sum(1 for d in drops if 60 <= d["juiceScore"] < 80)
    coach_context = {
        "total_priced_props_today": len(priced),
        "props_80plus_juice": high_juice,
        "props_60to79_juice": med_juice,
        "pct_of_pool_80plus_juice": round(100.0 * high_juice / len(priced), 1) if priced else 0,
        "demon_count": len(demons), "goblin_count": len(goblins),
        "trailing_7day_roi_pct": roi.get("roi_pct"),
        "trailing_7day_sample_plays": roi.get("priced_plays"),
        "top_drop_juice_score": drops[0]["juiceScore"] if drops else None,
        "sport_filter": sport or "all",
    }

    return {
        "updated_at": updated_at,
        "tiles": tiles,
        "drops": drops[:12],
        "movers": movers,
        "most_offered": _most_offered(priced),
        "upcoming_games": _upcoming_games(pool),
        "factor_weights": model_health.FACTOR_WEIGHTS,
        "data_health": dataos.health(lines, updated_at, errors),
        "model_health_summary": _model_summary(),
        "coach_context": coach_context,
    }


def projections(lines: list[dict], sport: Optional[str] = None, stat: Optional[str] = None,
                sort: str = "juice", limit: int = 400,
                odds_types: tuple[str, ...] = ("standard", "boosted")) -> list[dict]:
    """Full enriched projected-props feed for the Projections / Props Center / Top Movers
    views: every projected prop with juice/confidence/edge, filterable and sortable.

    odds_types scopes the pool (default: priced standard/boosted only, matching the page's
    long-standing behavior/payload size). Demon/goblin — structurally decisive by design, so
    they'd otherwise dominate a juice-sorted list — are a separate opt-in slice; pass
    odds_types=("demon","goblin") to get that lane instead. See dashboard.build()'s `priced`
    comment for why they're kept out of the default ranked pool."""
    pool = _projected(lines)
    pool = [l for l in pool if (l.get("odds_type") or "standard").lower() in odds_types]
    if sport and sport.lower() != "all":
        pool = [l for l in pool if (l.get("sport") or "").lower() == sport.lower()]
    if stat:
        s = stat.lower()
        pool = [l for l in pool if s in (l.get("stat_type") or "").lower()]
    rows = [_drop(l) for l in pool]
    keys = {
        "juice": lambda d: d["juiceScore"],
        "edge": lambda d: (d["edgePct"] if d["edgePct"] is not None else -999),
        "confidence": lambda d: d["confidence"],
        "projection": lambda d: (d["projection"] or 0),
    }
    rows.sort(key=keys.get(sort, keys["juice"]), reverse=True)
    return rows[:limit]


def injuries(lines: list[dict], limit: int = 80) -> list[dict]:
    """Players the model has flagged as scratched (confirmed out of a posted lineup) or
    returning from a long layoff (IL) — derived from lineup/workload status on the board."""
    out, seen = [], set()
    for l in lines:
        player = l.get("player")
        if not player or player in seen:
            continue
        st, ws = l.get("lineup_status"), l.get("workload_status")
        if st == "out":
            out.append({"player": player, "team": l.get("team"), "sport": l.get("sport"),
                        "status": "OUT",
                        "note": "Confirmed scratch — team posted a full lineup without them",
                        "headshot": l.get("headshot")})
            seen.add(player)
        elif ws == "returning":
            days = l.get("layoff_days")
            out.append({"player": player, "team": l.get("team"), "sport": l.get("sport"),
                        "status": "RETURNING",
                        "note": f"Back from a {days}-day layoff — workload capped"
                                if days else "Returning from the IL — workload capped",
                        "headshot": l.get("headshot")})
            seen.add(player)
    return out[:limit]


def _model_summary() -> dict:
    """Compact model-health line for the dashboard header (full report at /api/model/health)."""
    try:
        card = db.scorecard("MLB")
        return {
            "graded": card.get("graded"),
            "hit_rate": card.get("hit_rate"),
            "beats_breakeven": (None if card.get("hit_rate") is None
                                else card["hit_rate"] > card.get("breakeven", 0.5238)),
        }
    except Exception:
        return {}
