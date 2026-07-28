"""
dashboard.py — aggregation for the Juiced dashboard home (spec Vol VI Pt3 Ch 299-300).

Assembles everything the dashboard renders in ONE payload from data that already exists:
the live board cache (projections), the valuation engine (juice/EV/confidence), the CLV
ledger (line/projection movers), and the model/data health reports. No new modelling — it
composes the pieces built in earlier volumes.
"""
from __future__ import annotations

from typing import Any, Optional

import db
import valuation
import model_health
import dataos


def _edge_pct(line: dict) -> Optional[float]:
    """The play's expected value as a percent — the honest 'edge %'."""
    ev = valuation.expected_value(line.get("model_prob"), line)
    return None if ev is None else round(ev * 100.0, 1)


def _drop(line: dict) -> dict:
    return {
        "id": line.get("id"),
        "player": line.get("player"),
        "team": line.get("team"),
        "sport": line.get("sport"),
        "stat": line.get("stat_type"),
        "line": line.get("line"),
        "projection": line.get("model_proj"),
        "edge": line.get("model_edge"),
        "edgePct": _edge_pct(line),
        "probability": line.get("model_prob"),
        "juiceScore": valuation.juice_score(line),
        "confidence": valuation.confidence_score(line),
        "side": "over" if (line.get("model_prob") or 0) >= 0.5 else "under",
        "headshot": line.get("headshot"),
        "teamLogo": line.get("team_logo"),
        "position": line.get("position"),
        "startTime": line.get("start_time"),
        "source": line.get("source"),
    }


def _projected(lines: list[dict]) -> list[dict]:
    return [l for l in lines
            if l.get("model_proj") is not None and l.get("model_prob") is not None
            and l.get("line") is not None
            and (l.get("odds_type") or "standard").lower() in ("standard", "boosted")
            and l.get("lineup_status") != "out"]


def _tile(line: Optional[dict], value: Any, label: str) -> Optional[dict]:
    if not line:
        return None
    return {
        "player": line.get("player"), "team": line.get("team"),
        "stat": line.get("stat_type"), "line": line.get("line"),
        "side": "over" if (line.get("model_prob") or 0) >= 0.5 else "under",
        "headshot": line.get("headshot"), "value": value, "label": label,
    }


def _movers(limit: int = 5) -> dict:
    rows = db.recent_prop_moves(limit=400)
    line_moves, proj_up, proj_down = [], [], []
    for r in rows:
        lm = (r.get("close_line") or 0) - (r.get("open_line") or 0)
        if abs(lm) > 1e-9:
            line_moves.append({"player": r["player"], "stat": r["stat_type"],
                               "sport": r["sport"], "move": round(lm, 1)})
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

    drops = sorted((_drop(l) for l in pool), key=lambda d: d["juiceScore"], reverse=True)

    # Tiles — each from a real leader in the pool.
    juice_leader = max(pool, key=lambda l: valuation.juice_score(l), default=None)
    top_edge = max(pool, key=lambda l: (_edge_pct(l) or -999), default=None)
    best_value = max(pool, key=lambda l: abs(l.get("model_edge") or 0), default=None)
    moves = db.recent_prop_moves(limit=1)
    big_move = moves[0] if moves else None

    tiles = {
        "juice_leader": _tile(juice_leader,
                              valuation.juice_score(juice_leader) if juice_leader else None,
                              "Juice Score Leader"),
        "top_edge": _tile(top_edge, _edge_pct(top_edge) if top_edge else None,
                          "Top Projected Edge"),
        "biggest_line_move": ({
            "player": big_move["player"], "stat": big_move["stat_type"],
            "value": round((big_move.get("close_line") or 0) - (big_move.get("open_line") or 0), 1),
            "label": "Biggest Line Move",
        } if big_move else None),
        "best_value": _tile(best_value,
                            (round(abs(best_value.get("model_edge") or 0), 1)
                             if best_value else None),
                            "Best Value (PROJ vs LINE)"),
    }

    return {
        "updated_at": updated_at,
        "tiles": tiles,
        "drops": drops[:12],
        "movers": _movers(),
        "upcoming_games": _upcoming_games(pool),
        "factor_weights": model_health.FACTOR_WEIGHTS,
        "data_health": dataos.health(lines, updated_at, errors),
        "model_health_summary": _model_summary(),
    }


def projections(lines: list[dict], sport: Optional[str] = None, stat: Optional[str] = None,
                sort: str = "juice", limit: int = 400) -> list[dict]:
    """Full enriched projected-props feed for the Projections / Props Center / Top Movers
    views: every projected prop with juice/confidence/edge, filterable and sortable."""
    pool = _projected(lines)
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
