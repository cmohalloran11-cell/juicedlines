"""
Board analytics for WNBA — team logos, recent-game log, and
hit-rate vs the line — returned in the SAME schema the MLB drawer renders, so the
frontend needs no special-casing (drawerHTML routes available non-WC analytics to
the MLB drawer).
"""

from __future__ import annotations

from .data import gamelog_source
from .data.espn import _norm_name
from . import projections as P

# DFS fantasy weights (match projections._FANTASY_W) for the fantasy market.
_FANTASY_W = {"pts": 1.0, "reb": 1.2, "ast": 1.5, "stl": 3.0, "blk": 3.0, "to": -1.0}
_VIEW_COLS = ["MIN", "PTS", "REB", "AST", "STL", "BLK", "3PM"]


def team_asset(league: str, team: str | None) -> dict | None:
    if not team:
        return None
    try:
        return gamelog_source().team_assets(league).get(_norm_name(team))
    except Exception:
        return None


def team_logo(league: str, team: str | None) -> str | None:
    a = team_asset(league, team)
    return a.get("logo") if a else None


def _stat_val(g, key: str) -> float:
    if key == "fantasy":
        return round(sum(w * g.stat(s) for s, w in _FANTASY_W.items()), 1)
    return g.stat(key)


def analyze(line: dict) -> dict:
    league, player = line.get("sport"), line.get("player")
    ref = P.resolve(league, player)
    if not ref:
        return {"available": False, "reason": f"{player} not found in the {league} rosters."}
    src = gamelog_source()
    games = src.gamelog(league, ref.id)
    for g in games:
        g.player, g.team, g.team_id = ref.name, ref.team, ref.team_id

    label = line.get("stat_type") or ""
    key = P._resolve_market(label)
    line_val = line.get("line")
    asset = team_asset(league, ref.team) or team_asset(league, line.get("team"))

    # recent games table (most-recent-first) + per-game prop value
    recent = []
    for g in games[:12]:
        pv = round(_stat_val(g, key), 1) if key is not None else None
        recent.append({
            "date": g.date, "opp": g.opp, "home": None,
            "prop_val": pv,
            "cleared": (pv is not None and line_val is not None and pv > float(line_val)),
            "cells": {"MIN": round(g.minutes), "PTS": round(g.pts), "REB": round(g.reb),
                      "AST": round(g.ast), "STL": round(g.stl), "BLK": round(g.blk),
                      "3PM": round(g.tpm)},
        })

    hit_rate = None
    if key is not None and line_val is not None and games:
        vals = [_stat_val(g, key) for g in games]           # recent-first
        lv = float(line_val)
        over = sum(1 for v in vals if v > lv)
        l5 = vals[:5]
        hit_rate = {
            "stat": label, "line": line_val,
            "over": over, "n": len(vals), "over_pct": round(100 * over / len(vals)),
            "last5_over": sum(1 for v in l5 if v > lv), "last5_n": len(l5),
            "spark": list(reversed(vals[:15])),            # chronological for the sparkline
            "projection": line.get("model_proj"),
            "prob_over": line.get("model_prob"),
            "method": "per-possession model",
        }

    return {
        "available": True,
        "sport": league,
        "player": ref.name,
        "player_type": ref.position or "Player",
        "headshot": line.get("headshot"),
        "team": asset and {"abbr": asset.get("abbr"), "name": asset.get("name"),
                           "logo": asset.get("logo")},
        "stat": label,
        "line": line_val,
        "hit_rate": hit_rate,
        "recent": recent,
        "view_cols": _VIEW_COLS,
        "model_proj": line.get("model_proj"),
        "model_edge": line.get("model_edge"),
        "model_prob": line.get("model_prob"),
        "model_n": line.get("model_n"),
        "proj_kind": line.get("proj_kind"),
        "confidence": line.get("bball_confidence"),
        "note": "Recent ESPN box scores; projection is the per-possession model "
                "(rates × minutes × pace), market-anchored when the sample is thin.",
    }
