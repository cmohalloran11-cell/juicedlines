"""
Board analytics for NFL — the recent-game log, hit rate vs the line, and the REAL drivers
behind the projection, returned in the same schema the MLB/WNBA drawer already renders.

`drivers` is the part worth reading closely. Every string in it is generated from a model
field that actually moved this projection, with its number attached — an elevated target
projection says which snap share and which per-snap target rate produced it, and a low
playing-time confidence says which of the three evidence levels it fell back to. Nothing here
is a template that would print regardless of the inputs: a driver that has no real input is
absent, not phrased vaguely.
"""

from __future__ import annotations

from typing import Optional

from . import projections as P
from .config import cfg

_VIEW_COLS = {
    "QB": ["SNP", "ATT", "CMP", "PYD", "PTD", "CAR", "RYD"],
    "RB": ["SNP", "CAR", "RYD", "TGT", "REC", "RECYD"],
    "WR": ["SNP", "TGT", "REC", "RECYD", "CAR", "RYD"],
    "TE": ["SNP", "TGT", "REC", "RECYD"],
}


def _cells(week, snap_pct: Optional[float], position: str) -> dict:
    snp = f"{snap_pct*100:.0f}%" if snap_pct is not None else "—"
    base = {
        "SNP": snp, "ATT": round(week.pass_attempts), "CMP": round(week.completions),
        "PYD": round(week.pass_yards), "PTD": round(week.pass_tds),
        "CAR": round(week.carries), "RYD": round(week.rush_yards),
        "TGT": round(week.targets), "REC": round(week.receptions),
        "RECYD": round(week.rec_yards),
    }
    return {c: base[c] for c in _VIEW_COLS.get(position, _VIEW_COLS["WR"])}


def _drivers(proj: dict, line: dict) -> list[str]:
    """The actual reasons this projection is where it is, each carrying its own number."""
    out: list[str] = []
    pt, usage, env = proj["playing_time"], proj["usage"], proj["environment"]
    market = P._resolve_market(line.get("stat_type") or "")

    if pt.expected_snaps is not None:
        lo, hi = pt.snap_share_range()
        out.append(f"Projected for {pt.expected_snaps:.0f} snaps "
                   f"({pt.expected_snap_share*100:.0f}% share, 80% range "
                   f"{lo*100:.0f}-{hi*100:.0f}%) as the {pt.role}.")

    if pt.basis == "own_snap_history":
        out.append(f"Playing time is off his own last {pt.n_games} games of snap counts.")
    elif pt.basis == "depth_chart":
        out.append(f"No snap history for this player, so playing time comes from the measured "
                   f"snap share of a depth-chart rank {pt.depth_rank} "
                   f"{proj['position']} ({pt.expected_snap_share*100:.0f}%).")
    else:
        out.append("Neither a snap history nor a depth-chart entry resolved for this player, "
                   "so the projection sits on the positional average with the widest band.")

    if market in ("rec_yards", "receptions", "targets"):
        rate = usage.opportunity.get("targets_per_snap")
        if rate is not None:
            out.append(f"Target opportunity: {rate:.3f} targets per snap "
                       f"({proj['expected_targets']:.1f} expected targets), separate from "
                       f"{usage.efficiency.get('yards_per_target', 0):.1f} yards per target.")
    if market in ("rush_yards", "rush_attempts", "rush_rec_yards"):
        rate = usage.opportunity.get("carries_per_snap")
        if rate is not None:
            out.append(f"Rush opportunity: {rate:.3f} carries per snap "
                       f"({proj['expected_carries']:.1f} expected carries) at "
                       f"{usage.efficiency.get('yards_per_carry', 0):.2f} yards per carry.")
    if market in ("pass_yards", "pass_attempts", "completions", "pass_tds"):
        rate = usage.opportunity.get("pass_att_per_snap")
        if rate is not None:
            out.append(f"Dropback volume: {proj['expected_pass_attempts']:.1f} expected "
                       f"attempts at {usage.efficiency.get('completion_rate', 0)*100:.1f}% "
                       f"completion rate.")

    if env.basis == "market_priced":
        out.append(f"Game environment: {env.game_total} total, {env.team_total} team total, "
                   f"{env.spread:+.1f} spread — {env.expected_scrimmage_plays:.0f} expected "
                   f"team plays at a {env.expected_dropback_share*100:.0f}% pass rate.")
    else:
        out.append("No market spread/total published for this game, so team volume and "
                   "the pass/rush split sit on the league baseline.")

    if env.weather:
        out.append(f"Conditions: {env.weather}.")

    defense = proj.get("opponent_defense")
    if defense is not None and proj.get("matchup"):
        parts = [f"{k.replace('_', ' ')} x{v:.3f}" for k, v in sorted(proj["matchup"].items())]
        out.append(f"Opponent {defense.team} defense over {defense.games} games: "
                   + ", ".join(parts) + " (shrunk by measured split-half reliability, which "
                   "is low for defensive per-play rates).")

    if usage.n_games == 0:
        out.append("No game history at all for this player, so every rate is the positional "
                   "prior and the distribution is at its widest.")
    else:
        out.append(f"{usage.sample_weight*100:.0f}% of the usage estimate comes from his own "
                   f"{usage.n_games} games rather than the positional prior "
                   f"({usage.eff_games:.1f} recency-weighted games).")

    trust = line.get("trust_weight")
    if trust is not None and trust < 1.0:
        out.append(f"Projection is blended {trust*100:.0f}% model / {(1-trust)*100:.0f}% "
                   f"market line, in proportion to how much real evidence backs it.")
    return out


def analyze(line: dict) -> dict:
    """Player-detail payload for one NFL line."""
    player, team = line.get("player"), P.norm_team(line.get("team"))
    try:
        data = P.league_data()
    except Exception as exc:
        print(f"[RecentGames] player={player!r} sport=NFL provider=nflverse "
             f"-> league_data() failed: {type(exc).__name__}: {exc}", flush=True)
        return {"available": False, "reason": "NFL data source unavailable right now."}

    from .board import _schedule_index, match_game
    game = match_game(line, _schedule_index(data["schedule"]))
    proj = P.project_player(player, team=team, position=line.get("position"), game=game)
    if not proj:
        return {"available": False,
                "reason": f"{player} has no stat history, depth-chart entry or roster row in "
                          f"the NFL data feed, so there is nothing to project from."}

    nname = P._norm(player)
    weeks = sorted([w for w in data["weeks"] if P._norm(w.player) == nname],
                   key=lambda w: (w.season, w.week), reverse=True)
    idx = data["snap_index"]

    label = line.get("stat_type") or ""
    key = P._resolve_market(label)
    line_val = line.get("line")
    position = proj["position"]

    recent = []
    for w in weeks[:12]:
        s = idx.get((w.game_id, w.team, w.player.strip().lower()))
        pv = round(w.stat(key), 1) if key is not None else None
        recent.append({
            "date": None, "week": w.week, "season": w.season, "opp": w.opponent,
            "season_type": w.season_type,
            "prop_val": pv,
            "cleared": (pv is not None and line_val is not None and pv > float(line_val)),
            "cells": _cells(w, s.offense_pct if s else None, position),
        })

    hit_rate = None
    if key is not None and line_val is not None and weeks:
        vals = [w.stat(key) for w in weeks]
        lv = float(line_val)
        over = sum(1 for v in vals if v > lv)
        l5 = vals[:5]
        hit_rate = {
            "stat": label, "line": line_val, "over": over, "n": len(vals),
            "over_pct": round(100 * over / len(vals)),
            "last5_over": sum(1 for v in l5 if v > lv), "last5_n": len(l5),
            "spark": list(reversed(vals[:15])),
            "projection": line.get("model_proj"),
            "prob_over": line.get("model_prob"),
            "method": "per-snap opportunity x efficiency Monte Carlo",
        }

    env = proj["environment"]
    pt = proj["playing_time"]
    return {
        "available": True,
        "sport": "NFL",
        "player": proj["player"],
        "player_type": position,
        "headshot": line.get("headshot"),
        "team": {"abbr": proj["team"]} if proj["team"] else None,
        "opponent": env.opponent,
        "stat": label,
        "line": line_val,
        "hit_rate": hit_rate,
        "recent": recent,
        "view_cols": _VIEW_COLS.get(position, _VIEW_COLS["WR"]),
        "model_proj": line.get("model_proj"),
        "model_edge": line.get("model_edge"),
        "model_prob": line.get("model_prob"),
        "model_n": line.get("model_n"),
        "proj_kind": line.get("proj_kind"),
        "confidence": line.get("nfl_confidence"),
        "confidence_factors": line.get("nfl_confidence_factors"),
        "role": pt.role,
        "playing_time_confidence": pt.confidence,
        "expected_snaps": proj["expected_snaps"],
        "snap_range": proj["snap_range"],
        "expected_routes": proj["expected_routes"],
        "expected_routes_note": (
            "Pass-play snaps, an upper bound on routes run — no free data source publishes "
            "routes run." if proj["expected_routes"] is not None else None),
        "red_zone_opportunities": proj["red_zone_opportunities"],
        "red_zone_note": "Not available: red-zone splits require the play-by-play release, "
                         "which this engine does not pull.",
        "snap_percentiles": proj.get("snap_percentiles"),
        "drivers": _drivers(proj, line),
        "note": "Projection is expected opportunity x efficiency, simulated "
                f"{cfg('model', 'n_sims', default=10000):,} times with uncertainty in team "
                "plays, game script, playing time, opportunity and efficiency; "
                "market-anchored in proportion to how much real evidence backs it.",
    }
