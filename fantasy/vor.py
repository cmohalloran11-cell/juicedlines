"""
fantasy.vor — pure VOR (value over replacement) + tiering, computed from a league's actual
roster requirements and size (never a fixed 12-team/PPR assumption). Everything here is a
function of already-scored players (fantasy_points from scoring.py) plus league shape -- no
I/O, matching valuation.py's boundary.
"""
from __future__ import annotations

from typing import Sequence

# Sleeper roster_positions entries that are drafted starters (excludes bench/IR/taxi).
FLEX_ELIGIBLE: dict[str, set[str]] = {
    "FLEX": {"RB", "WR", "TE"},
    "SUPER_FLEX": {"QB", "RB", "WR", "TE"},
    "REC_FLEX": {"WR", "TE"},
    "WRRB_FLEX": {"RB", "WR"},
}
BENCH_SLOTS = {"BN", "IR", "TAXI"}


def starter_counts(roster_positions: Sequence[str], league_size: int) -> dict[str, float]:
    """Expected number of drafted starters at each base position (QB/RB/WR/TE/K/DEF) across
    the whole league, splitting FLEX-type slots evenly across their eligible positions -- an
    explicit, documented approximation (not a simulation of real draft behavior), same spirit
    as valuation.py preferring a disclosed heuristic over a hidden model."""
    counts: dict[str, float] = {}
    for slot in roster_positions:
        slot = slot.upper()
        if slot in BENCH_SLOTS:
            continue
        if slot in FLEX_ELIGIBLE:
            eligible = FLEX_ELIGIBLE[slot]
            for pos in eligible:
                counts[pos] = counts.get(pos, 0.0) + 1.0 / len(eligible)
        else:
            counts[slot] = counts.get(slot, 0.0) + 1.0
    return {pos: n * league_size for pos, n in counts.items()}


def replacement_levels(scored_players: list[dict], roster_positions: Sequence[str],
                       league_size: int) -> dict[str, float]:
    """The projected fantasy_points of the last startable player at each position -- the
    'replacement level' VOR is measured against. A position with zero counted starters (e.g.
    a league with no K slot) has no replacement level and is excluded."""
    counts = starter_counts(roster_positions, league_size)
    by_pos: dict[str, list[float]] = {}
    for p in scored_players:
        by_pos.setdefault(p["position"], []).append(p["fantasy_points"])
    levels: dict[str, float] = {}
    for pos, n in counts.items():
        pts = sorted(by_pos.get(pos, []), reverse=True)
        if not pts:
            continue
        idx = max(min(int(round(n)) - 1, len(pts) - 1), 0)
        levels[pos] = pts[idx]
    return levels


def compute_vor(scored_players: list[dict], roster_positions: Sequence[str],
                league_size: int) -> list[dict]:
    """Attach `vor` (fantasy_points - replacement level for that position) to a copy of every
    scored player, sorted descending. Players at a position with no replacement level (not
    started anywhere in this league's roster shape) get vor=None and sort last."""
    levels = replacement_levels(scored_players, roster_positions, league_size)
    out = []
    for p in scored_players:
        row = dict(p)
        level = levels.get(p["position"])
        row["vor"] = round(p["fantasy_points"] - level, 2) if level is not None else None
        out.append(row)
    out.sort(key=lambda r: (r["vor"] is None, -(r["vor"] or 0)))
    return out


def assign_tiers(vor_ranked: list[dict], gap_threshold: float = 0.75) -> list[dict]:
    """Group into tiers via natural gaps in VOR: a new tier starts whenever the drop to the
    next player exceeds `gap_threshold` standard deviations above the mean consecutive gap --
    an explicit, disclosed rule (not a clustering black box), per the task's "natural gaps"
    ask. Players with vor=None all land in one trailing tier."""
    ranked = [p for p in vor_ranked if p.get("vor") is not None]
    unranked = [p for p in vor_ranked if p.get("vor") is None]

    if len(ranked) < 2:
        for p in ranked:
            p["tier"] = 1
        for p in unranked:
            p["tier"] = 2 if ranked else 1
        return ranked + unranked

    gaps = [ranked[i]["vor"] - ranked[i + 1]["vor"] for i in range(len(ranked) - 1)]
    mean_gap = sum(gaps) / len(gaps)
    variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
    cutoff = mean_gap + gap_threshold * (variance ** 0.5)

    tier = 1
    ranked[0]["tier"] = tier
    for i, gap in enumerate(gaps):
        if gap > cutoff and gap > 0:
            tier += 1
        ranked[i + 1]["tier"] = tier
    for p in unranked:
        p["tier"] = tier + 1
    return ranked + unranked
