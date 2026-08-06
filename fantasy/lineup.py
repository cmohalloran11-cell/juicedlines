"""
fantasy.lineup — pure optimal-starting-lineup assignment (phase 2: Lineup Optimizer). Given
already-scored players (for one specific week) and a league's roster_positions, decides who
starts where. No I/O -- matches valuation.py's boundary; the route handler owns fetching the
roster and weekly projections.

Algorithm: fill starting slots most-constrained-first (fewest eligible positions before
FLEX-type slots), each time taking the highest-projected eligible player still available.
Correct for Sleeper's common nested slot taxonomy, where every FLEX-type slot's eligible set
is a strict superset of the plain-position slots it can also hold (FLEX ⊇ {RB,WR,TE},
SUPER_FLEX ⊇ {QB,RB,WR,TE}) -- a plain-position slot is never better served by waiting, since
nothing else can fill it. A league combining multiple NON-nested flex types at once (e.g. both
WRRB_FLEX and REC_FLEX in the same roster) can hit a rare suboptimal tie; a full bipartite
matching would close that gap but is out of scope for this heuristic-first house style (see
fantasy.vor's tiering for the same tradeoff, disclosed rather than hidden).
"""
from __future__ import annotations

from typing import Optional, Sequence

from .vor import BENCH_SLOTS, FLEX_ELIGIBLE


def _eligible_positions(slot: str) -> set[str]:
    return FLEX_ELIGIBLE.get(slot, {slot})


def optimal_lineup(scored_players: list[dict], roster_positions: Sequence[str]) -> dict:
    """scored_players: [{"player_id", "position", "fantasy_points", ...}], already scored for
    the target week under the league's rules (fantasy.scoring.score_players).

    Returns {"starters": [{"slot": str, "player": dict|None}], "bench": [dict],
             "projected_points": float}. `player` is None when no eligible player exists for
    that slot (thin bench/positions) -- an honest empty slot, not a fabricated fill."""
    starting_slots = [s.upper() for s in roster_positions if s.upper() not in BENCH_SLOTS]
    order = sorted(range(len(starting_slots)),
                   key=lambda i: len(_eligible_positions(starting_slots[i])))

    remaining = {p["player_id"]: p for p in scored_players}
    starters: list[Optional[dict]] = [None] * len(starting_slots)
    for idx in order:
        slot = starting_slots[idx]
        eligible = _eligible_positions(slot)
        candidates = [p for p in remaining.values() if p.get("position") in eligible]
        if not candidates:
            starters[idx] = {"slot": slot, "player": None}
            continue
        best = max(candidates, key=lambda p: p["fantasy_points"])
        starters[idx] = {"slot": slot, "player": best}
        del remaining[best["player_id"]]

    projected_points = round(sum(s["player"]["fantasy_points"] for s in starters if s["player"]), 2)
    bench = sorted(remaining.values(), key=lambda p: -p["fantasy_points"])
    return {"starters": starters, "bench": bench, "projected_points": projected_points}


def lineup_delta(optimal: dict, current_starter_ids: Sequence[str],
                 scored_by_id: dict[str, dict]) -> dict:
    """Compare the optimal lineup against a currently-set lineup (Sleeper roster's own
    `starters`), so the route can surface "you're leaving N points on the bench" rather than
    just the optimal lineup in isolation."""
    current_points = round(sum(
        scored_by_id[pid]["fantasy_points"] for pid in current_starter_ids if pid in scored_by_id
    ), 2)
    optimal_ids = {s["player"]["player_id"] for s in optimal["starters"] if s["player"]}
    swaps_in = [pid for pid in optimal_ids if pid not in set(current_starter_ids)]
    swaps_out = [pid for pid in current_starter_ids if pid not in optimal_ids]
    return {
        "current_points": current_points,
        "optimal_points": optimal["projected_points"],
        "points_gained": round(optimal["projected_points"] - current_points, 2),
        "start": swaps_in,
        "sit": swaps_out,
    }
