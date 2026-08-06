"""
fantasy.draft_state — pure live-draft logic: best-available, positional-run detection, and
need-weighted recommendations from an already-polled pick list. The route handler owns polling
Sleeper's draft-picks endpoint (10s cache) and passes the result in here -- nothing in this
module does I/O, so it's unit-testable without mocking HTTP (same boundary as valuation.py).
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

from .vor import BENCH_SLOTS, FLEX_ELIGIBLE


def remove_drafted(tiered_board: list[dict], picks: Sequence[dict]) -> list[dict]:
    """Drop every player_id present in `picks` (Sleeper draft_picks rows carry a top-level
    player_id) from the board."""
    drafted = {pick.get("player_id") for pick in picks if pick.get("player_id")}
    return [p for p in tiered_board if p.get("player_id") not in drafted]


def positional_runs(picks: Sequence[dict], window: int = 5, run_threshold: int = 3) -> dict[str, int]:
    """Flag a positional run: `run_threshold`+ picks at the same position within the last
    `window` picks (draft order). Returns {position: count} for positions currently running --
    an explicit, disclosed rule, not a statistical model. Sleeper pick rows carry the drafted
    player's position under metadata.position; callers should pass picks pre-annotated with a
    top-level "position" key (the route handler does this from the board lookup)."""
    recent = list(picks)[-window:]
    counts = Counter(p.get("position") for p in recent if p.get("position"))
    return {pos: n for pos, n in counts.items() if n >= run_threshold}


def remaining_needs(roster_positions: Sequence[str], drafted_by_user: Sequence[dict]) -> Counter:
    """Starting-slot needs still open for one team, given what they've already drafted. Bench
    slots don't count as a "need" for recommendation-weighting purposes. FLEX-type slots are
    tracked by slot name (resolved against eligible positions at recommendation time), not
    pre-collapsed into a single base position."""
    needed: Counter = Counter()
    for slot in roster_positions:
        slot = slot.upper()
        if slot in BENCH_SLOTS:
            continue
        needed[slot] += 1
    filled = Counter(p.get("position") for p in drafted_by_user if p.get("position"))
    remaining: Counter = Counter()
    for slot, n in needed.items():
        if slot in FLEX_ELIGIBLE:
            remaining[slot] = n
            continue
        remaining[slot] = max(0, n - filled.get(slot, 0))
    return remaining


def _need_weight(position: str, needs: Counter) -> float:
    if needs.get(position, 0) > 0:
        return 1.15
    for slot, n in needs.items():
        if n > 0 and slot in FLEX_ELIGIBLE and position in FLEX_ELIGIBLE[slot]:
            return 1.08
    return 1.0


def recommend(board: list[dict], roster_positions: Sequence[str],
             drafted_by_user: Sequence[dict], top_n: int = 10) -> list[dict]:
    """Best-available list re-weighted by the user's remaining roster needs: a player at a
    still-open starting position is boosted; a position with zero remaining need (including
    FLEX eligibility) isn't excluded, just not boosted -- bench/depth value still matters."""
    needs = remaining_needs(roster_positions, drafted_by_user)
    weighted = []
    for p in board:
        row = dict(p)
        row["need_weight"] = _need_weight(p.get("position", ""), needs)
        row["recommendation_score"] = round((p.get("vor") or 0.0) * row["need_weight"], 2)
        weighted.append(row)
    weighted.sort(key=lambda r: -r["recommendation_score"])
    return weighted[:top_n]
