"""
fantasy.trade — pure trade evaluation (phase 4: Trade Analyzer). Compares total value on each
side of a proposed trade under the league's actual scoring/roster shape. No I/O -- callers
pass in an already-computed, already-VOR'd board keyed by player_id (see
routes_fantasy._scored_board).
"""
from __future__ import annotations

from typing import Sequence

# Below this absolute VOR difference, call it even rather than implying false precision --
# VOR itself is a projection-driven estimate, not a certainty to the tenth of a point.
EVEN_THRESHOLD = 1.0


def _side(board_by_player: dict[str, dict], ids: Sequence[str]) -> dict:
    players, missing, total = [], [], 0.0
    for pid in ids:
        p = board_by_player.get(pid)
        if p is None:
            missing.append(pid)   # no projection available -- flagged, never valued at zero
            continue
        players.append(p)
        total += p.get("vor") or 0.0
    return {"players": players, "missing_player_ids": missing, "total_vor": round(total, 2)}


def evaluate_trade(board_by_player: dict[str, dict], side_a_ids: Sequence[str],
                   side_b_ids: Sequence[str]) -> dict:
    """VOR, not raw fantasy_points, is the fair basis for a cross-position trade comparison --
    replacement level differs wildly by position, so a low-VOR QB can outscore a high-VOR RB in
    raw points while being worth far less in a real trade."""
    a = _side(board_by_player, side_a_ids)
    b = _side(board_by_player, side_b_ids)
    diff = round(a["total_vor"] - b["total_vor"], 2)
    if abs(diff) < EVEN_THRESHOLD:
        verdict = "Roughly even trade"
    elif diff > 0:
        verdict = f"Side A gains {diff} VOR"
    else:
        verdict = f"Side B gains {-diff} VOR"
    return {"side_a": a, "side_b": b, "vor_diff": diff, "verdict": verdict}
