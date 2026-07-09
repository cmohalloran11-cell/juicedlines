"""
Value finder — convert sportsbook odds to implied probability and rank props by
(model probability − implied probability), i.e. the biggest mismatches.

edge = model_prob − implied_prob   (positive = model likes it more than the book)
ev   = model_prob × decimal_odds − 1   (expected profit per $1 staked)
"""

from __future__ import annotations


def implied_prob(american: float) -> float:
    """American odds → implied probability (includes the book's vig)."""
    n = float(american)
    return (-n) / ((-n) + 100.0) if n < 0 else 100.0 / (n + 100.0)


def to_decimal(american: float) -> float:
    n = float(american)
    return 1.0 + (100.0 / (-n) if n < 0 else n / 100.0)


def model_prob_for(entry: dict, market: str, line, side: str):
    """Model probability matching an odds line's market/line/side, or None."""
    if market == "goal":
        return (entry.get("goal") or {}).get("anytime")
    if market == "card":
        return (entry.get("card") or {}).get("yes")
    if market in ("sot", "saves"):
        over = ((entry.get(market) or {}).get("over")) or {}
        p = over.get(line)
        if p is None:
            return None
        return p if side == "over" else round(1.0 - p, 4)
    return None


def find_value(sim_result: dict, odds_lines: list) -> list[dict]:
    """Rank the supplied odds lines by model edge vs the book."""
    players = sim_result.get("players", {})
    rows = []
    for o in odds_lines:
        e = players.get(o.player)
        if not e:
            continue
        mp = model_prob_for(e, o.market, o.line, o.side)
        if mp is None:
            continue
        ip = implied_prob(o.price)
        dec = to_decimal(o.price)
        rows.append({
            "player": o.player, "team": e.get("team"), "market": o.market,
            "line": o.line, "side": o.side, "price": o.price, "book": o.book,
            "model_prob": round(mp, 4), "implied_prob": round(ip, 4),
            "edge": round(mp - ip, 4), "ev": round(mp * dec - 1.0, 4),
            "confidence": e.get("confidence"),
        })
    rows.sort(key=lambda r: r["edge"], reverse=True)
    return rows
