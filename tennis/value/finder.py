"""Odds → implied probability, and edge / EV vs the model."""

from __future__ import annotations


def implied_prob(american: float) -> float:
    n = float(american)
    return (-n) / ((-n) + 100.0) if n < 0 else 100.0 / (n + 100.0)


def to_decimal(american: float) -> float:
    n = float(american)
    return 1.0 + (100.0 / (-n) if n < 0 else n / 100.0)


def value_row(player, market, line, side, price, model_prob, confidence=None, team=None) -> dict:
    ip = implied_prob(price)
    dec = to_decimal(price)
    return {
        "player": player, "market": market, "line": line, "side": side, "price": price,
        "model_prob": round(model_prob, 4), "implied_prob": round(ip, 4),
        "edge": round(model_prob - ip, 4), "ev": round(model_prob * dec - 1.0, 4),
        "confidence": confidence,
    }
