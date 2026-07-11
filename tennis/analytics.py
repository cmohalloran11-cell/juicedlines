"""
Light board analytics for tennis — the model's serve/return rates + Elo + the
projection + line movement. There's no live per-match feed for current players
(the historical sample is ~2016–22), so this is the honest lighter view, rendered
by the soccer-style drawer.
"""

from __future__ import annotations

from . import projections as P


def _pct(x):
    return round(100 * x, 1) if x is not None else None


def analyze(line: dict) -> dict:
    player, opp = line.get("player"), line.get("matchup")
    serve = ret = elo = tour = n = None
    for t in ("ATP", "WTA"):
        try:
            m = P._model(t)
            if P._norm(player) not in m["name"]:
                continue
            pr, pid = P.resolve(t, player)
            serve, ret, tour = _pct(pr.spw), _pct(pr.rpw), t
            n = getattr(pr, "n_matches", None)
            try:
                elo = round(m["elo"].rating(pid, "Hard"))
            except Exception:
                elo = None
            break
        except Exception:
            continue

    bits = []
    if serve is not None:
        bits.append(f"Serve {serve}% · Return {ret}%")
    if elo:
        bits.append(f"Elo {elo} ({tour})")
    note = ("Serve/return + Elo Monte-Carlo model"
            + (" — " + " · ".join(bits) if bits else "")
            + ". Historical sample is ~2016–22, so current / thin players are "
              "anchored to the market line (confidence gates them).")

    return {
        "available": True,
        "sport": "Tennis",
        "player": player,
        "player_type": (f"{tour} player" if tour else "Player"),
        "headshot": line.get("headshot"),
        "team": None,
        "matchup": opp,
        "stat": line.get("stat_type"),
        "line": line.get("line"),
        "over_price": line.get("over_price"),
        "under_price": line.get("under_price"),
        "model_proj": line.get("model_proj"),
        "model_edge": line.get("model_edge"),
        "model_prob": line.get("model_prob"),
        "model_n": line.get("model_n"),
        "proj_kind": line.get("proj_kind"),
        "confidence": line.get("tennis_confidence"),
        "serve_pct": serve, "return_pct": ret, "elo": elo, "tour": tour,
        "note": note,
        "movement": [],
    }
