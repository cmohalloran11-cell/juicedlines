"""
Light board analytics for tennis — the model's serve/return rates + Elo + the
projection + line movement, PLUS real recently-completed matches from ESPN's live
scoreboard (see `_recent_matches`). The Sackmann mirror the serve/return rates are FIT
from has no current match results at all (frozen ~2016–22 — see tennis/data/espn.py's own
docstring: as of writing, #1/#2 ATP have zero rows in it), so "recent games" can only ever
come from the live ESPN feed, never from the model's own historical sample.
"""

from __future__ import annotations

from . import projections as P
from .data import espn as _espn


def _pct(x):
    return round(100 * x, 1) if x is not None else None


def _recent_matches(player: str, tour: str, days_back: int = 90, limit: int = 10) -> list[dict]:
    """Real recently-completed singles matches for this player, from ESPN's live scoreboard.
    Matched by normalized name — the same join tennis/board.py's own fatigue-penalty and
    live-surface lookups already use (`_live_surface_lookup`/`_recent_match_counts`) — because
    ESPN's tennis feed carries no cross-provider player id to match the Sackmann-model's own
    id against. A name-collision here (two players sharing a normalized name) would show the
    wrong player's matches; not currently guarded against, same exposure the existing
    board.py lookups already carry."""
    nname = P._norm(player)
    try:
        matches = _espn.completed_matches(tour, days_back=days_back)
    except Exception as exc:
        print(f"[RecentGames] player={player!r} sport=Tennis provider=espn tour={tour} "
             f"-> {type(exc).__name__}: {exc}", flush=True)
        return []
    out = []
    for m in matches:
        is_a = P._norm(m.get("a_name") or "") == nname
        is_b = P._norm(m.get("b_name") or "") == nname
        if not (is_a or is_b):
            continue
        own_games = m["a_games"] if is_a else m["b_games"]
        opp_games = m["b_games"] if is_a else m["a_games"]
        out.append({
            "date": m.get("date"),
            "opponent": m["b_name"] if is_a else m["a_name"],
            "tournament": m.get("tournament"), "round": m.get("round"),
            "surface": m.get("surface"),
            "won": bool(m["a_winner"] if is_a else m["b_winner"]),
            "score": ("-".join(f"{a}/{b}" for a, b in zip(own_games, opp_games))
                     if own_games and opp_games else None),
        })
    out.sort(key=lambda r: r["date"] or "", reverse=True)
    return out[:limit]


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

    # `tour` above is the Sackmann-model's own classification, absent for a player the
    # historical mirror has never seen (a brand-new tour player). Don't let that also skip
    # real recent-match history — try both ESPN tours rather than assume "no model row" means
    # "no matches".
    recent = []
    for t in ([tour] if tour else ("ATP", "WTA")):
        recent = _recent_matches(player, t)
        if recent:
            tour = tour or t
            break

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
        "recent": recent,
        "recent_note": (None if recent else
                        "No completed matches found for this player in the last 90 days on "
                        "ESPN's live tennis scoreboard — the only live source of current "
                        "match results this engine has (the historical model this projection "
                        "is fit from is a frozen ~2016-22 sample, not a live feed). This can "
                        "mean the player genuinely hasn't played recently, or a name mismatch "
                        "between the book's spelling and ESPN's."),
        "movement": [],
    }
