"""
Public API — fit a player from live data, project the game to a distribution per
stat, and read any prop market off it with an over/under probability + confidence.

    from basketball import projections as P
    proj = P.project_player("WNBA", "Aliyah Boston")
    P.market_prob(proj, "Rebounds", 8.5, "over")   # -> 0.61
"""

from __future__ import annotations

import time
import unicodedata

import numpy as np

from . import BASE_STATS, COMBOS
from .config import cfg, league_cfg
from .data import gamelog_source, background_source
from .model import rates as R
from .model import priors as PR
from .model import minutes as MIN
from .model.pace import matchup_pace
from .sim import engine as E

# PrizePicks/Underdog "Fantasy Score" weights (standard DFS scoring).
_FANTASY_W = {"pts": 1.0, "reb": 1.2, "ast": 1.5, "stl": 3.0, "blk": 3.0, "to": -1.0}

_proj_cache: dict = {}          # (league, player_id) -> (ts, projection)
_PROJ_TTL = 600.0


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace(".", "").replace("'", "").split())


# ── market resolution (real book labels → canonical key) ──────────────────────

def _resolve_market(label: str) -> str | None:
    l = _norm(label)
    if not l:
        return None
    # period / quarter / half markets are not modelled
    if any(t in l for t in ("period", "1h", "2h", "1q", "2q", "3q", "4q", "quarter", "half", "first basket")):
        return None
    # markets we don't simulate → skip rather than mis-map onto a modelled stat
    # (attempts, 2-pointers, split rebounds, FG/FT, double-doubles, minutes).
    if any(t in l for t in ("attempt", "offensive", "defensive", "two point", "2 point",
                            "2pt", "2-pt", "free throw", "fg ", "field goal",
                            "double double", "minutes")):
        return None
    if "fantasy" in l:
        return "fantasy"
    # combos first (before the single-stat substrings)
    if ("pts" in l or "point" in l) and ("reb" in l or "rebound" in l) and ("ast" in l or "assist" in l):
        return "pra"
    if ("pts" in l or "point" in l) and ("reb" in l or "rebound" in l):
        return "pr"
    if ("pts" in l or "point" in l) and ("ast" in l or "assist" in l):
        return "pa"
    if ("reb" in l or "rebound" in l) and ("ast" in l or "assist" in l):
        return "ra"
    if ("blk" in l or "block" in l) and ("stl" in l or "steal" in l):
        return "stocks"
    if "steals" in l and "block" in l:
        return "stocks"
    # singles
    if "three" in l or "3-pt" in l or "3 pt" in l or "3pt" in l or "3-point" in l or "threes" in l:
        return "3pm"
    if "rebound" in l:
        return "reb"
    if "assist" in l:
        return "ast"
    if "block" in l:
        return "blk"
    if "steal" in l:
        return "stl"
    if "turnover" in l:
        return "to"
    if "point" in l or l == "pts":
        return "pts"
    return None


# ── fit + project one player ──────────────────────────────────────────────────

def resolve(league: str, name: str):
    return gamelog_source().players(league).get(_norm(name))


def _confidence(eff_games: float, sample_weight: float, league: str) -> str:
    hi, md = cfg("confidence", "high"), cfg("confidence", "medium")
    if league == "NBA Summer League":
        # wide by design — cap at medium; low unless a few games + real sample weight
        return "medium" if (eff_games >= 4 and sample_weight >= 0.12) else "low"
    if eff_games >= hi:
        return "high"
    if eff_games >= md:
        return "medium"
    return "low"


def project_player(league: str, name: str, news_minutes: float | None = None,
                   n: int | None = None, rng=None):
    ref = resolve(league, name)
    if not ref:
        return None
    ck = (league, ref.id)
    hit = _proj_cache.get(ck)
    if hit and news_minutes is None and time.time() - hit[0] < _PROJ_TTL:
        return hit[1]

    lc = league_cfg(league)
    game_len = lc.get("game_minutes", 40)
    src = gamelog_source()
    lg_pace = src.league_pace(league)
    games = src.gamelog(league, ref.id)
    for g in games:                                    # gamelog omits identity fields
        g.player, g.team, g.team_id = ref.name, ref.team, ref.team_id

    # prior (the league hook)
    bg = None
    if league == "NBA Summer League":
        try:
            bg = background_source().background(ref.name)
        except Exception:
            bg = None
        if bg:
            prior_poss, _ = PR.translated_prior_poss(bg, lg_pace)
        else:
            prior_poss = PR.positional_prior_poss(ref.position, lg_pace, league)
    else:
        prior_poss = PR.positional_prior_poss(ref.position, lg_pace, league)

    # rates (shrunk toward prior)
    if games:
        rates = R.fit_rates(games, league, prior_poss, game_len, lg_pace,
                            lc.get("shrink_poss", 300), cfg("model", "recency_halflife"))
    else:
        rates = R.prior_only_rates(league, prior_poss)
    rates.player = ref.name

    # minutes (own component)
    proj_min, min_sd = MIN.project_minutes(
        rates.minutes_sample, league, lc.get("minutes_shrink_games", 4),
        lc.get("min_sd_frac", 0.15), background=bg, news_minutes=news_minutes)

    pace = matchup_pace(lg_pace)                        # v1: league baseline
    sim = E.simulate(rates, proj_min, min_sd, pace, lc.get("pace_sd_frac", 0.06),
                     game_len, lc.get("disp", 0.12), n=n or cfg("model", "n_sims"), rng=rng)

    proj = {
        "player": ref.name, "team": ref.team, "position": ref.position, "league": league,
        "proj_minutes": proj_min, "minutes_sd": min_sd, "pace": pace,
        "eff_games": rates.eff_games, "n_games": rates.n_games,
        "sample_weight": rates.sample_weight,
        "confidence": _confidence(rates.eff_games, rates.sample_weight, league),
        "prior_source": ("translated" if bg else "positional"),
        "sim": sim, "rates": rates,
    }
    if news_minutes is None:
        _proj_cache[ck] = (time.time(), proj)
    return proj


# ── read markets ──────────────────────────────────────────────────────────────

def _fantasy_array(sim: dict) -> np.ndarray:
    return sum(w * sim[s] for s, w in _FANTASY_W.items() if s in sim)


def market_dist(proj: dict, label: str):
    key = _resolve_market(label)
    if key is None:
        return None
    if key == "fantasy":
        return _fantasy_array(proj["sim"])
    return E.market_array(proj["sim"], key)


def market_prob(proj: dict, label: str, line: float, side: str = "over"):
    arr = market_dist(proj, label)
    if arr is None:
        return None
    p_over = E.prob_over(arr, float(line))
    return round(p_over if side.lower() in ("over", "higher", "yes") else 1.0 - p_over, 4)
