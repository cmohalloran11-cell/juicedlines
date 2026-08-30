"""
Public API — fit a player from real nflverse data, project one game to a distribution per
market, and read any prop market off it.

    from nfl import projections as P
    proj = P.project_player("Ja'Marr Chase", team="CIN")
    P.market_prob(proj, "Receiving Yards", 84.5, "over")
"""

from __future__ import annotations

import re
import time
import unicodedata
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from . import MARKETS
from .config import cfg
from .data import nfl_source
from .data.base import ScheduleGame
from .model import usage as U
from .model import playing_time as PT
from .model import matchup as MU
from .model import environment as ENV
from .model.combo_corr import empirical_combo_corr
from .sim import engine as E

_proj_cache: dict = {}
_PROJ_TTL = 900.0
_league_cache: dict = {}

# nflverse team codes, plus the aliases the books actually publish. LA is the Rams in
# nflverse; PrizePicks/Underdog ship LAR.
_TEAM_ALIASES = {"LAR": "LA", "STL": "LA", "SL": "LA", "JAC": "JAX", "WSH": "WAS",
                 "WFT": "WAS", "ARZ": "ARI", "CLV": "CLE", "BLT": "BAL", "HST": "HOU",
                 "OAK": "LV", "SD": "LAC", "LVR": "LV", "GNB": "GB", "KAN": "KC",
                 "NWE": "NE", "NOR": "NO", "SFO": "SF", "TAM": "TB"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace(".", "").replace("'", "").split())


def norm_team(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    c = str(code).strip().upper()
    return _TEAM_ALIASES.get(c, c)


def current_season(today: Optional[str] = None) -> int:
    """The NFL season a date belongs to. An NFL season is named for the calendar year it
    STARTS in and runs into February, so January/February belong to the previous season."""
    d = datetime.fromisoformat(today) if today else datetime.now(timezone.utc)
    return d.year - 1 if d.month <= 2 else d.year


# ── market resolution (real book labels -> canonical key) ────────────────────────────────

def _resolve_market(label: str) -> Optional[str]:
    l = _norm(label)
    if not l:
        return None
    tokens = set(re.split(r"[^a-z0-9]+", l))
    if any(t in l for t in ("1h", "2h", "1q", "2q", "3q", "4q", "quarter", "half",
                            "first td", "anytime", "longest", "period")):
        return None
    # Season-long futures ("Season Receiving Yards", "Season Pass Tds", "Regular Season
    # Games Started") are a REST-OF-SEASON total, not a single game — the same class of bug
    # pullers.py's MLB/WNBA "SZN"/period sub-league exclusion exists to prevent (see that
    # module's docstring). Found live 2026-08 (production verification): Underdog ships these
    # under the same "NFL" league as real per-game props, distinguished only by this "season"
    # qualifier in the label text, so it has to be caught here rather than at the league-key
    if "season" in tokens:
        return None
    # Touchdown props: passing TDs ARE modelled; rushing/receiving TDs deliberately are not
    # (the TD probability model isn't proven), so they resolve to None and get skipped rather
    # than mis-mapped onto a yardage market.
    if "td" in l.split() or "touchdown" in l or "tds" in l:
        if "pass" in l:
            return "pass_tds"
        return None
    if "fantasy" in l:
        return None
    # Books abbreviate the combo prop ("Rush+Rec Yards"), so the bare tokens count too — but
    # only as whole tokens, or "rec" would match inside "recent"/"record".
    rush = "rush" in l or "rush" in tokens
    rec = "receiv" in l or "reception" in l or bool(tokens & {"rec", "recs", "rcv"})
    if rush and rec and "yard" in l:
        return "rush_rec_yards"
    if "yard" in l:
        if "pass" in l:
            return "pass_yards"
        if rush:
            return "rush_yards"
        if rec:
            return "rec_yards"
        return None
    if "completion" in l:
        return "completions"
    if "attempt" in l:
        if "pass" in l:
            return "pass_attempts"
        if rush:
            return "rush_attempts"
        return None
    if "carr" in l:
        return "rush_attempts"
    if "target" in l:
        return "targets"
    if "reception" in l or l == "receptions":
        return "receptions"
    return None


# Which markets each position's projection is meaningful for. A WR "Pass Attempts" line is
# not a real market; returning None keeps it unprojected rather than shipping a ~0 projection
# that would read as a huge Under edge.
_POSITION_MARKETS = {
    "QB": {"pass_yards", "pass_attempts", "completions", "pass_tds", "rush_yards",
           "rush_attempts"},
    "RB": {"rush_yards", "rush_attempts", "receptions", "rec_yards", "targets",
           "rush_rec_yards"},
    "WR": {"receptions", "rec_yards", "targets", "rush_yards", "rush_attempts",
           "rush_rec_yards"},
    "TE": {"receptions", "rec_yards", "targets", "rush_rec_yards"},
}


def supports(position: str, market: str) -> bool:
    return market in _POSITION_MARKETS.get((position or "").upper(), set())


# ── league data (fetched once, shared by every player on the board) ──────────────────────

def league_data(season: Optional[int] = None) -> dict:
    """Every release this engine needs for one season plus the previous one, fetched once and
    memoized. `weeks` covers both seasons so a player's fit spans an offseason; `schedule`,
    `depth` and `rosters` are current-season only."""
    season = season or current_season()
    hit = _league_cache.get(season)
    if hit and time.time() - hit[0] < _PROJ_TTL:
        return hit[1]
    src = nfl_source()
    lookback = int(cfg("model", "lookback_seasons", default=2))
    weeks, snaps = [], []
    for s in range(season - lookback + 1, season + 1):
        weeks.extend(src.player_weeks(s))
        snaps.extend(src.snap_counts(s))
    data = {
        "season": season,
        "weeks": weeks,
        "snaps": snaps,
        "snap_index": U.snap_index(snaps),
        "schedule": src.schedule(season),
        "depth": src.depth_charts(season),
        "rosters": src.rosters(season),
    }
    data["priors"] = U.positional_priors(weeks, snaps)
    data["defense"] = MU.fit_defense([w for w in weeks if w.season == season] or weeks)
    data["depth_rank"] = {(d.team, _norm(d.player)): d.rank for d in data["depth"] if d.rank}
    data["positions"] = {}
    for w in weeks:
        data["positions"][_norm(w.player)] = w.position
    for d in data["depth"]:
        data["positions"].setdefault(_norm(d.player), d.position)
    for r in data["rosters"]:
        data["positions"].setdefault(_norm(r.player), r.position)
    comp = sum(w.completions for w in weeks if w.position == "QB")
    tds = sum(w.pass_tds for w in weeks if w.position == "QB")
    data["td_per_completion"] = (tds / comp) if comp > 500 else None
    _league_cache[season] = (time.time(), data)
    return data


def clear_cache() -> None:
    _proj_cache.clear()
    _league_cache.clear()


def resolve_position(name: str, hint: Optional[str] = None,
                     data: Optional[dict] = None) -> Optional[str]:
    pos = (data or {}).get("positions", {}).get(_norm(name))
    if pos:
        return pos
    h = (hint or "").upper()
    return h if h in _POSITION_MARKETS else None


# ── fit + project one player ─────────────────────────────────────────────────────────────

def _confidence(eff_games: float) -> str:
    if eff_games >= cfg("confidence", "high", default=8):
        return "high"
    if eff_games >= cfg("confidence", "medium", default=3):
        return "medium"
    return "low"


def project_player(name: str, team: Optional[str] = None, position: Optional[str] = None,
                   game: Optional[ScheduleGame] = None, season: Optional[int] = None,
                   n: Optional[int] = None, rng=None) -> Optional[dict]:
    """One player-game projection. None when the player can't be resolved to a position at all
    (no stat history, no depth-chart entry, no roster row) — that is a genuine "we know
    nothing", and projecting it would be inventing a player."""
    data = league_data(season)
    nname = _norm(name)
    team = norm_team(team)
    pos = resolve_position(name, position, data)
    if pos is None:
        return None

    ck = (nname, team, pos, game.game_id if game else None, n)
    hit = _proj_cache.get(ck)
    if hit and time.time() - hit[0] < _PROJ_TTL:
        return hit[1]

    weeks = sorted([w for w in data["weeks"] if _norm(w.player) == nname],
                   key=lambda w: (w.season, w.week), reverse=True)
    team = team or (weeks[0].team if weeks else None)

    priors = U.prior_for(pos, data["priors"])
    fit = (U.fit_usage(weeks, data["snap_index"], pos, priors) if weeks
           else U.prior_only_usage(pos, priors))
    fit.player = fit.player or name
    fit.team = fit.team or (team or "")

    environment = ENV.game_environment(game, team or "")
    depth_rank = data["depth_rank"].get((team, nname)) if team else None

    # Availability has to be counted against the TEAM's games, not the player's own rows:
    # a missed game leaves no player row at all, so counting only his own rows would make
    # every player look 100% available. The window is his most recent 17 appearances.
    played = {(w.season, w.week) for w in weeks[:17]}
    team_played = ({(w.season, w.week) for w in data["weeks"]
                    if w.team == team and (w.season, w.week) >= min(played)}
                   if played and team else set())
    pt = PT.project_playing_time(fit.snap_sample, pos, depth_rank,
                                 environment.expected_team_snaps,
                                 active_games=len(played),
                                 team_games=len(team_played) or None)

    defense = data["defense"].get(environment.opponent) if environment.opponent else None
    multipliers = MU.matchup_multipliers(defense)

    combo_corr = empirical_combo_corr(weeks) if pos == "RB" else None
    sim = E.simulate(fit, pt, environment, multipliers, n=n, rng=rng, combo_corr=combo_corr,
                     td_per_completion=data.get("td_per_completion"))

    snap_lo, snap_hi = pt.snap_share_range()
    plays = environment.expected_team_snaps
    dropback = environment.expected_dropback_share

    # Real Monte Carlo output, not re-derived from the Beta shape alone: `sim["snaps"]` already
    # carries BOTH stages of playing-time uncertainty (the team-snaps draw and the snap-share
    # draw — see sim/engine.py's module docstring), so its own percentiles are the honest
    # distribution, asymmetric where the underlying Beta is (a low-mean share is right-skewed —
    # see model/playing_time.py's PlayingTime.beta_params) rather than a forced-symmetric range.
    snap_percentiles = E.summary(sim["snaps"]) if "snaps" in sim else None

    proj = {
        "player": fit.player, "team": team, "position": pos,
        "season": data["season"],
        "usage": fit, "playing_time": pt, "environment": environment,
        "matchup": multipliers, "opponent_defense": defense,
        "pass_rush_matchup": MU.pass_rush_pressure(defense),
        "depth_rank": depth_rank,
        "expected_snaps": pt.expected_snaps,
        "snap_range": ([round(snap_lo * plays, 1), round(snap_hi * plays, 1)]
                       if pt.expected_snaps is not None else None),
        "expected_routes": (round(U.expected_pass_play_snaps(pt.expected_snaps, dropback), 1)
                            if pos in ("RB", "WR", "TE") and pt.expected_snaps is not None
                            else None),
        "expected_routes_basis": ("pass_play_snaps_upper_bound" if pos in ("RB", "WR", "TE")
                                  else None),
        "expected_targets": round(float(sim["targets"].mean()), 2),
        "expected_carries": round(float(sim["rush_attempts"].mean()), 2),
        "expected_pass_attempts": round(float(sim["pass_attempts"].mean()), 2),
        "red_zone_opportunities": U.red_zone_opportunities(fit),
        "n_games": fit.n_games, "eff_games": fit.eff_games,
        "sample_weight": fit.sample_weight,
        "confidence": _confidence(fit.eff_games),
        "snap_percentiles": snap_percentiles,
        "sim": sim,
    }
    _proj_cache[ck] = (time.time(), proj)
    return proj


# ── read markets ─────────────────────────────────────────────────────────────────────────

def market_dist(proj: dict, label: str) -> Optional[np.ndarray]:
    key = _resolve_market(label)
    if key is None or not supports(proj.get("position", ""), key):
        return None
    return E.market_array(proj["sim"], key)


def market_prob(proj: dict, label: str, line: float, side: str = "over") -> Optional[float]:
    arr = market_dist(proj, label)
    if arr is None:
        return None
    p = E.prob_over(arr, float(line))
    return round(p if side.lower() in ("over", "higher", "yes") else 1.0 - p, 4)


def market_summary(proj: dict, label: str) -> Optional[dict]:
    arr = market_dist(proj, label)
    return E.summary(arr) if arr is not None else None
