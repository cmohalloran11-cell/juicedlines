"""
Serve/return rate fitting with cold-start shrinkage — the core skill estimates.

For each player: serve-points-won (`spw`) and return-points-won (`rpw`), overall and
per surface, regressed toward the tour baseline via pseudo-counts (thin samples shrink
harder). Surface rates fall back to `overall + surface_shift` when the surface sample
is thin. Also fits ace/DF rates and points-per-service-game (for count props). Every
player carries an effective sample size that drives the confidence/variance value.

ATP and WTA are fit separately — pass one tour's matches at a time; never mix.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date as _date

from ..config import cfg

_HALF_LIFE_DAYS = 730.0   # recency weight: a match 2 years older counts half as much


def _parse_date(s) -> _date | None:
    s = str(s or "").replace("-", "")
    if len(s) == 8 and s.isdigit():
        try:
            return _date(int(s[:4]), int(s[4:6]), int(s[6:8]))
        except ValueError:
            return None
    return None


def _recency_weight(d: _date | None, latest: _date | None) -> float:
    """1.0 for the most recent match in the fitted set, decaying by half every
    `_HALF_LIFE_DAYS` for older ones — "recent surface performance should carry more
    weight than career averages" without discarding the career signal entirely."""
    if d is None or latest is None:
        return 1.0
    return 0.5 ** (max(0, (latest - d).days) / _HALF_LIFE_DAYS)


@dataclass
class TourBaselines:
    tour: str
    spw_avg: float
    rpw_avg: float
    ace_rate_avg: float
    df_rate_avg: float
    pts_per_svgame_avg: float
    surface_spw: dict = field(default_factory=dict)   # surface -> avg spw (for the shift)


@dataclass
class PlayerRates:
    player_id: str
    player: str
    tour: str
    spw: float
    rpw: float
    ace_rate: float
    df_rate: float
    pts_per_svgame: float
    surface_spw: dict = field(default_factory=dict)
    surface_rpw: dict = field(default_factory=dict)
    n_matches: int = 0
    n_by_surface: dict = field(default_factory=dict)
    raw_spw: float | None = None      # pre-shrinkage (diagnostics)

    def eff_matches(self, surface: str | None = None) -> int:
        return self.n_by_surface.get(surface, 0) if surface else self.n_matches


def _shrink(won: float, played: float, prior: float, pseudo: float) -> float:
    return (won + pseudo * prior) / (played + pseudo) if (played + pseudo) else prior


def _latest_date(matches) -> _date | None:
    ds = [d for d in (_parse_date(m.date) for m in matches) if d is not None]
    return max(ds) if ds else None


def baselines(matches, tour: str) -> TourBaselines:
    latest = _latest_date(matches)
    sw = sp = aces = dfs = svg = 0.0
    surf = defaultdict(lambda: [0.0, 0.0])
    for m in matches:
        w = _recency_weight(_parse_date(m.date), latest)
        sw += w * m.serve_won; sp += w * m.serve_played
        aces += w * m.aces; dfs += w * m.dfs; svg += w * m.sv_games
        if m.surface:
            surf[m.surface][0] += w * m.serve_won; surf[m.surface][1] += w * m.serve_played
    sp = max(1.0, sp)
    return TourBaselines(
        tour=tour, spw_avg=sw / sp, rpw_avg=1 - sw / sp,
        ace_rate_avg=aces / sp, df_rate_avg=dfs / sp,
        pts_per_svgame_avg=sp / max(1.0, svg),
        surface_spw={s: v[0] / max(1.0, v[1]) for s, v in surf.items()})


def fit(matches, tour: str) -> tuple[TourBaselines, dict[str, PlayerRates]]:
    """Return (tour baselines, {player_id: PlayerRates}). One tour's matches only.

    Serve/return RATES are recency-weighted (a match near the end of the fitted window
    counts more than one near the start — "recent form/surface performance should carry
    more weight than career averages"). n_matches/n_by_surface stay RAW counts — those
    drive confidence bucketing, which is about genuine sample size, not recency."""
    base = baselines(matches, tour)
    latest = _latest_date(matches)
    Ks = cfg("model", "shrink_serve_pts")
    Kr = cfg("model", "shrink_return_pts")
    Ksurf = cfg("model", "shrink_surface_matches") * max(30.0, base.pts_per_svgame_avg * 10)
    Ka = 250.0

    agg: dict[str, dict] = defaultdict(lambda: {
        "name": "", "sw": 0.0, "sp": 0.0, "rw": 0.0, "rp": 0.0, "ace": 0.0, "df": 0.0,
        "svg": 0.0, "n": 0, "surf": defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])})
    for m in matches:
        if not m.player_id:
            continue
        w = _recency_weight(_parse_date(m.date), latest)
        a = agg[m.player_id]
        a["name"] = m.player or a["name"]
        a["sw"] += w * m.serve_won; a["sp"] += w * m.serve_played
        a["rw"] += w * m.return_won; a["rp"] += w * m.return_played
        a["ace"] += w * m.aces; a["df"] += w * m.dfs; a["svg"] += w * m.sv_games
        a["n"] += 1
        if m.surface:
            s = a["surf"][m.surface]
            s[0] += w * m.serve_won; s[1] += w * m.serve_played
            s[2] += w * m.return_won; s[3] += w * m.return_played; s[4] += 1

    out: dict[str, PlayerRates] = {}
    for pid, a in agg.items():
        spw = _shrink(a["sw"], a["sp"], base.spw_avg, Ks)
        rpw = _shrink(a["rw"], a["rp"], base.rpw_avg, Kr)
        ace = _shrink(a["ace"], a["sp"], base.ace_rate_avg, Ka)
        df = _shrink(a["df"], a["sp"], base.df_rate_avg, Ka)
        surf_spw, surf_rpw, n_surf = {}, {}, {}
        for s, v in a["surf"].items():
            shift_s = base.surface_spw.get(s, base.spw_avg) - base.spw_avg
            surf_spw[s] = _shrink(v[0], v[1], spw + shift_s, Ksurf)     # toward player overall + surface shift
            surf_rpw[s] = _shrink(v[2], v[3], rpw - shift_s, Ksurf)     # return mirrors (surface that helps serve hurts return)
            n_surf[s] = int(v[4])
        out[pid] = PlayerRates(
            player_id=pid, player=a["name"], tour=tour, spw=spw, rpw=rpw,
            ace_rate=ace, df_rate=df,
            pts_per_svgame=(a["sp"] / a["svg"]) if a["svg"] else base.pts_per_svgame_avg,
            surface_spw=surf_spw, surface_rpw=surf_rpw, n_matches=a["n"],
            n_by_surface=n_surf, raw_spw=(a["sw"] / a["sp"]) if a["sp"] else None)
    return base, out
