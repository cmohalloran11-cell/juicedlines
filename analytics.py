"""
analytics.py — per-player analytics for the research drawer.

Given one normalized Line, returns a structured analytics payload: recent form,
hit-rate vs the posted line, vs-opponent splits, headshots/logos, and a plain
narrative. MLB is fully wired against the free official Stats API
(statsapi.mlb.com). World Cup gets a lighter view (matchup + our own line-movement
history) because there is no equivalent free per-player game-log feed.

    ┌──────────────────────────────────────────────────────────────────┐
    │  PROJECTION MODEL HOOK                                            │
    │  analyze_mlb() already computes an EMPIRICAL hit-rate from real   │
    │  game logs (mlb.empirical_prob_over). To plug in your own model,  │
    │  replace `model_projection()` below — it receives the recent      │
    │  per-game values + line and returns {projection, prob_over}.      │
    └──────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import math
import re
import sys
import threading
import time

import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

# Reuse the betting_dashboard MLB client (session, cache, name-normalisation,
# the empirical hit-rate model, and the player→id index).
_BD = Path(__file__).parent.parent / "betting_dashboard"
# Local dev uses the sibling betting_dashboard; a standalone deploy uses the vendored
# mlb_model.py alongside this file (see also underdog.py).
if _BD.exists() and str(_BD) not in sys.path:
    sys.path.insert(0, str(_BD))

try:
    import mlb_model as mlb
    _MLB_OK = True
except Exception:
    _MLB_OK = False

# History DB for line-movement based soccer analytics
import db

# stat-projector engine (optional) — powers MLB projections with a real
# Bayesian + Monte-Carlo distribution when importable; falls back to empirical.
try:
    import projector_bridge
    _BRIDGE_OK = projector_bridge.ENGINE_OK
except Exception:
    _BRIDGE_OK = False

_HEADSHOT = "https://midfield.mlbstatic.com/v1/people/{id}/spots/120"
_TEAM_LOGO = "https://www.mlbstatic.com/team-logos/{id}.svg"

_LOCK = threading.Lock()
_CACHE: dict[str, tuple[float, Any]] = {}


def _cached(key: str, ttl: float, producer: Callable[[], Any]) -> Any:
    now = time.time()
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    val = producer()
    with _LOCK:
        _CACHE[key] = (time.time(), val)
    return val


# ───────────────────────────────────── stat-label resolvers ──────────────────

def _f(d: dict, k: str) -> float:
    try:
        return float(d.get(k) or 0)
    except (TypeError, ValueError):
        return 0.0


def _ip_to_float(ip: Any) -> float:
    """'6.1' (6 ip + 1 out) → 6.333; '6.2' → 6.667."""
    try:
        s = str(ip)
        whole, _, frac = s.partition(".")
        outs = int(frac) if frac else 0
        return int(whole) + outs / 3.0
    except (TypeError, ValueError):
        return 0.0


def _hitter_fantasy(s: dict) -> float:
    """
    PrizePicks MLB hitter fantasy score — official scoring:
      1B×3, 2B×5, 3B×8, HR×10, R×2, RBI×2, BB×2, HBP×2, SB×5.
    Source: prizepicks.com playbook "How to Play PrizePicks MLB & Fantasy
    Scoring System" (verified against the live board's posted lines, mean
    over-rate ~0.50). The earlier H+R+RBI shortcut was the reported bug.
    """
    hr = _f(s, "homeRuns"); d = _f(s, "doubles"); t = _f(s, "triples")
    singles = _f(s, "hits") - d - t - hr
    return (singles * 3 + d * 5 + t * 8 + hr * 10
            + _f(s, "runs") * 2 + _f(s, "rbi") * 2
            + _f(s, "baseOnBalls") * 2 + _f(s, "hitByPitch") * 2
            + _f(s, "stolenBases") * 5)


def _pitcher_fantasy(s: dict) -> float:
    """
    PrizePicks MLB pitcher fantasy score — official scoring:
      Out ×1, Strikeout ×3, Earned Run ×-3, Win ×6, Quality Start ×4.
    (Quality Start = >=6 IP and <=3 ER.) Same source as _hitter_fantasy.
    """
    ip = _ip_to_float(s.get("inningsPitched"))
    outs = round(ip * 3)
    er = _f(s, "earnedRuns")
    qs = 1 if (ip >= 6 and er <= 3) else 0
    return (outs * 1 + _f(s, "strikeOuts") * 3 + er * -3
            + _f(s, "wins") * 6 + qs * 4)


# label (normalized) → function over a pitching game-log stat dict
PITCH_MAP: dict[str, Callable[[dict], float]] = {
    "pitcher strikeouts": lambda s: _f(s, "strikeOuts"),
    "strikeouts": lambda s: _f(s, "strikeOuts"),
    "pitches thrown": lambda s: _f(s, "numberOfPitches"),
    "pitch count": lambda s: _f(s, "numberOfPitches"),
    "pitching outs": lambda s: round(_ip_to_float(s.get("inningsPitched")) * 3),
    "outs recorded": lambda s: round(_ip_to_float(s.get("inningsPitched")) * 3),
    "innings pitched": lambda s: _ip_to_float(s.get("inningsPitched")),
    "earned runs allowed": lambda s: _f(s, "earnedRuns"),
    "earned runs": lambda s: _f(s, "earnedRuns"),
    "runs allowed": lambda s: _f(s, "runs"),
    "hits allowed": lambda s: _f(s, "hits"),
    "walks allowed": lambda s: _f(s, "baseOnBalls"),
    "pitcher walks": lambda s: _f(s, "baseOnBalls"),
    "pitcher fantasy score": _pitcher_fantasy,
    "fantasy score": _pitcher_fantasy,
    "fantasy points": _pitcher_fantasy,
}

# hitting labels — extend mlb_model.STAT_MAP with the source label variants
HIT_MAP: dict[str, Callable[[dict], float]] = {
    "hits": lambda s: _f(s, "hits"),
    "total bases": lambda s: _f(s, "totalBases"),
    "bases": lambda s: _f(s, "totalBases"),
    "home runs": lambda s: _f(s, "homeRuns"),
    "runs": lambda s: _f(s, "runs"),
    "rbis": lambda s: _f(s, "rbi"),
    "runs batted in": lambda s: _f(s, "rbi"),
    "hits runs rbis": lambda s: _f(s, "hits") + _f(s, "runs") + _f(s, "rbi"),
    "hits+runs+rbis": lambda s: _f(s, "hits") + _f(s, "runs") + _f(s, "rbi"),
    "hitter fantasy score": _hitter_fantasy,
    "fantasy score": _hitter_fantasy,
    "fantasy points": _hitter_fantasy,
    "stolen bases": lambda s: _f(s, "stolenBases"),
    "doubles": lambda s: _f(s, "doubles"),
    "triples": lambda s: _f(s, "triples"),
    "singles": lambda s: _f(s, "hits") - _f(s, "doubles") - _f(s, "triples") - _f(s, "homeRuns"),
    "walks": lambda s: _f(s, "baseOnBalls"),
    "hitter strikeouts": lambda s: _f(s, "strikeOuts"),
    "batter strikeouts": lambda s: _f(s, "strikeOuts"),
    "hits + runs + rbis": lambda s: _f(s, "hits") + _f(s, "runs") + _f(s, "rbi"),
}


def _norm(label: str) -> str:
    s = re.sub(r"\s+", " ", str(label).strip().lower())
    return s


def _resolve_stat(label: str, is_pitcher: bool) -> Optional[Callable[[dict], float]]:
    n = _norm(label)
    # Full-game logs can't represent inning/half-specific props, so we don't
    # compute a (wrong) hit-rate for them — recent form still shows.
    if re.search(r"\binning\b|\b1h\b|\b2h\b|1st inning|first inning|\bperiod\b", n):
        return None
    table = PITCH_MAP if is_pitcher else HIT_MAP
    if n in table:
        return table[n]
    # exact-ish fallback: prefer the longest matching key to avoid e.g. "runs"
    # swallowing "earned runs". Only match on whole-key containment.
    best = None
    for k, fn in table.items():
        if k in n or n in k:
            if best is None or len(k) > best[0]:
                best = (len(k), fn)
    return best[1] if best else None


# ───────────────────────────────────────── MLB reference data ────────────────

def _team_map() -> dict:
    """{'abbr'->{id,name}, '_by_id'->{id->{abbr,name}}}"""
    if not _MLB_OK:
        return {"_by_id": {}}

    def produce():
        r = mlb._session.get(f"{mlb.BASE}/teams",
                             params={"sportId": 1, "season": mlb.season()}, timeout=25)
        r.raise_for_status()
        by_abbr, by_id = {}, {}
        for t in r.json().get("teams", []):
            entry = {"id": t["id"], "name": t["name"], "abbr": t.get("abbreviation")}
            if t.get("abbreviation"):
                by_abbr[t["abbreviation"]] = entry
            by_id[t["id"]] = entry
        by_abbr["_by_id"] = by_id
        return by_abbr
    return _cached(f"teams_{mlb.season()}", 12 * 3600, produce)


def _player_meta() -> dict:
    """
    normalized name → LIST of candidate dicts. A list (not a single value) so we
    can disambiguate same-name players (e.g. the two Max Muncys, LAD vs ATH) by
    the line's team — see _resolve_player.
    """
    if not _MLB_OK:
        return {}

    def produce():
        r = mlb._session.get(f"{mlb.BASE}/sports/1/players",
                             params={"season": mlb.season()}, timeout=30)
        r.raise_for_status()
        out: dict[str, list] = {}
        for p in r.json().get("people", []):
            pos = p.get("primaryPosition", {}) or {}
            out.setdefault(mlb._norm_name(p.get("fullName", "")), []).append({
                "id": p.get("id"),
                "name": p.get("fullName"),
                "pos_type": pos.get("type"),
                "pos_abbr": pos.get("abbreviation"),
                "team_id": (p.get("currentTeam", {}) or {}).get("id"),
                "bats": (p.get("batSide", {}) or {}).get("code"),
                "throws": (p.get("pitchHand", {}) or {}).get("code"),
            })
        return out
    return _cached(f"player_meta_{mlb.season()}", 12 * 3600, produce)


def _prop_is_pitcher(stat_label: str | None) -> Optional[bool]:
    """True/False if the prop label clearly indicates a pitcher/hitter prop, else None."""
    s = (stat_label or "").lower()
    # pitcher signals first ("Hits Allowed" is a pitcher prop, not a hitter one)
    if re.search(r"\bpitch|innings|earned run|outs recorded|pitcher|allowed|strikeouts thrown", s):
        return True
    if re.search(r"\bhitter|\bbatter|total bases|home run|\brbis?\b|stolen base|\bhits\b|doubles|singles", s):
        return False
    return None


def _resolve_player(name: str | None, team_abbr: str | None = None,
                    want_pitcher: Optional[bool] = None) -> Optional[dict]:
    """
    Pick the right player for a name, disambiguating same-name players by team
    first, then by whether the prop is a pitcher/hitter prop (handles the two
    Max Muncys, and the Jose Fermin / Luis Garcia hitter+pitcher pairs).
    """
    if not name:
        return None
    cands = _player_meta().get(mlb._norm_name(name), [])
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    # 1) narrow by team
    if team_abbr:
        teams = _team_map()
        want = team_abbr.strip().upper()
        matched = [c for c in cands
                   if (teams["_by_id"].get(c.get("team_id"), {}) or {}).get("abbr", "").upper() == want]
        if len(matched) == 1:
            return matched[0]
        if matched:
            cands = matched
    # 2) narrow by pitcher/hitter role
    if want_pitcher is not None:
        role = [c for c in cands if (c.get("pos_type") == "Pitcher") == want_pitcher]
        if role:
            return role[0]
    return cands[0]


def _today_opponents() -> dict:
    """
    team_id → {opp_id, opp_abbr, opp_name, is_home, opp_pitcher_id,
    opp_pitcher_name} for today's slate (incl. the opposing probable pitcher).
    """
    if not _MLB_OK:
        return {}

    def produce():
        try:
            r = mlb._session.get(f"{mlb.BASE}/schedule",
                                 params={"sportId": 1, "date": date.today().isoformat(),
                                         "hydrate": "team,probablePitcher"}, timeout=20)
            r.raise_for_status()
        except Exception:
            return {}
        out = {}
        for d in r.json().get("dates", []):
            for g in d.get("games", []):
                home, away = g["teams"]["home"], g["teams"]["away"]
                h, a = home["team"], away["team"]
                hp = home.get("probablePitcher") or {}
                ap = away.get("probablePitcher") or {}
                # each team faces the OTHER team's probable pitcher
                out[h["id"]] = {"opp_id": a["id"], "opp_name": a["name"],
                                "opp_abbr": a.get("abbreviation"), "is_home": True,
                                "opp_pitcher_id": ap.get("id"), "opp_pitcher_name": ap.get("fullName")}
                out[a["id"]] = {"opp_id": h["id"], "opp_name": h["name"],
                                "opp_abbr": h.get("abbreviation"), "is_home": False,
                                "opp_pitcher_id": hp.get("id"), "opp_pitcher_name": hp.get("fullName")}
        return out
    return _cached(f"sched_{date.today().isoformat()}", 1800, produce)


def _batter_vs_pitcher(batter_id: int, pitcher_id: int) -> Optional[dict]:
    """Career batter-vs-pitcher (BvP) line."""
    def produce():
        try:
            r = mlb._session.get(f"{mlb.BASE}/people/{batter_id}/stats",
                                 params={"stats": "vsPlayerTotal", "group": "hitting",
                                         "opposingPlayerId": pitcher_id}, timeout=20)
            r.raise_for_status()
            stats = r.json().get("stats", [])
            splits = stats[0].get("splits", []) if stats else []
            return splits[0].get("stat", {}) if splits else None
        except Exception:
            return None
    return _cached(f"bvp_{batter_id}_{pitcher_id}", 6 * 3600, produce)


def _full_logs(pid: int, group: str) -> list[dict]:
    """[{date, opp_id, opp_name, is_home, stat}], oldest→newest. Cached 3h."""
    def produce():
        hyd = f"stats(group=[{group}],type=[gameLog],season=[{mlb.season()}])"
        try:
            r = mlb._session.get(f"{mlb.BASE}/people",
                                 params={"personIds": pid, "hydrate": hyd}, timeout=25)
            r.raise_for_status()
        except Exception:
            return []
        people = r.json().get("people", [])
        if not people:
            return []
        splits = []
        for b in people[0].get("stats", []) or []:
            splits = b.get("splits", []) or splits
        out = []
        for sp in splits:
            opp = sp.get("opponent", {}) or {}
            out.append({
                "date": sp.get("date"),
                "opp_id": opp.get("id"),
                "opp_name": opp.get("name"),
                "is_home": sp.get("isHome"),
                "stat": sp.get("stat", {}),
            })
        return out
    return _cached(f"log_{group}_{pid}_{mlb.season()}", 3 * 3600, produce)


def _median(vals: list[float]) -> float:
    """Median — robust to right-skew. Projecting the median (typical game) instead
    of the mean keeps the projection comparable to the market line, which is set
    at the ~50% point. The mean over-projects skewed stats (e.g. fantasy score)."""
    s = sorted(v for v in vals if v is not None)
    if not s:
        return 0.0
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _proj_games(logs: list[dict], is_pitcher: bool) -> list[dict]:
    """
    Pick which game logs feed a projection.
      • Hitters: last ~40 games where they actually played (≥2 PA) — drops
        pinch-hit cameos that deflate the per-game average; the line assumes a
        normal start.
      • Pitchers: only games matching their CURRENT role (starter vs reliever),
        so a player who converted roles (e.g. Tyler Phillips: reliever→starter)
        isn't projected off the wrong workload.
    """
    if not logs:
        return []
    if not is_pitcher:
        def pa(g):
            s = g.get("stat") or {}
            return float(s.get("plateAppearances") or s.get("atBats") or 0)
        real = [g for g in logs if pa(g) >= 2]
        return (real if len(real) >= 10 else logs)[-mlb._RECENT_GAMES:]

    def started(g):
        return float((g.get("stat") or {}).get("gamesStarted") or 0) >= 1

    # current role = majority of the last 3 appearances
    is_starter = sum(1 for g in logs[-3:] if started(g)) >= 2
    role = [g for g in logs if started(g) == is_starter]
    return role[-12:] if len(role) >= 3 else logs[-mlb._RECENT_GAMES:]


def _vs_team(pid: int, group: str, opp_team_id: int) -> Optional[dict]:
    """Career + recent vs-opponent summary for the relevant group (spans fetched in parallel)."""
    def fetch_span(span_label):
        span, label = span_label
        try:
            r = mlb._session.get(f"{mlb.BASE}/people/{pid}/stats",
                                 params={"stats": span, "group": group,
                                         "opposingTeamId": opp_team_id}, timeout=20)
            r.raise_for_status()
            stats = r.json().get("stats", [])
            splits = stats[0].get("splits", []) if stats else []
            return (label, splits[0].get("stat", {})) if splits else (label, None)
        except Exception:
            return (label, None)

    def produce():
        out = {}
        with ThreadPoolExecutor(max_workers=2) as ex:
            for label, stat in ex.map(fetch_span, (("vsTeamTotal", "career"), ("vsTeam5Y", "last5y"))):
                if stat:
                    out[label] = stat
        return out or None
    return _cached(f"vsteam_{group}_{pid}_{opp_team_id}", 6 * 3600, produce)


# ───────────────────────────────── matchup context (engine) ──────────────────
# Park run factors keyed by MLB API abbr (AZ/CWS/SF/WSH). >1 = hitter-friendly.
PARK_FACTORS = {
    "COL": 1.15, "CIN": 1.08, "BOS": 1.06, "KC": 1.05, "PHI": 1.04, "TEX": 1.03,
    "AZ": 1.03, "BAL": 1.03, "LAA": 1.02, "TOR": 1.02, "ATL": 1.01, "CHC": 1.01,
    "WSH": 1.01, "HOU": 1.00, "MIN": 1.00, "NYY": 1.00, "CWS": 0.99, "STL": 0.99,
    "PIT": 0.99, "MIL": 0.99, "LAD": 0.99, "NYM": 0.98, "CLE": 0.98, "TB": 0.97,
    "DET": 0.97, "MIA": 0.97, "ATH": 0.97, "SF": 0.95, "SD": 0.95, "SEA": 0.94,
}


def _park_factor(abbr):
    return PARK_FACTORS.get((abbr or "").upper())


def _pitcher_quality(pid):
    """Opposing starter quality → {fip, k_per_pa} from season pitching (cached 12h)."""
    if not pid or not _MLB_OK:
        return None

    def produce():
        try:
            r = mlb._session.get(f"{mlb.BASE}/people/{pid}/stats",
                                 params={"stats": "season", "group": "pitching",
                                         "season": mlb.season()}, timeout=20)
            r.raise_for_status()
            sp = (r.json().get("stats", [{}])[0].get("splits", []) or [{}])
            s = sp[0].get("stat", {}) if sp else {}
        except Exception:
            return None
        ip = _ip_to_float(s.get("inningsPitched"))
        if ip < 10:                       # too small a sample to trust
            return None
        k, bf = _f(s, "strikeOuts"), _f(s, "battersFaced")
        fip = (13 * _f(s, "homeRuns") + 3 * _f(s, "baseOnBalls") - 2 * k) / ip + 3.15
        out = {"fip": round(max(2.0, min(7.0, fip)), 2)}
        if bf > 0:
            out["k_per_pa"] = round(k / bf, 3)
        return out
    return _cached(f"pq_{pid}_{mlb.season()}", 12 * 3600, produce)


def _team_offense(team_id):
    """Opposing lineup strength → runs/game (for pitcher props), cached 12h."""
    if not team_id or not _MLB_OK:
        return None

    def produce():
        try:
            r = mlb._session.get(f"{mlb.BASE}/teams/{team_id}/stats",
                                 params={"stats": "season", "group": "hitting",
                                         "season": mlb.season(), "sportId": 1}, timeout=20)
            r.raise_for_status()
            sp = (r.json().get("stats", [{}])[0].get("splits", []) or [{}])
            s = sp[0].get("stat", {}) if sp else {}
        except Exception:
            return None
        runs, g = _f(s, "runs"), _f(s, "gamesPlayed")
        return round(runs / g, 2) if g > 0 else None
    return _cached(f"toff_{team_id}_{mlb.season()}", 12 * 3600, produce)


def _matchup_ctx(meta: dict, is_pitcher: bool, teams: dict) -> dict:
    """Engine game context from today's schedule: park + opponent quality. The
    engine clips + dampens these (Vegas primary, park/opp at sqrt) so they can't
    blow up; empty when there's no scheduled game (→ neutral projection)."""
    tid = meta.get("team_id")
    sched = _today_opponents().get(tid) if tid else None
    if not sched:
        return {}
    ctx: dict = {}
    park_team = tid if sched.get("is_home") else sched.get("opp_id")
    pf = _park_factor((teams["_by_id"].get(park_team, {}) or {}).get("abbr"))
    if pf:
        ctx["park_factor"] = pf
    if is_pitcher:
        rt = _team_offense(sched.get("opp_id"))
        if rt:
            ctx["opp_team_total"] = rt
    else:
        q = _pitcher_quality(sched.get("opp_pitcher_id"))
        if q:
            ctx["opp_pitcher_xfip"] = q["fip"]
            if "k_per_pa" in q:
                ctx["opp_pitcher_k_per_pa"] = q["k_per_pa"]
    return ctx


# ──────────────────────────────────── projection model hook ──────────────────

def model_projection(values: list[float], line: float) -> dict:
    """
    ── PROJECTION MODEL HOOK ──
    Default: empirical hit-rate from the player's own recent game-by-game values
    (no distributional assumption — see mlb_model for why). Swap the body to call
    your own model; keep the return shape {projection, prob_over, method}.
    """
    clean = [v for v in values if v is not None]
    prob = mlb.empirical_prob_over(clean, line) if (_MLB_OK and clean) else None
    # median, not mean — comparable to the line and unbiased on skewed stats
    proj = round(_median(clean), 2) if clean else None
    return {"projection": proj, "prob_over": prob, "method": "empirical_median"}


# ─────────────────────────────────────────── MLB analyzer ────────────────────

# key per-game fields shown in the compact recent-form table, by player type
_PITCH_VIEW = [("IP", lambda s: s.get("inningsPitched", "0")),
               ("K", lambda s: int(_f(s, "strikeOuts"))),
               ("ER", lambda s: int(_f(s, "earnedRuns"))),
               ("H", lambda s: int(_f(s, "hits"))),
               ("BB", lambda s: int(_f(s, "baseOnBalls"))),
               ("P", lambda s: int(_f(s, "numberOfPitches")))]
_HIT_VIEW = [("H", lambda s: int(_f(s, "hits"))),
             ("HR", lambda s: int(_f(s, "homeRuns"))),
             ("RBI", lambda s: int(_f(s, "rbi"))),
             ("R", lambda s: int(_f(s, "runs"))),
             ("TB", lambda s: int(_f(s, "totalBases"))),
             ("BB", lambda s: int(_f(s, "baseOnBalls")))]


def analyze_mlb(line: dict) -> dict:
    if not _MLB_OK:
        return {"available": False, "reason": "MLB data module unavailable."}

    name = line.get("player")
    if not name:
        # team/market line with no player — no per-player analytics
        return {"available": False, "reason": "Team/market line — no player splits."}

    # Resolve the player, disambiguating same-name players by team + prop type
    # (the two Max Muncys; the Fermin / Garcia hitter+pitcher pairs).
    stat_label = line.get("stat_type") or ""
    prop_pitcher = _prop_is_pitcher(stat_label)
    meta = _resolve_player(name, line.get("team"), prop_pitcher)
    if not meta or not meta.get("id"):
        return {"available": False, "reason": f"No MLB player match for “{name}”."}

    pid = meta["id"]
    teams = _team_map()
    is_pitcher = (meta.get("pos_type") == "Pitcher")
    # a clear pitcher prop label overrides (e.g. two-way player's pitching prop)
    if prop_pitcher is True:
        is_pitcher = True
    elif prop_pitcher is False:
        is_pitcher = False
    group = "pitching" if is_pitcher else "hitting"

    logs = _full_logs(pid, group)
    view = _PITCH_VIEW if is_pitcher else _HIT_VIEW
    line_val = line.get("line")
    value_fn = _resolve_stat(stat_label, is_pitcher)

    # recent compact game table (newest first, last 10) — includes the prop's own
    # per-game value + whether it cleared the line, so hit/miss is transparent.
    recent = []
    for g in logs[-10:][::-1]:
        s = g["stat"]
        row = {
            "date": g["date"],
            "opp": (teams["_by_id"].get(g["opp_id"], {}) or {}).get("abbr") or
                   (g.get("opp_name") or "")[:3].upper(),
            "home": g.get("is_home"),
            "cells": {lbl: fn(s) for lbl, fn in view},
        }
        if value_fn and line_val is not None:
            v = round(value_fn(s), 1)
            row["prop_val"] = int(v) if v == int(v) else v
            row["cleared"] = v > line_val
        recent.append(row)

    # hit-rate + projection vs the posted line for the specific stat. Pitchers use
    # current-role games only (see _proj_games) so a reliever→starter conversion
    # isn't projected off the wrong workload.
    hit = None
    if value_fn and line_val is not None and logs:
        pg = _proj_games(logs, is_pitcher)
        vals = [value_fn(g["stat"]) for g in pg]
        overs = sum(1 for v in vals if v > line_val)
        last5 = [value_fn(g["stat"]) for g in pg[-5:]]

        # stat-projector engine (Monte-Carlo distribution) → empirical fallback.
        # The drawer is on-demand (one player), so we can afford the Statcast prior
        # pull (xBA/barrel% — the part with measured skill); cached 72h after first.
        eng = None
        prior = None
        if _BRIDGE_OK:
            if not is_pitcher:
                prior = projector_bridge.statcast_prior(meta["name"], is_pitcher)
            # matchup context: park + opposing-pitcher quality (hitters) /
            # opposing-lineup strength (pitchers), from today's schedule.
            mctx = _matchup_ctx(meta, is_pitcher, teams)
            projs = projector_bridge.project_player(pg, is_pitcher, predictive=prior, ctx=mctx)
            sc_corr = _stat_corrections().get((stat_label or "").lower(), 0.0)
            eng = projector_bridge.for_stat(projs, stat_label, float(line_val), is_pitcher, sc_corr)
        if eng:
            projection, prob_over = eng["projection"], eng.get("prob_over")
            method = "engine+xBA" if prior else "engine"
        else:
            mp = model_projection(vals, float(line_val))
            projection, prob_over, method = mp["projection"], mp["prob_over"], mp["method"]

        hit = {
            "line": line_val,
            "stat": stat_label,
            "n": len(vals),
            "over": overs,
            "under": len(vals) - overs,
            "over_pct": round(100 * overs / len(vals)) if vals else None,
            "last5_over": sum(1 for v in last5 if v > line_val),
            "last5_n": len(last5),
            "projection": projection,
            "prob_over": prob_over,
            "method": method,
            "floor": eng["floor"] if eng else None,
            "ceiling": eng["ceiling"] if eng else None,
            "drivers": eng.get("drivers") if eng else None,
            "spark": [value_fn(g["stat"]) for g in pg[-15:]],
        }

    # ── opponent / matchup ──
    # Prefer today's schedule (authoritative, team-id based). Only fall back to
    # the prop's matchup field — which is often messy ("ATH 1+2+3 Innings",
    # "ATH/LAA") — and then only its clean leading team abbreviation.
    sched = _today_opponents().get(meta.get("team_id")) if meta.get("team_id") else None
    opp_entry = None
    if sched and sched.get("opp_id"):
        opp_entry = {"id": sched["opp_id"], "name": sched["opp_name"], "abbr": sched["opp_abbr"]}
    if not opp_entry:
        mu = re.match(r"^([A-Za-z]{2,3})\b", (line.get("matchup") or "").strip())
        if mu:
            opp_entry = teams.get(mu.group(1).upper())
    opp = opp_entry and {**opp_entry, "logo": _TEAM_LOGO.format(id=opp_entry["id"])}

    # Pitchers → vs the opposing TEAM (lineup). Hitters → vs the opposing
    # PITCHER (BvP), which is the matchup that actually moves a hitter's line.
    vs_team = None
    vs_pitcher = None
    if is_pitcher:
        if opp_entry and opp_entry.get("id"):
            raw = _vs_team(pid, group, opp_entry["id"])
            if raw:
                vs_team = {"opp": opp_entry, "group": group, "spans": raw}
    else:
        opp_pid = sched.get("opp_pitcher_id") if sched else None
        opp_pname = sched.get("opp_pitcher_name") if sched else None
        if opp_pid:
            pmeta = _player_meta()  # for throws hand of the pitcher
            stat = _batter_vs_pitcher(pid, opp_pid)
            vs_pitcher = {
                "pitcher": {"id": opp_pid, "name": opp_pname,
                            "headshot": _HEADSHOT.format(id=opp_pid)},
                "stat": stat,   # may be None / tiny sample — UI says so
            }

    team_entry = teams["_by_id"].get(meta.get("team_id")) if meta.get("team_id") else None

    return {
        "available": True,
        "sport": "MLB",
        "player": meta["name"],
        "player_type": "Pitcher" if is_pitcher else "Hitter",
        "pos": meta.get("pos_abbr"),
        "bats": meta.get("bats"),
        "throws": meta.get("throws"),
        "headshot": _HEADSHOT.format(id=pid),
        "team": team_entry and {**team_entry, "logo": _TEAM_LOGO.format(id=team_entry["id"])},
        "opponent": opp,
        "view_cols": [lbl for lbl, _ in view],
        "recent": recent,
        "hit_rate": hit,
        "vs_team": vs_team,
        "vs_pitcher": vs_pitcher,
        "season_games": len(logs),
    }


# ────────────────────────────────────────── Soccer analyzer ──────────────────

def analyze_soccer(line: dict) -> dict:
    """
    Lighter view: matchup + our own stored line-movement history + implied prices.
    (No free per-player soccer game-log feed comparable to statsapi; this is the
    honest best-effort until a real source is wired — see model hook note.)
    """
    hist = db.get_history(line["id"], limit=50)
    movement = [{"ts": h["ts"], "line": h["line_value"],
                 "over": h["over_implied"]} for h in hist]
    country = line.get("country") or line.get("team")
    return {
        "available": True,
        "sport": "World Cup",
        "player": line.get("player"),
        "player_type": line.get("position") or "Player",
        "headshot": line.get("headshot"),
        "team": country and {"name": country, "flag": _flag(country)},
        "matchup": line.get("matchup"),
        "stat": line.get("stat_type"),
        "line": line.get("line"),
        "over_price": line.get("over_price"),
        "under_price": line.get("under_price"),
        "model_proj": line.get("model_proj"),
        "model_edge": line.get("model_edge"),
        "model_prob": line.get("model_prob"),
        "model_n": line.get("model_n"),
        "proj_kind": line.get("proj_kind"),
        "movement": movement,
        "note": "Projection = Poisson expected count from the de-vigged market "
                "price where a two-sided price exists, else cross-book consensus. "
                "No free per-player soccer game-log feed (see attach_projections).",
    }


_FLAGS = {
    "france": "🇫🇷", "argentina": "🇦🇷", "brazil": "🇧🇷", "england": "🏴",
    "spain": "🇪🇸", "portugal": "🇵🇹", "germany": "🇩🇪", "usa": "🇺🇸",
    "united states": "🇺🇸", "belgium": "🇧🇪", "netherlands": "🇳🇱",
    "italy": "🇮🇹", "croatia": "🇭🇷", "uruguay": "🇺🇾", "mexico": "🇲🇽",
    "senegal": "🇸🇳", "japan": "🇯🇵", "south korea": "🇰🇷", "korea": "🇰🇷",
    "morocco": "🇲🇦", "colombia": "🇨🇴", "switzerland": "🇨🇭", "denmark": "🇩🇰",
    "poland": "🇵🇱", "australia": "🇦🇺", "canada": "🇨🇦", "ghana": "🇬🇭",
    "ecuador": "🇪🇨", "serbia": "🇷🇸", "egypt": "🇪🇬", "nigeria": "🇳🇬",
    "wales": "🏴", "iran": "🇮🇷", "saudi arabia": "🇸🇦", "qatar": "🇶🇦",
    "cameroon": "🇨🇲", "ivory coast": "🇨🇮", "paraguay": "🇵🇾", "peru": "🇵🇪",
    "chile": "🇨🇱", "austria": "🇦🇹", "turkey": "🇹🇷", "turkiye": "🇹🇷",
    # FIFA 3-letter codes
    "fra": "🇫🇷", "arg": "🇦🇷", "bra": "🇧🇷", "eng": "🏴", "esp": "🇪🇸",
    "por": "🇵🇹", "ger": "🇩🇪", "bel": "🇧🇪", "ned": "🇳🇱", "ita": "🇮🇹",
    "usa": "🇺🇸", "uru": "🇺🇾", "mex": "🇲🇽", "sen": "🇸🇳", "cro": "🇭🇷",
    # ISO-3 codes (Underdog `country` field)
    "prt": "🇵🇹", "deu": "🇩🇪", "gbr": "🏴", "nld": "🇳🇱", "hrv": "🇭🇷",
    "ury": "🇺🇾", "che": "🇨🇭", "dnk": "🇩🇰", "pol": "🇵🇱", "aus": "🇦🇺",
    "can": "🇨🇦", "gha": "🇬🇭", "ecu": "🇪🇨", "srb": "🇷🇸", "egy": "🇪🇬",
    "nga": "🇳🇬", "irn": "🇮🇷", "sau": "🇸🇦", "qat": "🇶🇦", "cmr": "🇨🇲",
    "civ": "🇨🇮", "pry": "🇵🇾", "per": "🇵🇪", "chl": "🇨🇱", "aut": "🇦🇹",
    "tur": "🇹🇷", "jpn": "🇯🇵", "kor": "🇰🇷", "mar": "🇲🇦", "col": "🇨🇴",
    "mex_": "🇲🇽", "nor": "🇳🇴", "swe": "🇸🇪", "sco": "🏴",
}


def _flag(team: str | None) -> str:
    if not team:
        return ""
    t = team.strip().lower()
    return _FLAGS.get(t) or _FLAGS.get(t[:3], "")


# ─────────────────────────────────────── line enrichment (board) ─────────────

def enrich_lines(lines: list[dict]) -> None:
    """
    Mutate lines in place, attaching cheap display fields the board uses:
    MLB headshot + team logo + position; World Cup country flag. All from the
    cached player/team maps — one dict lookup per line, no extra HTTP.
    """
    if not _MLB_OK:
        for l in lines:
            if l.get("sport") == "World Cup" and l.get("team"):
                l["flag"] = _flag(l["team"])
        return
    try:
        teams = _team_map()
    except Exception:
        return
    for l in lines:
        sport = l.get("sport")
        if sport == "MLB" and l.get("player"):
            # disambiguate same-name players by team + pitcher/hitter prop type
            m = _resolve_player(l["player"], l.get("team"),
                                _prop_is_pitcher(l.get("stat_type")))
            if m and m.get("id"):
                l["mlb_id"] = m["id"]   # stable group key across books
                l["headshot"] = _HEADSHOT.format(id=m["id"])
                if not l.get("position"):
                    l["position"] = m.get("pos_abbr")
                tid = m.get("team_id")
                if tid:
                    l["team_logo"] = _TEAM_LOGO.format(id=tid)
                    if not l.get("team"):
                        l["team"] = (teams["_by_id"].get(tid, {}) or {}).get("abbr")
        elif sport == "World Cup":
            l["flag"] = _flag(l.get("country") or l.get("team"))
        elif sport in ("WNBA", "NBA Summer League"):
            try:
                from basketball.analytics import team_asset
                from basketball import projections as _bp
                # resolve the player → their ESPN team (always has a logo); the book's
                # own team abbr often doesn't match ESPN's, so it's only a fallback.
                ref = _bp.resolve(sport, l.get("player"))
                a = (team_asset(sport, ref.team) if ref else None) or team_asset(sport, l.get("team"))
                if a:
                    if a.get("logo"):
                        l["team_logo"] = a["logo"]
                    if a.get("abbr"):
                        l["team"] = a["abbr"]          # normalize to the ESPN abbr
            except Exception:
                pass


# ──────────────────────────────────── model projections (board) ──────────────

_BATCH = 40


def _prewarm_logs(ids_by_group: dict[str, list[int]]) -> None:
    """
    Batch-fetch game logs for many players at once and populate the SAME cache
    _full_logs reads (`log_{group}_{pid}_{season}`). This both powers board
    projections and makes the drawer fast (its log fetch becomes a cache hit).
    """
    season = mlb.season()
    now = time.time()
    for group, ids in ids_by_group.items():
        miss = []
        with _LOCK:
            for i in ids:
                hit = _CACHE.get(f"log_{group}_{i}_{season}")
                if not (hit and now - hit[0] < 3 * 3600):
                    miss.append(i)
        for start in range(0, len(miss), _BATCH):
            batch = miss[start:start + _BATCH]
            hyd = f"stats(group=[{group}],type=[gameLog],season=[{season}])"
            try:
                r = mlb._session.get(
                    f"{mlb.BASE}/people",
                    params={"personIds": ",".join(map(str, batch)), "hydrate": hyd},
                    timeout=30)
                r.raise_for_status()
                people = r.json().get("people", [])
            except Exception:
                people = []
            got: dict[int, list] = {}
            for person in people:
                splits = []
                for b in person.get("stats", []) or []:
                    splits = b.get("splits", []) or splits
                rows = []
                for sp in splits:
                    opp = sp.get("opponent", {}) or {}
                    rows.append({"date": sp.get("date"), "opp_id": opp.get("id"),
                                 "opp_name": opp.get("name"), "is_home": sp.get("isHome"),
                                 "stat": sp.get("stat", {})})
                got[person.get("id")] = rows
            with _LOCK:
                for i in batch:
                    _CACHE[f"log_{group}_{i}_{season}"] = (time.time(), got.get(i, []))


def _prewarm_statcast(hitter_names: list[str], limit: int = 6) -> None:
    """
    Bounded, best-effort: pull Statcast xBA for up to `limit` not-yet-cached hitters
    each refresh, so the A/B variant B gains xBA coverage over a few cycles (cached
    72h) without a big burst of Baseball Savant pulls. Network I/O (releases the GIL).
    """
    if not _BRIDGE_OK:
        return
    done = 0
    for name in hitter_names:
        if done >= limit:
            break
        try:
            if projector_bridge.statcast_prior_cached(name, False):
                continue                                   # already cached
            if projector_bridge.statcast_prior(name, False):   # pulls + caches
                done += 1
        except Exception:
            continue


_BIAS_DAMP = 0.5     # apply half the measured bias (conservative on thin data)


def _stat_corrections() -> dict:
    """
    Damped per-stat calibration offsets from the graded ledger (cached 1h). If a
    stat is systematically under/over-projected vs actual outcomes, nudge future
    projections toward reality. Gated (db.stat_biases min_n) + only meaningful
    biases (|bias|≥0.08) + damped 50% + capped ±0.3, so it corrects real bias
    without chasing one week of noise. Self-improves as the ledger grows.
    """
    def produce():
        try:
            biases = db.stat_biases("MLB", min_n=60)
        except Exception:
            return {}
        return {k: max(-0.3, min(0.3, -v * _BIAS_DAMP))
                for k, v in biases.items() if abs(v) >= 0.08}
    return _cached("stat_corrections", 3600, produce)


def _is_allstar_mlb(l: dict) -> bool:
    """MLB All-Star props use league 'teams' (AL / NL), never a real club. The game is an
    exhibition — hitters get ~2-3 PA and pitchers ~1-2 innings — so the full-game model
    projects a normal start/lineup and posts absurd edges against the (low) All-Star lines
    (e.g. Pitches Thrown proj 104 vs a 14.5 line). We can't model who plays or for how long,
    so these get NO projection (defer to the market line the book already set for it)."""
    return (l.get("sport") == "MLB"
            and (l.get("team") or "").strip().upper() in {"AL", "NL"})


# Temporary per-day override: UTC dates on which ALL MLB projections are suppressed (the
# board still shows the MLB lines, just no projections/edges). Auto-reverts the next day —
# remove the date to re-enable. Used to sit out an off slate (e.g. All-Star day).
_MLB_MUTE_DATES = frozenset({"2026-07-14"})


def _mlb_muted_today() -> bool:
    return datetime.now(timezone.utc).date().isoformat() in _MLB_MUTE_DATES


def attach_projections(lines: list[dict]) -> None:
    """
    Attach `model_proj` + `model_edge` (+ `proj_kind`) to every line so the board
    can show a projection per row.
      • MLB  → empirical recent-game average for the prop's stat ("model").
      • World Cup → cross-book consensus of the posted lines ("consensus"),
        since there is no free soccer game-log feed.
    MLB All-Star props (AL/NL teams) are skipped — an exhibition with unknown playing
    time can't be projected from full-game logs (see _is_allstar_mlb).
    """
    if _MLB_OK:
        mlb_muted = _mlb_muted_today()      # per-day override — no MLB projections today
        resolved: dict[str, tuple] = {}
        ids_by_group: dict[str, set] = {"hitting": set(), "pitching": set()}
        hitter_names: list[str] = []
        for l in lines:
            if l.get("sport") != "MLB" or not l.get("player") or l.get("line") is None:
                continue
            if mlb_muted or _is_allstar_mlb(l):   # muted day / exhibition — no projection
                continue
            stat = l.get("stat_type") or ""
            pp = _prop_is_pitcher(stat)
            m = _resolve_player(l["player"], l.get("team"), pp)
            if not m or not m.get("id"):
                continue
            is_p = (m.get("pos_type") == "Pitcher")
            if pp is True:
                is_p = True
            elif pp is False:
                is_p = False
            group = "pitching" if is_p else "hitting"
            fn = _resolve_stat(stat, is_p)
            if not fn:
                continue
            resolved[l["id"]] = (m, group, fn)          # store full meta (for matchup ctx)
            ids_by_group[group].add(m["id"])
            if not is_p and m.get("name") and m["name"] not in hitter_names:
                hitter_names.append(m["name"])

        try:
            _prewarm_logs({g: list(s) for g, s in ids_by_group.items() if s})
        except Exception:
            pass
        try:
            _prewarm_statcast(hitter_names, limit=6)     # A/B variant B: xBA coverage
        except Exception:
            pass
        teams = _team_map()
        corr = _stat_corrections()              # per-stat calibration offsets (ledger-driven)

        engine_cache: dict[tuple, Any] = {}     # (pid,is_pitcher[,'b']) → engine projections
        for l in lines:
            r = resolved.get(l["id"])
            if not r:
                continue
            meta, group, fn = r
            pid = meta["id"]
            logs = _full_logs(pid, group)
            if not logs:
                continue
            is_pitcher = (group == "pitching")
            line_val = float(l["line"])
            # current-role / real-start games only (the reliever→starter fix and
            # pinch-hit drop) — feed the engine the SAME clean sample.
            pg = _proj_games(logs, is_pitcher)
            if not pg:
                continue

            # 1) stat-projector engine (variant A): plain Bayesian blend → Monte-Carlo.
            if _BRIDGE_OK:
                ck = (pid, is_pitcher)
                sc_corr = corr.get((l.get("stat_type") or "").lower(), 0.0)
                if ck not in engine_cache:
                    engine_cache[ck] = projector_bridge.project_player(pg, is_pitcher)
                eng = projector_bridge.for_stat(
                    engine_cache[ck], l.get("stat_type") or "", line_val, is_pitcher, sc_corr)
                if eng:
                    l["model_proj"] = eng["projection"]
                    l["model_edge"] = round(eng["projection"] - line_val, 1)
                    l["model_floor"] = eng["floor"]
                    l["model_ceiling"] = eng["ceiling"]
                    l["proj_kind"] = "engine"
                    if eng.get("prob_over") is not None:
                        l["model_prob"] = eng["prob_over"]
                    l["model_n"] = len(pg)

                    # variant B (A/B test): engine + matchup context + cached xBA prior.
                    ckb = (pid, is_pitcher, "b")
                    if ckb not in engine_cache:
                        try:
                            mctx = _matchup_ctx(meta, is_pitcher, teams)
                            pri = (projector_bridge.statcast_prior_cached(meta.get("name"), is_pitcher)
                                   if not is_pitcher else None)
                            engine_cache[ckb] = projector_bridge.project_player(
                                pg, is_pitcher, predictive=pri, ctx=mctx)
                        except Exception:
                            engine_cache[ckb] = None
                    if engine_cache.get(ckb):
                        engb = projector_bridge.for_stat(
                            engine_cache[ckb], l.get("stat_type") or "", line_val, is_pitcher, sc_corr)
                        if engb:
                            l["model_proj_b"] = engb["projection"]
                            if engb.get("prob_over") is not None:
                                l["model_prob_b"] = engb["prob_over"]
                    continue

            # 2) fallback: empirical median + empirical P(over)
            vals = [fn(g["stat"]) for g in pg]
            if not vals:
                continue
            proj = round(_median(vals), 1)   # median: unbiased on skewed stats
            l["model_proj"] = proj
            l["model_edge"] = round(proj - line_val, 1)
            l["proj_kind"] = "model"
            prob = mlb.empirical_prob_over(vals, line_val)
            if prob is not None:
                l["model_prob"] = round(prob, 3)
            l["model_n"] = len(vals)

    _attach_soccer_projections(lines)

    # tennis: serve/return + Monte-Carlo model on live ATP/WTA prop lines
    try:
        from tennis.board import attach_tennis
        attach_tennis(lines)
    except Exception as exc:
        print(f"[tennis] attach failed: {exc}")

    # basketball: per-possession core on live WNBA + NBA Summer League prop lines
    try:
        from basketball.board import attach_basketball
        attach_basketball(lines)
    except Exception as exc:
        print(f"[basketball] attach failed: {exc}")


# ───────────────────────────────────── soccer Poisson model ──────────────────

def _poisson_sf(k: int, lam: float) -> float:
    """P(X >= k) for a Poisson(lam) count."""
    if k <= 0:
        return 1.0
    cdf = math.exp(-lam)   # P(X=0)
    term = cdf
    for i in range(1, k):
        term *= lam / i
        cdf += term
    return max(0.0, min(1.0, 1.0 - cdf))


def _am_prob(american: Any) -> Optional[float]:
    """American odds → implied probability (with vig)."""
    if american in (None, ""):
        return None
    try:
        n = float(str(american).strip())
    except (TypeError, ValueError):
        return None
    return (-n) / ((-n) + 100) if n < 0 else 100 / (n + 100)


def _lambda_from_over_prob(line: float, p_over: float) -> Optional[float]:
    """
    Solve the Poisson mean λ such that P(X >= ceil(line)) == p_over (bisection).
    This is the market's de-vigged expected count for the stat — the sharpest
    projection available for soccer, where there's no free per-game stat feed.
    """
    if p_over is None or not (0.02 < p_over < 0.98):
        return None
    k = max(1, math.ceil(line))
    lo, hi = 1e-4, 150.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if _poisson_sf(k, mid) < p_over:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 3)


# ─────────────────────────────── ESPN soccer per-game stats ──────────────────
# Free, unofficial ESPN API. World Cup rosters give a clean name→athleteId map;
# the athlete gamelog gives per-game G/A/shots/SOT/fouls/cards/offsides — enough
# for a real recency projection (the soccer analog of MLB's statsapi logs).

_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world"
_ESPN_GL = "https://site.web.api.espn.com/apis/common/v3/sports/soccer/all/athletes/{id}/gamelog"
_espn_session = requests.Session()
_espn_session.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json"})

# prop stat label (normalized) → function over one ESPN per-game stat dict.
# EXACT match only — ESPN's gamelog only has goals/assists/shots/SOT/fouls/offsides/
# cards. Anything else (shots ASSISTED, goals ALLOWED, saves, passes, tackles,
# clearances, crosses, combos, keeper stats) must NOT be force-mapped to a base
# stat — it falls back to the market model instead. Loose substring matching was
# producing nonsense (e.g. "Shots Assisted"→shots taken, "Goals Allowed"→0).
_ESPN_STAT: dict[str, Callable[[dict], float]] = {
    "goals": lambda g: g.get("totalGoals", 0),
    "assists": lambda g: g.get("goalAssists", 0),
    "goals assists": lambda g: g.get("totalGoals", 0) + g.get("goalAssists", 0),
    "goals + assists": lambda g: g.get("totalGoals", 0) + g.get("goalAssists", 0),
    "goal + assist": lambda g: g.get("totalGoals", 0) + g.get("goalAssists", 0),
    "shots": lambda g: g.get("totalShots", 0),
    "shots attempted": lambda g: g.get("totalShots", 0),
    "shots on target": lambda g: g.get("shotsOnTarget", 0),
    "fouls": lambda g: g.get("foulsCommitted", 0),
    "fouls drawn": lambda g: g.get("foulsSuffered", 0),
    "offsides": lambda g: g.get("offsides", 0),
    "cards": lambda g: g.get("yellowCards", 0) + g.get("redCards", 0),
}


def _espn_stat_fn(label: str) -> Optional[Callable[[dict], float]]:
    n = _norm(label)
    if "(combo)" in n or re.search(r"\binning\b|1st|first goal|period|\b1h\b|\b2h\b|half", n):
        return None  # combos / partial-game props can't come from full-game logs
    return _ESPN_STAT.get(n)  # exact match only — no risky substring matching


def _espn_roster_map() -> dict:
    """normalized player name → ESPN athleteId, from all World Cup rosters."""
    def produce() -> dict:
        out: dict[str, str] = {}
        try:
            tr = _espn_session.get(f"{_ESPN_BASE}/teams", timeout=20).json()
            teams = tr["sports"][0]["leagues"][0]["teams"]
        except Exception:
            return out

        def fetch(tid):
            res = {}
            try:
                rr = _espn_session.get(f"{_ESPN_BASE}/teams/{tid}/roster", timeout=20).json()
                for a in rr.get("athletes", []):
                    for p in (a.get("items", [a]) if "items" in a else [a]):
                        nm = mlb._norm_name(p.get("displayName", ""))
                        if nm and p.get("id"):
                            res[nm] = str(p["id"])
            except Exception:
                pass
            return res

        with ThreadPoolExecutor(max_workers=10) as ex:
            for d in ex.map(fetch, [t["team"]["id"] for t in teams]):
                out.update(d)
        return out
    return _cached("espn_roster", 12 * 3600, produce)


def _espn_gamelog(athlete_id: str) -> list[dict]:
    """Per-game stat dicts for an ESPN athlete (current-season scoped), cached."""
    def produce() -> list[dict]:
        try:
            d = _espn_session.get(_ESPN_GL.format(id=athlete_id), timeout=20).json()
        except Exception:
            return []
        names = d.get("names") or []
        if not names:
            return []
        games = []
        for stp in d.get("seasonTypes", []) or []:
            for cat in stp.get("categories", []) or []:
                for ev in cat.get("events", []) or []:
                    stats = ev.get("stats") or []
                    if len(stats) != len(names):
                        continue
                    g = {}
                    for k, v in zip(names, stats):
                        try:
                            g[k] = float(v)
                        except (TypeError, ValueError):
                            g[k] = 0.0
                    games.append(g)
        return games
    return _cached(f"espn_gl_{athlete_id}", 6 * 3600, produce)


_ESPN_MIN_GAMES = 5
_ESPN_WINDOW = 20  # most recent N games used for the projection

# ESPN only exposes a CLUB gamelog for World Cup players (e.g. "2025-26 Serie A"), never a
# World-Cup one, so `form` is club-league form. The World Cup is a lower-event, tougher
# environment (better defenders, cagier knockout games, minutes spread across a deep squad),
# so club rates systematically over-project WC output. Deflate the club-form component per
# stat toward WC reality before blending with the market. This is a blanket bias correction
# — it can't capture an individual's WC role (a fringe sub whose club form is high but who
# barely features), which needs an actual WC box-score feed.
_WC_FORM_DEFLATE = {
    "shots": 0.82, "shots on target": 0.78, "goals": 0.62, "assists": 0.66,
    "goals assists": 0.63, "goal + assist": 0.63, "goal assist": 0.63,
    "fouls": 0.85, "fouls drawn": 0.85, "offsides": 0.80, "cards": 0.90,
}


def _attach_soccer_espn(lines: list[dict]) -> set:
    """
    Real recency projection from ESPN per-game logs. Returns the set of line ids
    that got an ESPN-backed projection (the rest fall back to the market model).
    Safeguards: exact normalized name match against WC rosters + ≥5 games, else skip.
    """
    done: set = set()
    try:
        roster = _espn_roster_map()
    except Exception:
        return done
    if not roster:
        return done

    # which athletes do we actually need logs for (players with a mappable prop)?
    need: dict[str, bool] = {}
    for l in lines:
        if l.get("sport") != "World Cup" or not l.get("player") or l.get("line") is None:
            continue
        if not _espn_stat_fn(l.get("stat_type") or ""):
            continue
        aid = roster.get(mlb._norm_name(l["player"]))
        if aid:
            need[aid] = True
    if not need:
        return done
    try:
        with ThreadPoolExecutor(max_workers=10) as ex:
            list(ex.map(_espn_gamelog, list(need.keys())))
    except Exception:
        pass

    # Market anchor per (player, stat): the standard line is the market's fair
    # estimate for the WORLD CUP game. Club form is informative but biased
    # (different opponents/role/minutes), so we blend the two — this both
    # improves accuracy and tames wild club-vs-line gaps. See _blend below.
    from collections import defaultdict
    std: dict[tuple, list[float]] = defaultdict(list)
    allv: dict[tuple, list[float]] = defaultdict(list)
    for l in lines:
        if l.get("sport") == "World Cup" and l.get("player") and l.get("line") is not None:
            k = (mlb._norm_name(l["player"]), (l.get("stat_type") or "").strip().lower())
            allv[k].append(float(l["line"]))
            if l.get("odds_type") in ("standard", None):
                std[k].append(float(l["line"]))

    def market_anchor(key):
        src = std.get(key) or allv.get(key)
        if not src:
            return None
        s = sorted(src)
        return s[len(s) // 2]

    for l in lines:
        if l.get("sport") != "World Cup" or not l.get("player") or l.get("line") is None:
            continue
        fn = _espn_stat_fn(l.get("stat_type") or "")
        if not fn:
            continue
        aid = roster.get(mlb._norm_name(l["player"]))
        if not aid:
            continue
        games = _espn_gamelog(aid)
        if len(games) < _ESPN_MIN_GAMES:
            continue
        vals = [fn(g) for g in games[-_ESPN_WINDOW:]]
        form = sum(vals) / len(vals)                 # raw club-form rate
        form *= _WC_FORM_DEFLATE.get((l.get("stat_type") or "").strip().lower(), 0.75)  # club → WC
        n = len(vals)
        key = (mlb._norm_name(l["player"]), (l.get("stat_type") or "").strip().lower())
        anchor = market_anchor(key)
        if anchor is not None:
            # weight on club form: grows with sample, capped at 0.5 (club→intl bias)
            w = min(0.5, n / (n + 12))
            proj = w * form + (1 - w) * anchor
        else:
            proj = form
        line = float(l["line"])
        l["model_proj"] = round(proj, 1)
        l["model_edge"] = round(proj - line, 1)
        l["model_prob"] = round(_poisson_sf(max(1, math.ceil(line)), proj), 3)
        l["model_n"] = n
        l["proj_kind"] = "espn"
        done.add(l["id"])
    return done


def _attach_soccer_projections(lines: list[dict]) -> None:
    """
    World Cup projections, best signal first:
      1. ESPN per-game stats → real recency projection (proj differs from line).
      2. Poisson mean from the de-vigged market price (where two-sided prices exist).
      3. Cross-book consensus line (last resort).
    """
    try:
        espn_done = _attach_soccer_espn(lines)
    except Exception:
        espn_done = set()
    _attach_soccer_market(lines, skip=espn_done)


# Partial-game props (tagged "(1H)"/"(2H)"/"(1Q)"/"(Half)" by the puller) describe a
# period, not the full game — we only model full games, so they must NOT get a
# projection, or the consensus fallback would price them and they'd resurface as phantom
# edges. Basketball already skips them via _resolve_market; this guards soccer.
_PARTIAL_STAT_RE = re.compile(r"\((?:1h|2h|[1-4][hqp]|ot|half|partial)\)", re.I)


def _is_partial_stat(stat_type: str | None) -> bool:
    return bool(_PARTIAL_STAT_RE.search(stat_type or ""))


def _attach_soccer_market(lines: list[dict], skip: set | None = None) -> None:
    """Poisson-from-price / consensus fallback for lines ESPN didn't cover."""
    skip = skip or set()
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for l in lines:
        if l.get("sport") == "World Cup" and l.get("player") and l.get("line") is not None:
            key = (l["player"].strip().lower(), (l.get("stat_type") or "").strip().lower())
            groups[key].append(l)

    proj: dict[tuple, tuple[float, str]] = {}
    for key, ls in groups.items():
        lam_est = []
        for l in ls:
            if l.get("odds_type") not in ("standard", None):
                continue
            po, pu = _am_prob(l.get("over_price")), _am_prob(l.get("under_price"))
            if po and pu:
                lam = _lambda_from_over_prob(float(l["line"]), po / (po + pu))
                if lam:
                    lam_est.append(lam)
        if lam_est:
            proj[key] = (round(sum(lam_est) / len(lam_est), 2), "poisson")
        else:
            vals = sorted(float(x["line"]) for x in ls)
            proj[key] = (vals[len(vals) // 2], "consensus")

    for l in lines:
        if (l.get("sport") != "World Cup" or l.get("line") is None or not l.get("player")
                or l["id"] in skip or _is_partial_stat(l.get("stat_type"))):
            continue
        key = (l["player"].strip().lower(), (l.get("stat_type") or "").strip().lower())
        pk = proj.get(key)
        if not pk:
            continue
        lam, kind = pk
        line = float(l["line"])
        if kind == "consensus":
            # No gamelog signal for this stat — ESPN's soccer log has no tackles / passes /
            # clearances / dribbles / saves, so there's nothing to project. Defer to THIS
            # line (edge 0) instead of a group median that invents edges out of line
            # dispersion (e.g. Rodri tackles proj 3.5 vs a 2.5 line). No data ⇒ no edge.
            lam = line
        l["model_proj"] = round(lam, 1)
        l["model_edge"] = round(lam - line, 1)
        l["model_prob"] = round(_poisson_sf(max(1, math.ceil(line)), lam), 3)
        l["proj_kind"] = kind


# ─────────────────────────────────────────────── dispatcher ──────────────────

def analyze(line: dict) -> dict:
    sport = line.get("sport")
    try:
        if sport == "MLB":
            return analyze_mlb(line)
        if sport == "World Cup":
            return analyze_soccer(line)
        if sport in ("WNBA", "NBA Summer League"):
            from basketball.analytics import analyze as _bball
            return _bball(line)
        if sport == "Tennis":
            from tennis.analytics import analyze as _tennis
            return _tennis(line)
    except Exception as exc:
        return {"available": False, "reason": f"analytics error: {exc}"}
    return {"available": False, "reason": f"No analytics wired for {sport} yet."}


# ─────────────────────────── CLV ledger grading ──────────────────────────────

def grade_pending() -> dict:
    """
    Settle logged props from past game-days: resolve each player, pull the
    game-log for the logged date, apply the stat function → actual value, and
    record it (db.set_actual). Props whose player didn't play (or whose stat
    isn't gradeable from a box score, e.g. fantasy/inning props) get voided once
    stale so they stop being retried. Reuses the same cached game-log lookups as
    the board, so it's cheap. MLB only (no free per-game soccer grading feed).
    """
    if not _MLB_OK:
        return {"graded": 0, "voided": 0}
    from datetime import datetime, timezone, timedelta
    today = date.today().isoformat()
    pend = db.pending_grades(today, "MLB", limit=150)
    if not pend:
        return {"graded": 0, "voided": 0}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stale_before = (date.today() - timedelta(days=2)).isoformat()
    logcache: dict = {}
    graded = voided = 0
    for p in pend:
        try:
            name, stat_label, gd, lid = p["player"], p["stat_type"], p["game_date"], p["line_id"]
            pp = _prop_is_pitcher(stat_label)
            meta = _resolve_player(name, None, pp)
            fn = _resolve_stat(stat_label, pp is True) if meta else None
            if not meta or not meta.get("id") or not fn:
                if gd < stale_before:           # unresolvable / ungradeable stat
                    db.set_actual(lid, gd, None, now); voided += 1
                continue
            is_pitcher = (meta.get("pos_type") == "Pitcher")
            if pp is True: is_pitcher = True
            elif pp is False: is_pitcher = False
            group = "pitching" if is_pitcher else "hitting"
            fn = _resolve_stat(stat_label, is_pitcher) or fn
            logs = logcache.get((meta["id"], group))
            if logs is None:
                logs = _full_logs(meta["id"], group)
                logcache[(meta["id"], group)] = logs
            game = next((g for g in logs if g.get("date") == gd), None)
            if game:
                db.set_actual(lid, gd, float(fn(game["stat"])), now); graded += 1
            elif gd < stale_before:             # player didn't play that day
                db.set_actual(lid, gd, None, now); voided += 1
        except Exception:
            continue
    return {"graded": graded, "voided": voided}
