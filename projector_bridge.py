"""
projector_bridge.py — power JUICED's projections with the stat-projector engine.

JUICED already fetches per-game logs from the MLB Stats API. This bridge feeds
those logs into the stat-projector engine (Bayesian recency blend → predictive
prior → 10k-sim Monte-Carlo distribution) and returns a real distribution per
prop — projection (median), P(over), floor (p10), ceiling (p90), p25/p75 —
instead of a bare empirical median.

Everything is best-effort: if the engine isn't importable or a stat isn't
supported, the functions return None and analytics.py falls back to its existing
empirical model. No new network calls — it reuses the logs JUICED already has.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# The engine is VENDORED at ./projector so the deploy build (which checks out juicedlines
# only) can run it — without this, every deployed MLB prop silently fell back to the empirical
# median. In local dev the full stat-projector sibling is the source of truth and takes
# precedence; the .exists() guard keeps a bogus path out of sys.path in CI so the vendored
# copy resolves cleanly.
_SP = Path(__file__).parent.parent / "stat-projector"
if _SP.exists() and str(_SP) not in sys.path:
    sys.path.insert(0, str(_SP))

try:
    from projector.features import mlb_features as _mf
    from projector.models import mlb_model as _mm
    ENGINE_OK = True
except Exception as _exc:  # pragma: no cover - engine optional
    ENGINE_OK = False
    _IMPORT_ERR = str(_exc)

_BOARD_SIMS = 5000   # a touch lighter than the 10k default; plenty for the board

# prop stat label (normalized) → engine stat key
_BAT = {
    "hits": "hits", "total bases": "total_bases", "bases": "total_bases",
    "home runs": "home_runs", "runs": "runs", "rbis": "rbis",
    "runs batted in": "rbis", "stolen bases": "stolen_bases", "walks": "walks",
    "hitter strikeouts": "strikeouts", "batter strikeouts": "strikeouts",
}
_PIT = {
    "strikeouts": "strikeouts", "pitcher strikeouts": "strikeouts",
    "earned runs": "earned_runs", "earned runs allowed": "earned_runs",
    "hits allowed": "hits_allowed", "walks allowed": "walks_allowed",
    "pitcher walks": "walks_allowed", "outs recorded": "outs_recorded",
    "pitching outs": "outs_recorded", "innings pitched": "innings_pitched",
}
# combos summed from component samples (a true combined distribution)
_COMBO = {
    "hits runs rbis": ("hits", "runs", "rbis"),
    "hits + runs + rbis": ("hits", "runs", "rbis"),
    "hits+runs+rbis": ("hits", "runs", "rbis"),
    "hits runs and rbis": ("hits", "runs", "rbis"),
}


def _f(d: dict, k: str) -> float:
    try:
        return float(d.get(k) or 0)
    except (TypeError, ValueError):
        return 0.0


def _ip_to_outs(ip) -> int:
    """'6.1' (6 ip + 1 out) → 19 outs."""
    try:
        whole, _, frac = str(ip).partition(".")
        return int(whole or 0) * 3 + (int(frac) if frac else 0)
    except (TypeError, ValueError):
        return 0


def _norm(label: str) -> str:
    return re.sub(r"\s+", " ", str(label or "").strip().lower())


def _to_form_logs(logs: list[dict]) -> list[dict]:
    """JUICED _full_logs ([{date, stat:{statsapi keys}}]) → engine log dicts."""
    out = []
    for g in logs:
        s = g.get("stat") or {}
        outs = _ip_to_outs(s.get("inningsPitched"))
        bf = _f(s, "battersFaced") or (outs + _f(s, "hits") + _f(s, "baseOnBalls"))
        out.append({
            "date": g.get("date"),
            # batter
            "PA": _f(s, "plateAppearances"), "AB": _f(s, "atBats"),
            "H": _f(s, "hits"), "2B": _f(s, "doubles"), "3B": _f(s, "triples"),
            "HR": _f(s, "homeRuns"), "BB": _f(s, "baseOnBalls"),
            "SO": _f(s, "strikeOuts"), "RBI": _f(s, "rbi"), "R": _f(s, "runs"),
            "SB": _f(s, "stolenBases"),
            # pitcher
            "BF": bf, "ER": _f(s, "earnedRuns"), "outs": outs,
        })
    return out


def project_player(logs: list[dict], is_pitcher: bool, n: int = _BOARD_SIMS,
                   predictive: dict | None = None, ctx: dict | None = None):
    """
    Run the engine once per player → {stat: Projection}. None on any failure.
    `predictive` (optional Statcast metrics: xBA, barrel%, …) engages the Bayesian
    prior — the part with measured skill over a naive average (+1.5% on hits).
    `ctx` (optional game context: park_factor, opp_pitcher_xfip, opp_team_total, …)
    engages the engine's matchup adjustment layer; empty = neutral (board default).
    """
    if not ENGINE_OK or not logs:
        return None
    try:
        form_logs = _to_form_logs(logs)
        if is_pitcher:
            form = _mf.build_pitcher_form(form_logs, predictive)
        else:
            form = _mf.build_batter_form(form_logs, predictive)
        # ensemble off for speed + determinism.
        return _mm.project(form, ctx or {}, role=("pitcher" if is_pitcher else "batter"),
                           n=n, use_ensemble=False)
    except Exception:
        return None


def statcast_prior(name: str, is_pitcher: bool):
    """
    Current-season Statcast predictive metrics for a player (xBA, barrel%, EV for
    hitters; SwStr% for pitchers), cached 72h. Slow on first pull (~5s) so this is
    for the on-demand drawer, not the board. Returns None on any failure.
    """
    if not ENGINE_OK or not name:
        return None
    try:
        from projector.data import mlb_statcast as sc
        m = sc.statcast_metrics(name, "pitcher" if is_pitcher else "batter", seasons=1)
        return m or None
    except Exception:
        return None


def statcast_prior_cached(name: str, is_pitcher: bool):
    """
    Board-safe: return Statcast metrics ONLY if already cached (no network pull),
    so the A/B variant B can use xBA where it's available without slowing the board.
    Reuses the same cache key statcast_metrics writes.
    """
    if not ENGINE_OK or not name:
        return None
    try:
        from projector import db as pdb
        role = "pitcher" if is_pitcher else "batter"
        return pdb.cache_get(f"statcast:{name.lower()}:{role}", ttl_hours=72) or None
    except Exception:
        return None



def for_stat(projs, stat_label: str, line, is_pitcher: bool, correction: float = 0.0):
    """
    Extract a plain projection dict for one prop from a project_player() result.
    `correction` is an optional per-stat calibration offset (measured bias from the
    ledger) that shifts the whole distribution. None for unsupported stats.
    """
    if not projs:
        return None
    key = _norm(stat_label)
    # inning / half / period props can't come from full-game logs
    if re.search(r"\binning\b|\b1h\b|\b2h\b|1st inning|first inning|\bperiod\b", key):
        return None

    combo = _COMBO.get(key)
    if combo and not is_pitcher:
        import numpy as np
        parts = [projs.get(s) for s in combo]
        if any(p is None or p.samples is None for p in parts):
            return None
        s = np.sum([p.samples for p in parts], axis=0)
        return _payload_from_samples(s, line, correction)

    table = _PIT if is_pitcher else _BAT
    stat = table.get(key) or _best_match(table, key)
    if not stat:
        return None
    p = projs.get(stat)
    if p is None:
        return None
    return _payload(p, line, correction)


def _payload(p, line, correction: float = 0.0) -> dict:
    # Compute from the sample distribution so a calibration `correction` shifts the
    # projection AND prob_over together. Projection = the MEAN (continuous); the
    # median of integer count samples is lumpy.
    s = getattr(p, "samples", None)
    if s is not None:
        out = _payload_from_samples(s, line, correction)
    else:
        out = {"projection": round(float(p.mean), 2), "median": round(float(p.median), 1),
               "floor": round(float(p.floor), 1), "ceiling": round(float(p.ceiling), 1),
               "p25": round(float(p.p25), 1), "p75": round(float(p.p75), 1), "method": "engine"}
        if line is not None:
            out["prob_over"] = round(float(p.prob_over(float(line))), 3)
    out["drivers"] = list(getattr(p, "drivers", []) or [])   # matchup/park factors
    return out


def _payload_from_samples(s, line, correction: float = 0.0) -> dict:
    import numpy as np
    if correction:
        s = np.clip(s + correction, 0, None)     # calibration shift, floored at 0
    q = np.percentile(s, [10, 25, 50, 75, 90])
    out = {
        "projection": round(float(s.mean()), 2), "median": round(float(q[2]), 1),
        "floor": round(float(q[0]), 1), "ceiling": round(float(q[4]), 1),
        "p25": round(float(q[1]), 1), "p75": round(float(q[3]), 1),
        "method": "engine",
    }
    if line is not None:
        out["prob_over"] = round(float(np.mean(s > float(line))), 3)
    return out


def _best_match(table: dict, key: str):
    """Longest whole-key containment match (so 'runs' doesn't swallow 'earned runs')."""
    best = None
    for k, v in table.items():
        if k in key or key in k:
            if best is None or len(k) > best[0]:
                best = (len(k), v)
    return best[1] if best else None
