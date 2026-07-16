"""
mlb_model.py — single-game MLB projections.

Input
-----
form : dict of blended, prior-regressed *rates* for a player (built by
       features.mlb_features). Batter rates are per-PA; pitcher rates per-BF.
ctx  : game context — park, opponent metrics, weather, Vegas implied team total,
       opposing pitcher handedness, lineup spot.

Pipeline per stat: blended rate → multiplicative adjustments (platoon, park,
weather, opponent matchup, Vegas scaling) → Monte-Carlo distribution → optional
XGBoost ensemble re-centring → summarised Projection.
"""

from __future__ import annotations

from typing import Any

from . import montecarlo as mc
from . import ensemble
from .base import Projection, summarize, _f

# league baselines used to convert metrics into multipliers
LG_PITCHER_XFIP = 4.10
LG_TEAM_RUNS = 4.30          # implied runs / team / game
LG_K_PER_PA = 0.225

BATTER_STATS = ["hits", "home_runs", "rbis", "runs", "stolen_bases",
                "total_bases", "walks", "strikeouts"]
PITCHER_STATS = ["strikeouts", "earned_runs", "innings_pitched",
                 "hits_allowed", "walks_allowed", "outs_recorded"]


# ── adjustment multipliers ───────────────────────────────────────────────────

def _platoon_mult(form: dict, ctx: dict, key: str) -> float:
    """Apply vs-LHP / vs-RHP split as a multiplier on a batter rate."""
    hand = (ctx.get("opp_pitcher_hand") or "").upper()[:1]
    splits = form.get("platoon", {})
    if hand in ("L", "R") and splits:
        vs = splits.get(f"vs_{hand}", {})
        base = form.get(key)
        if vs.get(key) and base:
            return _f(vs[key]) / _f(base, 1.0)
    return 1.0


def _weather_hr_mult(ctx: dict) -> float:
    if not ctx.get("outdoor"):
        return 1.0
    temp = _f(ctx.get("temp_f"), 70)
    wind = _f(ctx.get("wind_mph"), 0)
    wdir = (ctx.get("wind_dir") or "").lower()   # 'out','in','cross'
    m = 1.0 + 0.004 * (temp - 70)                # ~0.4%/°F over 70
    if "out" in wdir:
        m *= 1.0 + 0.012 * wind
    elif "in" in wdir:
        m *= 1.0 - 0.010 * wind
    return float(max(0.7, min(1.4, m)))


def _opp_pitcher_mult(ctx: dict) -> float:
    """Batter offense vs the opposing starter (higher xFIP ⇒ more offense)."""
    xfip = _f(ctx.get("opp_pitcher_xfip"), LG_PITCHER_XFIP)
    return float(max(0.75, min(1.30, xfip / LG_PITCHER_XFIP)))


def _vegas_scale(ctx: dict) -> float:
    """Scale offensive volume by the team's Vegas implied run total."""
    tot = _f(ctx.get("vegas_team_total"), LG_TEAM_RUNS)
    return float(max(0.78, min(1.30, tot / LG_TEAM_RUNS)))


def _opp_offense_mult(ctx: dict) -> float:
    """For pitchers: strength of the opposing lineup (their implied total)."""
    tot = _f(ctx.get("opp_team_total"), LG_TEAM_RUNS)
    return float(max(0.78, min(1.30, tot / LG_TEAM_RUNS)))


# ── batters ──────────────────────────────────────────────────────────────────

def project_batter(form: dict, ctx: dict, n: int = mc.N_SIMS,
                   use_ensemble: bool = True) -> dict[str, Projection]:
    park = _f(ctx.get("park_factor"), 1.0)
    weather = _weather_hr_mult(ctx)
    opp = _opp_pitcher_mult(ctx)
    vegas = _vegas_scale(ctx)
    # Vegas implied total is the sharpest single signal and already PRICES the
    # park + opposing pitcher, so we take it at full strength and apply park/opp
    # at half strength (sqrt) to avoid double-counting.
    off = vegas * (park * opp) ** 0.5                         # overall offense level

    drivers_base = [
        f"park ×{park:.2f}", f"opp SP xFIP ×{opp:.2f}",
        f"Vegas total ×{vegas:.2f}",
    ]
    if ctx.get("outdoor"):
        drivers_base.append(f"weather(HR) ×{weather:.2f}")

    exp_pa = _f(form.get("exp_pa"), 4.2) * (0.85 + 0.15 * vegas)   # volume tracks team total
    trials = mc.trial_counts(exp_pa, n)

    # per-PA rates with platoon applied
    p_hit = _f(form.get("p_hit"), 0.23) * _platoon_mult(form, ctx, "p_hit")
    p_hr = _f(form.get("p_hr"), 0.035) * _platoon_mult(form, ctx, "p_hr")
    p_bb = _f(form.get("p_bb"), 0.085)
    p_k = _f(form.get("p_k"), 0.22)
    # hit-type composition (per PA), HR pulled out separately for park/weather
    p_1b = _f(form.get("p_1b"), p_hit * 0.62)
    p_2b = _f(form.get("p_2b"), p_hit * 0.20)
    p_3b = _f(form.get("p_3b"), p_hit * 0.02)

    out: dict[str, Projection] = {}

    # hits
    s_hits = mc.binomial_event(p_hit * off, trials)
    out["hits"] = summarize(s_hits, "hits", drivers_base)

    # home runs (park + weather are genuine HR drivers → full; opp/Vegas halved)
    hr_rate = p_hr * park * weather * (opp * vegas) ** 0.5
    s_hr = mc.binomial_event(hr_rate, trials)
    out["home_runs"] = summarize(s_hr, "home_runs",
                                 drivers_base + [f"HR rate {hr_rate*100:.1f}%/PA"])

    # walks (park/weather neutral)
    s_bb = mc.binomial_event(p_bb, trials)
    out["walks"] = summarize(s_bb, "walks", [f"BB {p_bb*100:.1f}%/PA"])

    # strikeouts (scaled by opposing pitcher K tendency)
    opp_k = _f(ctx.get("opp_pitcher_k_per_pa"), LG_K_PER_PA) / LG_K_PER_PA
    s_k = mc.binomial_event(p_k * max(0.7, min(1.4, opp_k)), trials)
    out["strikeouts"] = summarize(s_k, "strikeouts", [f"K {p_k*100:.0f}%/PA × opp {opp_k:.2f}"])

    # total bases (HR adjusted up by park/weather)
    per_pa = {"1b": p_1b * off, "2b": p_2b * off, "3b": p_3b * off, "hr": hr_rate}
    s_tb = mc.total_bases(per_pa, trials)
    out["total_bases"] = summarize(s_tb, "total_bases", drivers_base)

    # stolen bases — opportunity (on-base) × aggression
    exp_sb = _f(form.get("exp_sb"), 0.08) * (0.9 + 0.2 * off)
    out["stolen_bases"] = summarize(mc.negbinom_count(max(0.01, exp_sb), 0.2, n),
                                    "stolen_bases", [f"SB rate {exp_sb:.2f}/g"])

    # RBI & runs — contextual counts tied to offense level
    exp_rbi = _f(form.get("exp_rbi"), 0.5) * off
    exp_runs = _f(form.get("exp_runs"), 0.5) * off
    out["rbis"] = summarize(mc.negbinom_count(exp_rbi, 0.6, n), "rbis",
                            drivers_base + [f"RBI base {exp_rbi:.2f}"])
    out["runs"] = summarize(mc.negbinom_count(exp_runs, 0.6, n), "runs",
                            drivers_base + [f"runs base {exp_runs:.2f}"])

    if use_ensemble:
        _apply_ensemble("mlb", out, form)
    return out


# ── pitchers ─────────────────────────────────────────────────────────────────

def project_pitcher(form: dict, ctx: dict, n: int = mc.N_SIMS,
                    use_ensemble: bool = True) -> dict[str, Projection]:
    park = _f(ctx.get("park_factor"), 1.0)
    opp_off = _opp_offense_mult(ctx)
    exp_bf = _f(form.get("exp_bf"), 23.0)
    exp_outs = _f(form.get("exp_outs"), 17.0)
    drivers = [f"opp lineup ×{opp_off:.2f}", f"park ×{park:.2f}", f"~{exp_bf:.0f} BF"]

    p_k = _f(form.get("p_k"), 0.235)
    p_bb = _f(form.get("p_bb"), 0.075)
    p_h = _f(form.get("p_h"), 0.21)
    xera = _f(form.get("xera"), 4.00)

    trials = mc.trial_counts(exp_bf, n, lo=3, hi=int(exp_bf * 1.6 + 4))
    out: dict[str, Projection] = {}

    out["strikeouts"] = summarize(mc.binomial_event(p_k, trials), "strikeouts",
                                  [f"K {p_k*100:.0f}%/BF"])
    out["walks_allowed"] = summarize(mc.binomial_event(p_bb, trials), "walks_allowed",
                                     [f"BB {p_bb*100:.1f}%/BF"])
    out["hits_allowed"] = summarize(mc.binomial_event(p_h * opp_off, trials),
                                    "hits_allowed", drivers)

    exp_outs_adj = exp_outs / max(0.85, opp_off ** 0.5)      # tough lineup ⇒ shorter outing
    s_outs = mc.outs_recorded(exp_outs_adj, sd=4.0, n=n)
    out["outs_recorded"] = summarize(s_outs, "outs_recorded",
                                     [f"~{exp_outs_adj:.0f} outs"])
    out["innings_pitched"] = summarize(s_outs / 3.0, "innings_pitched",
                                       [f"~{exp_outs_adj/3:.1f} IP"])

    # earned runs from xERA scaled to expected innings, opponent & park
    exp_ip = exp_outs_adj / 3.0
    exp_er = (xera / 9.0) * exp_ip * opp_off * park
    out["earned_runs"] = summarize(mc.negbinom_count(max(0.05, exp_er), 0.7, n),
                                   "earned_runs", drivers + [f"xERA {xera:.2f}"])

    if use_ensemble:
        _apply_ensemble("mlb", out, form)
    return out


def _apply_ensemble(sport: str, projections: dict[str, Projection], form: dict) -> None:
    """Re-centre each distribution with a trained XGBoost model if one exists."""
    for stat, proj in projections.items():
        xm = XGB_CACHE.get((sport, stat))
        if xm is None:
            xm = ensemble.XGBStatModel(sport, stat)
            xm = xm if xm.train_from_db() else False
            XGB_CACHE[(sport, stat)] = xm
        if xm:
            pred = xm.predict(form)
            new_samples, blended = ensemble.blend_distribution(
                proj.samples, proj.mean, pred, sport, stat)
            if pred is not None and proj.samples is not None:
                projections[stat] = summarize(new_samples, stat,
                                              proj.drivers + [f"XGB {pred:.2f}"])


XGB_CACHE: dict[tuple, Any] = {}


def project(form: dict, ctx: dict, role: str | None = None, **kw) -> dict[str, Projection]:
    """Dispatch by role ('batter'/'pitcher'); inferred from form if not given."""
    role = role or form.get("role") or ("pitcher" if "exp_bf" in form else "batter")
    return (project_pitcher if role == "pitcher" else project_batter)(form, ctx, **kw)
