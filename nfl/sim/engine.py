"""
Monte-Carlo engine — one player-game to a distribution per market.

Uncertainty enters at five independent points, in the order a football game actually
resolves them, and every one of them is drawn PER TRIAL:

  1. team snaps        Normal around the environment estimate. Team OFFENSIVE SNAPS, which
                       are not the same quantity as scrimmage plays (65.3 vs 62.4 a game —
                       see model/environment.py) and are the denominator every per-snap rate
                       here was fit against. The sd is the MEASURED residual of the
                       spread/total fit (13.42% of the mean), not a guess: the fit explains
                       almost none of the variation and this is the rest of it.
  2. game script       the pass/rush split, Normal around the environment estimate with the
                       measured residual sd (0.1047 absolute), splitting the drawn snaps into
                       dropbacks and designed runs.
  3. playing time      the player's snap share, Beta with a concentration scaled by how much
                       real snap history exists (see model/playing_time.py).
  4. opportunity/      the two-stage parameter draw: before any outcome is sampled, each rate
     efficiency        gets its own multiplier scaled by the real evidence behind THAT rate
                       (usage.eff_n). Count rates use Gamma(eff_n, 1/eff_n) — mean 1, CV
                       1/sqrt(eff_n), the standard Gamma-Poisson effective-sample-size
                       approximation, the same construction MLB and WNBA use. Bounded rates
                       (completion rate, catch rate) use Beta(rate*eff_n, (1-rate)*eff_n) —
                       the same construction tennis uses for a serve probability, because a
                       Gamma multiplier on a probability can push it past 1.
  5. outcome           the actual draw, from a distribution that fits the stat family rather
                       than a Normal for everything:
                         attempts / carries / targets  Poisson. MEASURED, not assumed: the
                           implied NegBin dispersion of counts around a player's own rate
                           given real snaps is ~0 (QB attempts -0.004, RB carries +0.008, WR
                           targets -0.013), so the extra spread a NegBin would add would be
                           invented. All real excess spread comes from stages 1-4.
                         completions | attempts, receptions | targets  Binomial at the drawn
                           per-trial rate. That composition IS a beta-binomial, which
                           reproduces the measured overdispersion (QB completions 1.17x
                           binomial, WR receptions 1.12x) without a separate constant.
                         yards  Gamma, shape scaling with the drawn attempt count, using the
                           measured per-attempt sd. Continuous, strictly positive and right-
                           skewed, which is what a game's rushing/receiving yards actually
                           look like — a Normal would put mass below zero for a low-volume
                           back.

Passing/rushing/receiving TDs: `pass_tds` is simulated (Poisson on drawn completions at the
measured league TD-per-completion rate) because passing-TD props are in scope; rushing and
receiving TD props are NOT, so no rush/rec TD array is produced at all rather than one that
looks usable.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import cfg
from ..model.combo_corr import induce_corr, _COMBO_KEYS
from ..model.playing_time import PlayingTime, sample_snap_share
from ..model.usage import UsageRates

# League passing touchdowns per completion. MEASURED 2022-2025 REG over every QB week in the
# nflverse stats_player release: 3,101 passing TDs on 46,177 completions. projections.py
# recomputes this from the same weeks the priors are fit on and passes it in; this constant is
# only the fallback for a caller that has no weeks to measure from.
_TD_PER_COMPLETION_FALLBACK = 0.06715


def _theta(eff_n: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Per-trial multiplier on a count rate: mean 1, CV = 1/sqrt(eff_n)."""
    shape = max(float(eff_n), 1.0)
    return rng.gamma(shape, 1.0 / shape, n)


def _bounded(rate: float, eff_n: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Per-trial draw of a BOUNDED rate (completion rate, catch rate). Beta keeps it in
    [0,1]; a Gamma multiplier would not."""
    r = min(max(float(rate), 1e-4), 1 - 1e-4)
    c = max(float(eff_n), 2.0)
    return rng.beta(r * c, (1.0 - r) * c, n)


def _gamma_yards(count: np.ndarray, per_att_mean: np.ndarray, per_att_sd: float,
                 rng: np.random.Generator) -> np.ndarray:
    """Total yards over `count` attempts: mean count*m, variance count*sd^2, drawn as a Gamma
    so the distribution is right-skewed and non-negative. Zero attempts yields exactly zero
    yards, which is the real outcome, not a Gamma degenerate draw."""
    mean = count * per_att_mean
    var = count * (per_att_sd ** 2)
    out = np.zeros(len(count), dtype=float)
    live = (count > 0) & (mean > 1e-9) & (var > 1e-9)
    if not live.any():
        return out
    shape = (mean[live] ** 2) / var[live]
    scale = var[live] / mean[live]
    out[live] = rng.gamma(np.clip(shape, 1e-3, None), scale)
    return out


def simulate(usage: UsageRates, playing_time: PlayingTime, environment,
             matchup: Optional[dict] = None, n: int | None = None,
             rng: np.random.Generator | None = None,
             combo_corr=None, td_per_completion: Optional[float] = None) -> dict:
    """Distributions for every market this player's position supports. Returns
    {market key: np.ndarray} plus the intermediate `snaps`/`team_snaps` arrays."""
    rng = rng or np.random.default_rng()
    n = int(n or cfg("model", "n_sims", default=10000))
    matchup = matchup or {}
    pos = (usage.position or "").upper()
    sim_cfg = cfg("sim", default={})

    snaps_mean = float(environment.expected_team_snaps)
    team_snaps = np.clip(
        rng.normal(snaps_mean,
                   max(1.0, float(sim_cfg.get("snaps_sd_frac", 0.1342)) * snaps_mean), n),
        0.5 * snaps_mean, 1.6 * snaps_mean)
    dropback_share = np.clip(
        rng.normal(float(environment.expected_dropback_share),
                   float(sim_cfg.get("dropback_sd", 0.1047)), n), 0.2, 0.9)

    snap_share = sample_snap_share(playing_time, n, rng)
    snaps = snap_share * team_snaps
    pass_plays = snaps * dropback_share
    rush_plays = snaps * (1.0 - dropback_share)

    opp, eff, eff_n = usage.opportunity, usage.efficiency, usage.eff_n
    out: dict[str, np.ndarray] = {"team_snaps": team_snaps, "snaps": snaps,
                                  "snap_share": snap_share}

    # ── opportunity counts ───────────────────────────────────────────────────────────────
    # A carry is a rushing-down event and a target is a passing-down event, so each per-snap
    # opportunity rate is applied to ITS OWN play type rather than to raw snaps. The rates
    # were fit per total snap, so they're rescaled by the league's measured split — that
    # keeps a league-average game script landing exactly on the fitted rate and lets only the
    # DIFFERENCE from average move the projection (the same relative-not-absolute correction
    # basketball's matchup pace needed).
    lg_drop = float(cfg("environment", "league_dropback_share", default=0.5673))
    lg_rush = 1.0 - lg_drop

    carries = rng.poisson(np.clip(
        opp.get("carries_per_snap", 0.0) * rush_plays / lg_rush
        * _theta(eff_n.get("carries_per_snap", 1.0), n, rng), 0.0, None))
    targets = rng.poisson(np.clip(
        opp.get("targets_per_snap", 0.0) * pass_plays / lg_drop
        * _theta(eff_n.get("targets_per_snap", 1.0), n, rng), 0.0, None))
    pass_attempts = rng.poisson(np.clip(
        opp.get("pass_att_per_snap", 0.0) * pass_plays / lg_drop
        * _theta(eff_n.get("pass_att_per_snap", 1.0), n, rng), 0.0, None))

    out["rush_attempts"] = carries.astype(float)
    out["targets"] = targets.astype(float)
    out["pass_attempts"] = pass_attempts.astype(float)

    # ── conversions ──────────────────────────────────────────────────────────────────────
    catch_rate = _bounded(eff.get("catch_rate", 0.65), eff_n.get("catch_rate", 50.0), n, rng)
    catch_rate = np.clip(catch_rate * matchup.get("catch_rate", 1.0), 1e-4, 1 - 1e-4)
    out["receptions"] = rng.binomial(targets, catch_rate).astype(float)

    comp_rate = _bounded(eff.get("completion_rate", 0.645),
                         eff_n.get("completion_rate", 200.0), n, rng)
    comp_rate = np.clip(comp_rate * matchup.get("completion_rate", 1.0), 1e-4, 1 - 1e-4)
    out["completions"] = rng.binomial(pass_attempts, comp_rate).astype(float)

    # ── yards ────────────────────────────────────────────────────────────────────────────
    ypa_sd = sim_cfg.get("yards_per_attempt_sd", {})
    ypc_mean = (eff.get("yards_per_carry", 4.3)
                * _theta(eff_n.get("yards_per_carry", 50.0), n, rng)
                * matchup.get("yards_per_carry", 1.0))
    out["rush_yards"] = _gamma_yards(out["rush_attempts"], ypc_mean,
                                     float(ypa_sd.get((pos, "rush"), 6.288)), rng)

    ypt_mean = (eff.get("yards_per_target", 7.5)
                * _theta(eff_n.get("yards_per_target", 50.0), n, rng)
                * matchup.get("yards_per_target", 1.0))
    out["rec_yards"] = _gamma_yards(out["targets"], ypt_mean,
                                    float(ypa_sd.get((pos, "rec"), 9.0)), rng)

    ypat_mean = (eff.get("yards_per_attempt", 7.05)
                 * _theta(eff_n.get("yards_per_attempt", 200.0), n, rng)
                 * matchup.get("yards_per_attempt", 1.0))
    out["pass_yards"] = _gamma_yards(out["pass_attempts"], ypat_mean,
                                     float(ypa_sd.get((pos, "pass"), 9.957)), rng)

    tdpc = float(td_per_completion if td_per_completion is not None else _TD_PER_COMPLETION_FALLBACK)
    out["pass_tds"] = rng.poisson(np.clip(out["completions"] * tdpc, 0.0, None)).astype(float)

    # Re-pair rushing and receiving yards to carry the player's real measured dependence
    # (model/combo_corr.py) instead of only what falls out of sharing snaps/plays. Every
    # marginal is preserved exactly; only the pairing inside a trial changes, which is all
    # the Rush+Rec Yards combo depends on.
    if combo_corr is not None and all(k in out for k in _COMBO_KEYS):
        reordered = induce_corr([out[k] for k in _COMBO_KEYS], combo_corr)
        for k, arr in zip(_COMBO_KEYS, reordered):
            out[k] = arr

    return out


def market_array(sim: dict, key: str) -> Optional[np.ndarray]:
    """Array for a base market or a combo (summed from its already-correlated components)."""
    from .. import COMBOS
    if key in sim:
        return sim[key]
    if key in COMBOS:
        parts = [sim[p] for p in COMBOS[key] if p in sim]
        return sum(parts) if len(parts) == len(COMBOS[key]) else None
    return None


def prob_over(arr: np.ndarray, line: float) -> float:
    return float((arr > line).mean())


def summary(arr: np.ndarray) -> dict:
    """The distribution summary the output contract ships: mean/median/std_dev plus the
    p10/p25/p50/p75/p90 quantiles."""
    q = np.percentile(arr, [10, 25, 50, 75, 90])
    return {
        "mean": round(float(arr.mean()), 2),
        "median": round(float(q[2]), 2),
        "std_dev": round(float(arr.std()), 2),
        "p10": round(float(q[0]), 2), "p25": round(float(q[1]), 2),
        "p50": round(float(q[2]), 2), "p75": round(float(q[3]), 2),
        "p90": round(float(q[4]), 2),
    }
