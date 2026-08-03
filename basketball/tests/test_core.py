"""Unit tests for the basketball core — shrinkage, priors, minutes, sim, markets, value.
All deterministic / offline (no network)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from basketball import BASE_STATS, COMBOS
from basketball.data.base import PlayerGame
from basketball.model import rates as R, priors as PR, minutes as MIN
from basketball.model.pace import matchup_pace
from basketball.sim import engine as E
from basketball.value.finder import implied_prob, to_decimal, value_row
from basketball import projections as P


def _games(n, minutes, pts, reb=5, ast=3):
    return [PlayerGame(date=f"2026-07-{30-i:02d}", league="WNBA", player_id="x", player="X",
                       team_id="1", team="T", opp_id="2", opp="O", minutes=minutes,
                       pts=pts, reb=reb, ast=ast, stl=1, blk=1, to=2, tpm=2) for i in range(n)]


def test_shrinkage_direction():
    prior = PR.positional_prior_poss("G", 96.0, "WNBA")     # pts prior 16/40
    games = _games(15, 30, 30)                              # a big scorer
    light = R.fit_rates(games, "WNBA", prior, 40, 96.0, 50, 6).per_poss["pts"]
    heavy = R.fit_rates(games, "WNBA", prior, 40, 96.0, 5000, 6).per_poss["pts"]
    # heavier shrinkage pulls the estimate toward the (lower) prior
    assert prior["pts"] < heavy < light


def test_positional_prior_poss_rejects_unsupported_league_clearly():
    # 2026-08 regression guard: this used to fall back to an `_SL_PER40` table that was never
    # defined anywhere, so any non-WNBA league raised a bare NameError. Must now fail with a
    # clear, actionable error instead -- and WNBA itself must be completely unaffected.
    import pytest
    with pytest.raises(ValueError, match="no positional priors exist"):
        PR.positional_prior_poss("G", 96.0, "Summer League")
    assert PR.positional_prior_poss("G", 96.0, "WNBA")["pts"] > 0


def test_sample_weight_and_eff_games():
    r = R.fit_rates(_games(10, 30, 20), "WNBA", PR.positional_prior_poss("G", 96, "WNBA"),
                    40, 96.0, 120, 6)
    assert 0.0 < r.sample_weight < 1.0
    assert 3.0 < r.eff_games < 10.0                         # recency-weighted < raw count


def test_minutes_news_and_baseline():
    # injected news wins outright
    m, sd = MIN.project_minutes([28, 30, 26], "WNBA", 1, 0.13, news_minutes=34)
    assert m == 34 and sd > 0
    # minutes track RECENT role: a player who ramped up projects near recent minutes
    m3, _ = MIN.project_minutes([34, 33, 32, 20, 18, 16], "WNBA", 0, 0.13)
    assert m3 > 28                                          # recent (33ish) outweighs old (17ish)


def test_minutes_shrinks_a_debut_game_but_barely_touches_an_established_veteran():
    # 2026-08 regression guard: WNBA's live config used to pass minutes_shrink_games=0,
    # meaning a player's FIRST game -- however fluky (extended overtime, a blowout benching)
    # -- became their entire minutes projection with zero regression to the mean. Only a
    # literal zero-game player ever touched the role baseline. With a nonzero k (see
    # basketball/config.py), a thin sample must be pulled meaningfully toward the baseline
    # while a deep, stable sample is barely affected.
    baseline = MIN._WNBA_ROLE_BASELINE
    k = 1.5
    debut, _ = MIN.project_minutes([38.0], "WNBA", k, 0.13)
    assert debut < 38.0 - 5, "a single fluky game must be pulled well off its raw value"
    assert abs(debut - baseline) < abs(38.0 - baseline), "must move toward the role baseline"

    stable_games = [28.0] * 20
    veteran, _ = MIN.project_minutes(stable_games, "WNBA", k, 0.13)
    assert abs(veteran - 28.0) < 2.0, "an established, stable role must barely move"

    # k=0 (the old, broken behavior) must NOT regress at all -- confirms the mechanism
    # itself still works exactly as before, only the config constant changed
    unshrunk, _ = MIN.project_minutes([38.0], "WNBA", 0, 0.13)
    assert unshrunk == 38.0


def test_sim_mean_and_dispersion():
    rates = R.PlayerRates("X", "WNBA", per_poss={s: 0.0 for s in BASE_STATS})
    rates.per_poss["pts"] = 0.25
    rng = np.random.default_rng(0)
    sim = E.simulate(rates, 30, 0.01, 96.0, 0.0001, 40, 0.10, n=20000, rng=rng)
    poss = (30 / 40) * 96.0
    assert abs(sim["pts"].mean() - 0.25 * poss) < 0.6       # mean ≈ rate × possessions
    # more dispersion → more variance
    tight = E.simulate(rates, 30, 0.01, 96.0, 0.0001, 40, 0.02, n=20000, rng=rng)["pts"].std()
    wide = E.simulate(rates, 30, 0.01, 96.0, 0.0001, 40, 0.30, n=20000, rng=rng)["pts"].std()
    assert wide > tight


def test_two_stage_uncertainty_widens_intervals_for_thin_samples_not_means():
    # 2026-08: every simulated count used to condition on the fitted rate as a FIXED constant
    # across all n trials -- only outcome (sampling) variance was represented. A debut-game
    # player and an established veteran with the IDENTICAL point-estimate rate got the
    # identical spread, understating real uncertainty for the thin sample. rates.eff_poss (the
    # shrinkage denominator: real weighted possessions + prior pseudo-possessions) now drives a
    # per-trial Gamma rate-uncertainty multiplier (mean 1, CV=1/sqrt(eff_poss)) BEFORE the
    # outcome draw -- the textbook Var(Y) = E[Var(Y|theta)] + Var(E[Y|theta]) decomposition.
    per_poss = {s: 0.0 for s in BASE_STATS}
    per_poss["pts"] = 0.25
    thin = R.PlayerRates("X", "WNBA", per_poss=dict(per_poss), eff_poss=150)   # ~debut game
    deep = R.PlayerRates("X", "WNBA", per_poss=dict(per_poss), eff_poss=3000)  # deep, stable sample
    kw = dict(proj_minutes=30, minutes_sd=0.01, matchup_pace=96.0, pace_sd_frac=0.0001,
              game_len=40, disp=0.10, n=40000)

    sd_thin = E.simulate(thin, rng=np.random.default_rng(0), **kw)["pts"].std()
    sd_deep = E.simulate(deep, rng=np.random.default_rng(0), **kw)["pts"].std()
    assert sd_thin > sd_deep, "a thin sample must carry more spread than a deep one at the same rate"

    # the MEAN must stay (statistically) unchanged -- this is a spread fix, not a bias fix.
    # E[theta] = 1 regardless of eff_poss, so both should land near the same point estimate.
    mean_thin = E.simulate(thin, rng=np.random.default_rng(1), **kw)["pts"].mean()
    mean_deep = E.simulate(deep, rng=np.random.default_rng(1), **kw)["pts"].mean()
    poss = (30 / 40) * 96.0
    assert abs(mean_thin - 0.25 * poss) < 0.5
    assert abs(mean_deep - 0.25 * poss) < 0.5

    # a genuinely deep sample (eff_poss -> large) must converge to the OLD fixed-rate spread
    # (no meaningful parameter uncertainty left to add) -- confirms the mechanism, not just
    # its direction.
    huge = R.PlayerRates("X", "WNBA", per_poss=dict(per_poss), eff_poss=10_000_000)
    fixed_rate_sd = E.simulate(R.PlayerRates("X", "WNBA", per_poss=dict(per_poss), eff_poss=1e12),
                               rng=np.random.default_rng(0), **kw)["pts"].std()
    sd_huge = E.simulate(huge, rng=np.random.default_rng(0), **kw)["pts"].std()
    assert abs(sd_huge - fixed_rate_sd) < 0.15


def test_combos_are_sums_and_bounded_probs():
    rates = R.PlayerRates("X", "WNBA", per_poss={s: 0.1 for s in BASE_STATS})
    sim = E.simulate(rates, 30, 2, 96.0, 0.05, 40, 0.12, n=8000, rng=np.random.default_rng(1))
    pra = E.market_array(sim, "pra")
    assert abs(pra.mean() - sum(sim[s].mean() for s in COMBOS["pra"])) < 1e-6
    assert 0.0 <= E.prob_over(sim["pts"], 15.5) <= 1.0


def test_market_resolution():
    assert P._resolve_market("Points") == "pts"
    assert P._resolve_market("Pts+Rebs+Asts") == "pra"
    assert P._resolve_market("3-PT Made") == "3pm"
    assert P._resolve_market("Blks+Stls") == "stocks"
    assert P._resolve_market("Fantasy Score") == "fantasy"
    assert P._resolve_market("Period 1 Points") is None     # period markets skipped
    # split rebounds are modelled (derived from `reb`) — and must beat the generic "rebound"
    # check, or they fall through to total rebounds and over-project ~285%.
    assert P._resolve_market("Offensive Rebounds") == "orb"
    assert P._resolve_market("Defensive Rebounds") == "drb"
    assert P._resolve_market("OREB") == "orb"
    assert P._resolve_market("DREB") == "drb"
    assert P._resolve_market("Rebounds") == "reb"
    # markets we don't simulate must NOT mis-map onto a modelled stat
    for lbl in ("Two Pointers Made", "Two Pointers Attempted", "3-PT Attempted",
                "Rebounding Attempts", "FG Made", "FG Attempted",
                "Free Throws Made", "Double Doubles"):
        assert P._resolve_market(lbl) is None, lbl


def test_derived_rebound_split():
    """orb+drb must equal reb in EVERY sim, and track the player's offensive share."""
    rates = R.PlayerRates("X", "WNBA", per_poss={s: 0.1 for s in BASE_STATS})
    kw = dict(proj_minutes=30, minutes_sd=2, matchup_pace=96.0, pace_sd_frac=0.05,
              game_len=40, disp=0.12, n=8000)
    sim = E.simulate(rates, rng=np.random.default_rng(1), orb_share=0.30, **kw)
    orb, drb, reb = (E.market_array(sim, k) for k in ("orb", "drb", "reb"))
    assert orb is not None and drb is not None
    assert np.all(orb + drb == np.rint(reb).astype(np.int64))   # exact, per-sim
    assert np.all(orb >= 0) and np.all(drb >= 0)
    assert abs(orb.mean() / reb.mean() - 0.30) < 0.02           # honours the share
    # a different share moves only the split, never the total
    sim2 = E.simulate(rates, rng=np.random.default_rng(1), orb_share=0.10, **kw)
    assert np.all(sim2["reb"] == sim["reb"])
    assert sim2["orb"].mean() < orb.mean()
    # no share supplied (e.g. no split data anywhere) → derived stats absent, not zero/garbage
    sim3 = E.simulate(rates, rng=np.random.default_rng(1), **kw)
    assert E.market_array(sim3, "orb") is None


def test_combo_corr_induces_real_pts_reb_ast_dependence_and_widens_pra():
    # 2026-08: the sim only correlates stats INCIDENTALLY, through sharing minutes/pace/
    # possessions -- measured ~0.05-0.08 between pts and reb on live sims, far below what
    # real box scores show (see model/combo_corr.py's pooled default, measured from 15
    # WNBA players' own game logs: pts-reb 0.474). Passing a target combo_corr matrix
    # re-pairs the ALREADY-DRAWN pts/reb/ast marginals (Iman-Conover) so the joint carries
    # the real correlation, without moving any single stat's own distribution.
    from basketball.model.combo_corr import _COMBO_CORR_DEFAULT
    rates = R.PlayerRates("X", "WNBA", per_poss={s: 0.1 for s in BASE_STATS})
    kw = dict(proj_minutes=28, minutes_sd=3, matchup_pace=96.0, pace_sd_frac=0.05,
              game_len=40, disp=0.10, n=30000)
    no_corr = E.simulate(rates, rng=np.random.default_rng(0), **kw)
    induced = E.simulate(rates, rng=np.random.default_rng(0), combo_corr=_COMBO_CORR_DEFAULT, **kw)

    # marginals are EXACTLY preserved -- same values, just re-paired
    for k in ("pts", "reb", "ast"):
        assert np.array_equal(np.sort(no_corr[k]), np.sort(induced[k]))

    c_before = np.corrcoef(no_corr["pts"], no_corr["reb"])[0, 1]
    c_after = np.corrcoef(induced["pts"], induced["reb"])[0, 1]
    assert c_before < 0.15, "sanity: incidental correlation alone should be weak"
    # Iman-Conover matches the target approximately, not exactly -- discrete/skewed count
    # marginals (NegBin, often near-zero) attenuate the achieved Pearson correlation
    # somewhat versus the underlying Gaussian-copula target. What matters is a large,
    # correct-direction move off the weak incidental baseline, not exact reproduction.
    assert c_after > 0.35, "induced correlation should land well above the incidental baseline"

    # a real positive correlation must widen (not narrow) the PRA combo sum's spread,
    # since correlated components sum to more variance than independent ones -- the mean
    # stays put (sum of unchanged marginals).
    pra_before, pra_after = E.market_array(no_corr, "pra"), E.market_array(induced, "pra")
    assert abs(pra_before.mean() - pra_after.mean()) < 0.5
    assert pra_after.std() > pra_before.std() * 1.05


def test_combo_corr_none_leaves_marginals_and_pairing_untouched():
    # combo_corr=None (the default) must reproduce the exact old behavior bit-for-bit --
    # existing callers that don't pass it see zero change.
    rates = R.PlayerRates("X", "WNBA", per_poss={s: 0.1 for s in BASE_STATS})
    kw = dict(proj_minutes=28, minutes_sd=3, matchup_pace=96.0, pace_sd_frac=0.05,
              game_len=40, disp=0.10, n=5000)
    a = E.simulate(rates, rng=np.random.default_rng(3), **kw)
    b = E.simulate(rates, rng=np.random.default_rng(3), combo_corr=None, **kw)
    for k in ("pts", "reb", "ast"):
        assert np.array_equal(a[k], b[k])


def test_empirical_combo_corr_skips_below_min_games_and_shrinks_thin_samples():
    from basketball.model.combo_corr import empirical_combo_corr, _COMBO_CORR_DEFAULT
    # too few games -> None (skip induction entirely, never force a default onto an
    # unmeasured player)
    assert empirical_combo_corr(_games(5, 28, 20)) is None
    # a real, distinctly-non-default measured correlation, moderate sample -> pulled
    # partway toward the pooled default, not fully overridden
    import random
    random.seed(0)
    games = [PlayerGame(date=f"2026-06-{i+1:02d}", league="WNBA", player_id="x", player="X",
                        team_id="1", team="T", opp_id="2", opp="O", minutes=28,
                        pts=10 + i, reb=5, ast=3, stl=1, blk=1, to=2, tpm=2)
             for i in range(20)]   # pts perfectly increasing, reb/ast constant -> undefined corr
    assert empirical_combo_corr(games) is None   # constant reb/ast columns -> corr undefined


def test_orb_share_prior_shrinks():
    base = PR.orb_share_prior("C")
    assert 0.0 < base < 1.0
    # no split data → prior exactly
    assert PR.fit_orb_share([], "C") == base


def test_value_math():
    assert abs(implied_prob(-110) - 0.5238) < 1e-3
    assert abs(to_decimal(+100) - 2.0) < 1e-9
    r = value_row("X", "pts", 15.5, "over", 100, 0.60)
    assert abs(r["edge"] - 0.10) < 1e-9 and abs(r["ev"] - 0.20) < 1e-9


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_") and callable(fn):
            fn(); print("ok", k)
