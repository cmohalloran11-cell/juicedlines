"""Unit tests for the basketball core — shrinkage, priors, minutes, sim, markets, value.
All deterministic / offline (no network)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from basketball import BASE_STATS, COMBOS
from basketball.data.base import PlayerGame, PlayerBackground
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


def test_sample_weight_and_eff_games():
    r = R.fit_rates(_games(10, 30, 20), "WNBA", PR.positional_prior_poss("G", 96, "WNBA"),
                    40, 96.0, 120, 6)
    assert 0.0 < r.sample_weight < 1.0
    assert 3.0 < r.eff_games < 10.0                         # recency-weighted < raw count


def test_translated_prior_differentiates():
    pace = 102.0
    star = PlayerBackground("A", draft_pick=2, pre_league="NCAA",
                            rates40={s: v for s, v in zip(BASE_STATS, [24, 8, 4, 1.5, 1, 2.5, 3])})
    fringe = PlayerBackground("B", draft_pick=None, pre_league="International",
                              rates40={s: v for s, v in zip(BASE_STATS, [12, 4, 2, .8, .4, 1, 2.5])})
    sp, _ = PR.translated_prior_poss(star, pace)
    fp, _ = PR.translated_prior_poss(fringe, pace)
    assert sp["pts"] > fp["pts"]                            # better prospect → higher prior
    assert PR.draft_minutes_prior(2) > PR.draft_minutes_prior(None)


def test_minutes_news_and_baseline():
    # injected news wins outright
    m, sd = MIN.project_minutes([28, 30, 26], "WNBA", 1, 0.13, news_minutes=34)
    assert m == 34 and sd > 0
    # SL with no games → draft-slot baseline dominates
    bg = PlayerBackground("A", draft_pick=1)
    m2, _ = MIN.project_minutes([], "NBA Summer League", 3, 0.32, background=bg)
    assert 28 <= m2 <= 32                                   # top pick ~30
    # minutes track RECENT role: a player who ramped up projects near recent minutes
    m3, _ = MIN.project_minutes([34, 33, 32, 20, 18, 16], "WNBA", 0, 0.13)
    assert m3 > 28                                          # recent (33ish) outweighs old (17ish)


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


def test_orb_share_prior_shrinks_and_uses_college():
    base = PR.orb_share_prior("C")
    assert 0.0 < base < 1.0
    # no split data → prior exactly
    assert PR.fit_orb_share([], "C") == base
    # an extreme college split pulls the SL baseline toward it, but not all the way
    bg = PlayerBackground(player="X", pre_league="NCAA", rates40={"orb": 9.0, "drb": 1.0})
    pulled = PR.orb_share_prior("C", bg)
    assert base < pulled < 0.90
    # translation-invariance: scaling orb and drb alike must not move the share
    bg2 = PlayerBackground(player="X", pre_league="NCAA", rates40={"orb": 9.0 * 0.93, "drb": 0.93})
    assert abs(PR.orb_share_prior("C", bg2) - pulled) < 1e-9


def test_value_math():
    assert abs(implied_prob(-110) - 0.5238) < 1e-3
    assert abs(to_decimal(+100) - 2.0) < 1e-9
    r = value_row("X", "pts", 15.5, "over", 100, 0.60)
    assert abs(r["edge"] - 0.10) < 1e-9 and abs(r["ev"] - 0.20) < 1e-9


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_") and callable(fn):
            fn(); print("ok", k)
