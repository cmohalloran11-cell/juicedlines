"""
Mechanism tests for the CFB engine.

WHAT THESE PROVE AND WHAT THEY DO NOT. Every number asserted here is a CODE-CORRECTNESS
result: a planted effect goes into a synthetic league (cfb/tests/fakes.py) or a hand-built
fixture, and the fit is asserted to recover it, or a closed-form expectation is computed by
hand and the code asserted to match. None of it is a claim about the engine's accuracy against
real college football. That claim cannot be made anywhere in this repository today: no
CFBD_API_KEY exists in any environment this code has run in, and there are zero graded CFB
rows in the ledger, so there is nothing to calibrate against. CFB ships honestly
unmeasurable -- see provenance.MODEL_CHANGELOG's cfb-1.0.0 entry.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from cfb.data.base import PlayerGameStats
from cfb.model import GameContext
from cfb.model import garbage_time as GT
from cfb.model import opponent as OPP
from cfb.model import pace as PACE
from cfb.model import priors as PR
from cfb.model import rates as RT
from cfb.model.config import BLOWOUT_MARGIN
from cfb.sim import engine as E
from cfb.tests import fakes as F


def _box(pid="p", team="A", game="g", **kw) -> PlayerGameStats:
    base = dict(season=2026, week=1, season_type="regular", game_id=game, player_id=pid,
                player=pid, team=team, opponent="B")
    base.update(kw)
    return PlayerGameStats(**base)


# ── empirical-Bayes shrinkage ────────────────────────────────────────────────────────────

def test_shrinkage_k_is_estimated_from_the_leagues_own_between_player_spread():
    """50 players, every rate exact (no sampling noise injected): mean 0.20, population sd
    0.05 on identical 1,000-play denominators. Method of moments then has a closed form --
    k = mu / (between - n_players*mu/total_den) = 0.20 / (0.0025 - 0.0002) = 86.96 -- and the
    fitted k must match it, not a number written into a config file."""
    totals = {}
    for i in range(50):
        rate = 0.25 if i % 2 == 0 else 0.15
        t = PR.PlayerTotals(player_id=f"p{i}", player=f"p{i}", team="A", games=10, weight=10.0)
        t.totals["plays"] = 1000.0
        t.totals["rush_attempts"] = rate * 1000.0
        totals[f"p{i}"] = t
    rows = [_box(pid=f"p{i}", game=f"g{i}-{j}") for i in range(50) for j in range(5)]
    positions = {f"p{i}": "RB" for i in range(50)}

    priors = PR.fit_positional_priors(totals, rows, positions)["RB"]
    assert priors.mean["rush_att_per_play"] == pytest.approx(0.20)
    assert priors.k["rush_att_per_play"] == pytest.approx(0.20 / 0.0023, rel=1e-6)
    assert priors.k_basis["rush_att_per_play"] == "measured"


def test_shrinkage_falls_all_the_way_to_the_league_mean_when_players_are_indistinguishable():
    """Every player on the identical rate: the data says there is no real between-player
    spread, so the honest answer is maximum shrinkage, not a small k that pretends there is."""
    totals = {}
    for i in range(50):
        t = PR.PlayerTotals(player_id=f"p{i}", player=f"p{i}", team="A", games=10, weight=10.0)
        t.totals["plays"] = 1000.0
        t.totals["rush_attempts"] = 200.0
        totals[f"p{i}"] = t
    rows = [_box(pid=f"p{i}", game=f"g{i}-{j}") for i in range(50) for j in range(5)]
    priors = PR.fit_positional_priors(totals, rows, {f"p{i}": "RB" for i in range(50)})["RB"]
    assert priors.k_basis["rush_att_per_play"] == "prior_only"
    assert priors.k["rush_att_per_play"] == pytest.approx(50 * 1000.0)


def test_a_thin_league_emits_no_prior_at_all_rather_than_one_off_a_handful_of_rows():
    totals = {"p0": PR.PlayerTotals(player_id="p0", player="p0", team="A", games=1)}
    assert PR.fit_positional_priors(totals, [_box()], {"p0": "RB"}) == {}


# ── the three prior tiers ────────────────────────────────────────────────────────────────

def _usage_priors(mean=0.10, k=100.0) -> PR.PositionPriors:
    p = PR.PositionPriors(position="RB", n_player_games=1000)
    p.mean["rush_att_per_play"] = mean
    p.k["rush_att_per_play"] = k
    p.k_basis["rush_att_per_play"] = "measured"
    return p


def _history(team="Group 1", num=120.0, den=600.0) -> PR.PlayerTotals:
    t = PR.PlayerTotals(player_id="p1", player="Back", team=team, games=12, weight=12.0)
    t.totals["plays"] = den
    t.totals["rush_attempts"] = num
    return t


def test_tier_is_classified_from_the_evidence_that_exists_not_from_a_label():
    assert RT.classify_tier(None, None, "Power 0")[0] == RT.TIER_C
    assert RT.classify_tier(None, _history(team="Power 0"), "Power 0")[0] == RT.TIER_A
    assert RT.classify_tier(None, _history(team="Group 1"), "Power 0")[0] == RT.TIER_B


def test_tier_a_blends_prior_season_production_into_the_positional_prior():
    """Hand-computed: prior_den = 1.0*600 + 100 = 700, prior_rate = (120 + 100*0.10)/700."""
    r = RT.fit_player_rates("p1", _usage_priors(), None, _history(team="Power 0"),
                            team="Power 0", position="RB",
                            carryover={"rush_att_per_play": 1.0})
    assert r.tier == RT.TIER_A
    assert r.opportunity["rush_att_per_play"] == pytest.approx(130.0 / 700.0)
    assert r.eff_n["rush_att_per_play"] == pytest.approx(700.0)


def test_tier_b_translates_a_transfers_prior_by_the_measured_level_factor():
    """Same player, same 0.20 observed rate, but the prior season was at a G5 school. The
    measured level means are 0.10 (group) and 0.08 (power), so the translation is x0.8 and the
    hand-computed prior becomes (120*0.8 + 100*0.10)/700 = 106/700, strictly below tier A's
    130/700. The discount is the ratio of two real league means, not a chosen penalty."""
    factors = {"rush_att_per_play": {"group": 0.10, "power": 0.08}}
    r = RT.fit_player_rates("p1", _usage_priors(), None, _history(team="Group 1"),
                            team="Power 0", position="RB",
                            carryover={"rush_att_per_play": 1.0}, level_factors=factors,
                            origin_tier="group", destination_tier="power")
    assert r.tier == RT.TIER_B
    assert r.level_factor_applied == pytest.approx(0.8)
    assert r.opportunity["rush_att_per_play"] == pytest.approx(106.0 / 700.0)

    same_level = RT.fit_player_rates("p1", _usage_priors(), None, _history(team="Power 9"),
                                     team="Power 0", position="RB",
                                     carryover={"rush_att_per_play": 1.0},
                                     level_factors=factors)
    assert r.opportunity["rush_att_per_play"] < same_level.opportunity["rush_att_per_play"]


def test_tier_b_applies_no_translation_when_a_level_was_never_measurable():
    r = RT.fit_player_rates("p1", _usage_priors(), None, _history(team="Group 1"),
                            team="Power 0", position="RB",
                            carryover={"rush_att_per_play": 1.0},
                            level_factors={"rush_att_per_play": {"power": 0.08}},
                            origin_tier="fcs", destination_tier="power")
    assert r.opportunity["rush_att_per_play"] == pytest.approx(130.0 / 700.0)


def test_tier_c_recentres_the_prior_on_the_recruiting_rating_fit():
    """No college production at all: the prior is the rating-implied usage (0.25*0.9 = 0.225)
    at the SAME strength k as the flat prior -- better centred, not more confident."""
    rp = PR.RecruitingPrior(position="RB", coef={"rush_att_per_play": (0.0, 0.25)}, n=80)
    r = RT.fit_player_rates("p1", _usage_priors(), None, None, team="Power 0", position="RB",
                            recruiting_prior=rp, recruiting_rating=0.9)
    assert r.tier == RT.TIER_C
    assert r.opportunity["rush_att_per_play"] == pytest.approx(0.225)
    assert r.eff_n["rush_att_per_play"] == pytest.approx(100.0)

    unrated = RT.fit_player_rates("p2", _usage_priors(), None, None, team="Power 0",
                                  position="RB", recruiting_prior=rp, recruiting_rating=None)
    assert unrated.opportunity["rush_att_per_play"] == pytest.approx(0.10)


def test_an_unmeasurable_carryover_collapses_the_prior_to_the_positional_mean():
    r = RT.fit_player_rates("p1", _usage_priors(), None, _history(team="Power 0"),
                            team="Power 0", position="RB", carryover={})
    assert r.opportunity["rush_att_per_play"] == pytest.approx(0.10)
    assert r.eff_n["rush_att_per_play"] == pytest.approx(100.0)


def test_carryover_is_measured_as_the_real_year_over_year_correlation():
    older, recent = {}, {}
    for i in range(40):
        rate = 0.05 + 0.005 * i
        for store, r in ((older, rate), (recent, rate)):
            t = PR.PlayerTotals(player_id=f"p{i}", player=f"p{i}", team="A", games=10)
            t.totals["plays"] = 800.0
            t.totals["rush_attempts"] = r * 800.0
            store[f"p{i}"] = t
    assert RT.fit_carryover(recent, older)["rush_att_per_play"] == pytest.approx(1.0)


# ── opponent adjustment ──────────────────────────────────────────────────────────────────

def test_production_against_a_weak_opponent_is_normalised_and_counts_for_less_evidence():
    """Hand-computed: a 20-carry, 125-yard game against a defense measured to inflate yards by
    25% enters as 16 carries of evidence (min(1, 1/1.25)) and 80 adjusted yards -- a 5.0
    yards-per-carry rate (125/20/1.25) on a smaller effective sample than 20 carries."""
    rows = [_box(rush_attempts=20.0, rush_yards=125.0)]
    totals = PR.accumulate_totals(rows, {("g", "A"): 70.0}, lambda r: 1.25)["p"]
    assert totals.totals["rush_attempts"] == pytest.approx(16.0)
    assert totals.totals["rush_yards"] == pytest.approx(80.0)
    assert totals.totals["plays"] == pytest.approx(56.0)
    assert totals.rate("yards_per_carry") == pytest.approx(5.0)


def test_no_opponent_factor_leaves_the_box_line_exactly_as_played():
    rows = [_box(rush_attempts=20.0, rush_yards=125.0)]
    totals = PR.accumulate_totals(rows, {("g", "A"): 70.0})["p"]
    assert totals.totals["rush_attempts"] == pytest.approx(20.0)
    assert totals.totals["rush_yards"] == pytest.approx(125.0)


def test_the_opponent_fit_recovers_the_planted_fcs_and_defense_effects():
    """cfb/tests/fakes.py plants a 1.25x FCS production inflation and a 0.60 log-yards
    response to opponent defensive PPA. Mechanism check, not an accuracy claim -- and the FCS
    arm is deliberately a small, lopsided subsample (8 games a season, always the same handful
    of hosts), because that is what a real FBS schedule looks like; recovering the planted
    value to within ~0.06 in log space off it is the fit working, not a precision claim."""
    src = F.FakeSource()
    schedule = src.schedule(F.SEASON) + src.schedule(F.SEASON - 1)
    efficiency = src.team_efficiency(F.SEASON) + src.team_efficiency(F.SEASON - 1)
    rows = src.player_game_stats(F.SEASON) + src.player_game_stats(F.SEASON - 1)
    context = {}
    for g in schedule:
        for team, home in ((g.home_team, True), (g.away_team, False)):
            opp = g.opponent_of(team)
            context[(g.id, team)] = GameContext(
                game_id=g.id, season=g.season, week=g.week, team=team, opponent=opp,
                is_home=home,
                opponent_classification=(g.away_classification if home
                                         else g.home_classification))
    model = OPP.fit_opponent_model(efficiency, rows, context)
    assert model.basis == "measured"
    assert model.fcs_coef == pytest.approx(math.log(F.FCS_INFLATION), abs=0.10)
    assert model.ppa_coef == pytest.approx(F.DEFENSE_SENSITIVITY, abs=0.15)
    assert OPP.production_factor(model, "Lower 0", "fcs") > 1.15


def test_an_unmeasured_opponent_model_applies_no_adjustment_at_all():
    assert OPP.production_factor(OPP.OpponentModel(), "anyone", "fcs") == 1.0


# ── pace ─────────────────────────────────────────────────────────────────────────────────

def _pace_model() -> PACE.PaceModel:
    return PACE.PaceModel(league_plays=70.0, team_plays={"A": 80.0, "B": 60.0},
                          market_coef=(60.0, 0.2, 0.15), plays_sd_frac=0.10,
                          n_team_games=500, n_priced_team_games=400, basis="market_priced")


def test_pace_is_the_matchup_average_of_two_shrunk_tempos_plus_the_market_delta():
    """Hand-computed: matchup base (80+60)/2 = 70, market prediction 60 + 0.2*0 + 0.15*60 = 69,
    applied as a delta from the league mean 70, so 70 + (69-70) = 69."""
    ctx = GameContext(game_id="g", season=2026, week=3, team="A", opponent="B", is_home=True,
                      spread=0.0, over_under=60.0)
    p = PACE.projected_plays(_pace_model(), "A", "B", ctx)
    assert p.plays == pytest.approx(69.0)
    assert p.sd == pytest.approx(7.0)
    assert p.basis == "market_priced"


def test_pace_falls_back_to_the_matchup_tempo_for_an_unpriced_game():
    p = PACE.projected_plays(_pace_model(), "A", "B", None)
    assert p.plays == pytest.approx(70.0) and p.basis == "matchup"


def test_pace_declines_to_project_a_league_it_has_never_observed():
    assert PACE.projected_plays(PACE.PaceModel(), "A", "B", None) is None


def test_the_pace_fit_recovers_the_planted_per_team_tempo_spread():
    src = F.FakeSource()
    efficiency = src.team_efficiency(F.SEASON) + src.team_efficiency(F.SEASON - 1)
    model = PACE.fit_pace_model(efficiency, {})
    fast = max(F.FBS_TEAMS, key=F.base_plays)
    slow = min(F.FBS_TEAMS, key=F.base_plays)
    assert model.team_plays[fast] - model.team_plays[slow] > 10.0
    assert model.team_plays[fast] == pytest.approx(F.base_plays(fast), abs=3.0)


# ── garbage time ─────────────────────────────────────────────────────────────────────────

def _garbage_model() -> GT.GarbageTimeModel:
    return GT.GarbageTimeModel(margin_sd=14.0, n_margin_games=500,
                               share_coef=(0.70, -0.10), mean_blowout_p=0.20,
                               league_starter_share=0.60, n_team_games=900, basis="measured")


def _ctx(spread):
    return GameContext(game_id="g", season=2026, week=3, team="A", opponent="B", is_home=True,
                       spread=spread, over_under=55.0)


def test_blowout_probability_is_identical_for_a_heavy_favourite_and_a_heavy_underdog():
    """P(|final margin| >= 21) is symmetric in the sign of the expected margin by construction,
    which is exactly the both-ends requirement: a 35-point favourite and a 35-point underdog
    are the same game. Hand-computed at sd=14: (1-Phi(-1)) + Phi(-4) = 0.841376, and an even game is
    2*(1-Phi(1.5)) = 0.133614."""
    m = _garbage_model()
    fav = GT.blowout_estimate(m, _ctx(35.0))
    dog = GT.blowout_estimate(m, _ctx(-35.0))
    assert fav.probability == pytest.approx(dog.probability)
    assert fav.probability == pytest.approx(0.841376, abs=1e-5)
    assert GT.blowout_estimate(m, _ctx(0.0)).probability == pytest.approx(0.133614, abs=1e-5)


def test_the_garbage_time_discount_cuts_a_starters_usage_at_both_extremes():
    """Hand-computed from the fitted line: reference share at the league-average blowout
    probability is 0.70 - 0.10*0.20 = 0.68; at p = 0.841347 it is 0.615865; the multiplier is
    0.615865/0.68 = 0.905684, i.e. a 9.43% usage cut. Identical for the 35-point favourite and
    the 35-point underdog, and it scales down for a less starter-like player."""
    m = _garbage_model()
    for spread in (35.0, -35.0):
        p = GT.blowout_estimate(m, _ctx(spread)).probability
        assert GT.usage_multiplier(m, p, 1.0) == pytest.approx(0.905684, abs=1e-5)
        assert GT.usage_multiplier(m, p, 0.5) == pytest.approx(0.952842, abs=1e-5)
    even = GT.blowout_estimate(m, _ctx(0.0)).probability
    assert GT.usage_multiplier(m, even, 1.0) > GT.usage_multiplier(
        m, GT.blowout_estimate(m, _ctx(35.0)).probability, 1.0)


def test_the_discount_reaches_the_simulated_distribution_as_fewer_carries():
    """The multiplier is only meaningful if it moves the simulation. Same rates, same pace,
    only the garbage-time multiplier differs: projected carries fall by the same 9.43%."""
    rates = RT.PlayerRates(player_id="p", opportunity={"rush_att_per_play": 0.25},
                           efficiency={"yards_per_carry": 4.5},
                           eff_n={"rush_att_per_play": 2000.0, "yards_per_carry": 500.0})
    unit_sd = {"yards_per_carry": 3.8}
    rng = np.random.default_rng(7)
    full = E.simulate(rates, 70.0, 6.0, unit_sd, 1.0, n=40000, rng=rng)
    rng = np.random.default_rng(7)
    cut = E.simulate(rates, 70.0, 6.0, unit_sd, 0.905684, n=40000, rng=rng)
    ratio = float(cut["rush_attempts"].mean()) / float(full["rush_attempts"].mean())
    assert ratio == pytest.approx(0.905684, abs=0.01)
    assert float(cut["rush_yards"].mean()) < float(full["rush_yards"].mean())


def test_starterness_scales_the_discount_by_the_players_own_measured_share():
    m = _garbage_model()
    per_player = 0.60 / 3
    assert GT.starterness(m, per_player) == pytest.approx(1.0)
    assert GT.starterness(m, per_player / 2) == pytest.approx(0.5)
    assert GT.starterness(m, None) == 0.0


def test_an_unmeasured_garbage_time_model_never_moves_a_projection():
    m = GT.GarbageTimeModel()
    assert GT.blowout_estimate(m, _ctx(35.0)) is None
    assert GT.usage_multiplier(m, 0.9, 1.0) == 1.0


def test_the_garbage_time_fit_recovers_a_planted_starter_share_drop():
    """fakes.py cuts starter usage to 0.80x in games that end up blowouts, so the fitted slope
    of starter share on blowout probability must come back negative."""
    from cfb import projections as P
    P.set_source(F.FakeSource())
    P.set_positions_override(F.positions())
    try:
        model = P.league_data(F.SEASON)["garbage"]
    finally:
        P.set_source(None)
        P.set_positions_override(None)
    assert model.basis == "measured"
    assert model.share_coef[1] < 0.0
    assert model.margin_sd == pytest.approx(15.0, abs=2.0)


# ── two-stage uncertainty ────────────────────────────────────────────────────────────────

def test_a_thin_prior_is_wider_than_a_deep_one_at_the_same_point_estimate():
    """The two-stage layer's whole purpose, and the reason the three-tier prior records its
    evidence depth rather than only its value: a true freshman priced off a recruiting rating
    and a returning starter with the IDENTICAL rate must not get the identical distribution.
    Same rates, same pace, only eff_n differs."""
    def rates(eff_n):
        return RT.PlayerRates(player_id="p", opportunity={"rush_att_per_play": 0.25},
                              efficiency={"yards_per_carry": 4.5},
                              eff_n={"rush_att_per_play": eff_n, "yards_per_carry": eff_n})
    unit_sd = {"yards_per_carry": 3.8}
    deep = E.simulate(rates(2000.0), 70.0, 6.0, unit_sd, n=60000,
                      rng=np.random.default_rng(11))
    thin = E.simulate(rates(12.0), 70.0, 6.0, unit_sd, n=60000,
                      rng=np.random.default_rng(11))

    d_mean, t_mean = float(deep["rush_yards"].mean()), float(thin["rush_yards"].mean())
    d_sd, t_sd = float(deep["rush_yards"].std()), float(thin["rush_yards"].std())
    assert t_mean == pytest.approx(d_mean, rel=0.05)      # point estimate preserved
    assert t_sd > d_sd * 1.5                              # parameter uncertainty widened it
    assert float(thin["rush_attempts"].std()) > float(deep["rush_attempts"].std())


def test_the_simulation_is_plays_times_usage_times_efficiency_not_a_per_game_mean():
    """Doubling the projected play count doubles projected carries; doubling yards-per-carry
    doubles yards without touching carries. The decomposition is load-bearing, so it is
    asserted rather than assumed."""
    rates = RT.PlayerRates(player_id="p", opportunity={"rush_att_per_play": 0.25},
                           efficiency={"yards_per_carry": 4.5},
                           eff_n={"rush_att_per_play": 5000.0, "yards_per_carry": 5000.0})
    unit_sd = {"yards_per_carry": 3.8}
    a = E.simulate(rates, 40.0, 1.0, unit_sd, n=40000, rng=np.random.default_rng(3))
    b = E.simulate(rates, 80.0, 1.0, unit_sd, n=40000, rng=np.random.default_rng(3))
    assert float(b["rush_attempts"].mean()) == pytest.approx(
        2.0 * float(a["rush_attempts"].mean()), rel=0.03)
    assert float(a["rush_attempts"].mean()) == pytest.approx(10.0, rel=0.03)


def test_anytime_td_is_a_zero_one_outcome_derived_from_the_same_trial():
    rates = RT.PlayerRates(player_id="p", opportunity={"rush_att_per_play": 0.25},
                           efficiency={"yards_per_carry": 4.5, "rush_td_per_carry": 0.05},
                           eff_n={"rush_att_per_play": 5000.0, "yards_per_carry": 5000.0,
                                  "rush_td_per_carry": 5000.0})
    sim = E.simulate(rates, 70.0, 5.0, {"yards_per_carry": 3.8}, n=20000,
                     rng=np.random.default_rng(5))
    assert set(np.unique(sim["anytime_td"])) <= {0.0, 1.0}
    assert 0.3 < float(sim["anytime_td"].mean()) < 0.9
    assert BLOWOUT_MARGIN == 21.0
