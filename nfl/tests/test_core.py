"""Unit tests for the NFL core — shrinkage, playing time, rotation tiers, matchup,
environment, correlation, simulation and confidence. All deterministic / offline."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nfl import COMBOS
from nfl.config import cfg, position_cfg
from nfl.data.base import PlayerWeek
from nfl.model import usage as U
from nfl.model import playing_time as PT
from nfl.model import rotation as ROT
from nfl.model import matchup as MU
from nfl.model import environment as ENV
from nfl.model.combo_corr import empirical_combo_corr, pooled_default
from nfl.sim import engine as E
from nfl import projections as P
from nfl.tests import fakes as F


# ── shrinkage ────────────────────────────────────────────────────────────────────────────

def test_shrinkage_pulls_toward_the_prior_and_more_so_for_a_thinner_sample():
    priors = U.prior_for("WR")
    weeks, snaps = F.full_season(n=12, targets=12.0, receptions=8.0, rec_yards=140.0)
    idx = U.snap_index(snaps)
    deep = U.fit_usage(weeks, idx, "WR", priors).opportunity["targets_per_snap"]
    thin = U.fit_usage(weeks[:1], idx, "WR", priors).opportunity["targets_per_snap"]
    raw = 12.0 / snaps[0].offense_snaps
    assert priors["targets_per_snap"] < thin < deep < raw
    # and the fit never runs away from the prior when there is no sample at all
    none = U.prior_only_usage("WR", priors).opportunity["targets_per_snap"]
    assert none == pytest.approx(priors["targets_per_snap"])


def test_eff_n_records_real_evidence_and_grows_with_the_sample():
    priors = U.prior_for("WR")
    weeks, snaps = F.full_season(n=12, targets=8.0, receptions=5.0, rec_yards=70.0)
    idx = U.snap_index(snaps)
    thin = U.fit_usage(weeks[:2], idx, "WR", priors)
    deep = U.fit_usage(weeks, idx, "WR", priors)
    k = position_cfg("shrink_k", "WR")["targets_per_snap"]
    assert thin.eff_n["targets_per_snap"] > k          # own snaps added on top of the prior
    assert deep.eff_n["targets_per_snap"] > thin.eff_n["targets_per_snap"]
    assert deep.sample_weight > thin.sample_weight


def test_positional_priors_need_a_real_sample_before_overriding_the_measured_table():
    weeks, snaps = F.full_season(n=3, targets=9.0, receptions=6.0, rec_yards=95.0)
    assert U.positional_priors(weeks, snaps) == {}, "3 games must not redefine a league prior"
    big_w, big_s = F.league_filler(n_players=40, weeks_per=6)
    live = U.positional_priors(big_w, big_s)
    assert "WR" in live and live["WR"]["targets_per_snap"] > 0


def test_a_game_with_no_snap_row_does_not_inflate_a_per_snap_rate():
    """A missing snap row must drop the game from the per-snap DENOMINATOR only — it must not
    keep the numerator, which would silently raise the rate."""
    priors = U.prior_for("RB")
    weeks, snaps = F.full_season("Bell Cow", "RB", n=6, carries=18.0, rush_yards=80.0)
    idx = U.snap_index(snaps)
    full = U.fit_usage(weeks, idx, "RB", priors).opportunity["carries_per_snap"]
    partial = U.fit_usage(weeks, U.snap_index(snaps[:3]), "RB", priors).opportunity["carries_per_snap"]
    assert partial < full, "dropping snap rows must not raise the per-snap rate"


# ── playing time ─────────────────────────────────────────────────────────────────────────

def test_playing_time_range_is_wider_for_a_thin_sample_at_the_same_point_estimate():
    thin = PT.project_playing_time([0.80], "WR", depth_rank=1, expected_team_snaps=62.0)
    deep = PT.project_playing_time([0.80] * 10, "WR", depth_rank=1, expected_team_snaps=62.0)
    lo_t, hi_t = thin.snap_share_range()
    lo_d, hi_d = deep.snap_share_range()
    assert (hi_t - lo_t) > (hi_d - lo_d), "less evidence must mean a wider 80% band"
    assert deep.concentration > thin.concentration


def test_sampled_snap_share_preserves_the_mean_while_widening():
    rng = np.random.default_rng(0)
    thin = PT.project_playing_time([0.60], "RB", depth_rank=1, expected_team_snaps=62.0)
    deep = PT.project_playing_time([0.60] * 12, "RB", depth_rank=1, expected_team_snaps=62.0)
    a = PT.sample_snap_share(thin, 60000, rng)
    b = PT.sample_snap_share(deep, 60000, rng)
    assert a.mean() == pytest.approx(thin.expected_snap_share, abs=0.01)
    assert b.mean() == pytest.approx(deep.expected_snap_share, abs=0.01)
    assert a.std() > b.std()


def test_playing_time_falls_back_through_three_evidence_levels():
    own = PT.project_playing_time([0.9, 0.88], "WR", depth_rank=1)
    assert own.basis == "own_snap_history"
    depth = PT.project_playing_time([], "WR", depth_rank=2)
    assert depth.basis == "depth_chart"
    assert depth.expected_snap_share == pytest.approx(cfg("depth_rank_snap_share")["WR"][2][0])
    nothing = PT.project_playing_time([], "WR", depth_rank=None)
    assert nothing.basis == "positional_prior" and nothing.role == "unknown"
    assert nothing.confidence == "low"


def test_playing_time_probability_is_none_without_a_snap_feed_never_assumed_available():
    pt = PT.project_playing_time([], "WR", depth_rank=1)
    assert pt.playing_time_probability is None
    played = PT.project_playing_time([0.8] * 5, "WR", 1, active_games=5, team_games=5)
    missed = PT.project_playing_time([0.8] * 2, "WR", 1, active_games=2, team_games=5)
    assert played.playing_time_probability > missed.playing_time_probability


def test_depth_rank_3_and_beyond_are_pooled():
    assert PT.depth_rank_prior("RB", 5) == PT.depth_rank_prior("RB", 3)
    assert PT.depth_rank_prior("RB", None) is None
    assert PT.depth_rank_prior("XX", 1) is None


# ── preseason rotation ───────────────────────────────────────────────────────────────────

def test_rotation_tiers_classify_off_depth_chart_plus_real_workload():
    tier, why = ROT.classify_tier(1, 0.85, 12)
    assert tier == "confirmed_starter" and "85%" in why
    assert ROT.classify_tier(1, None, 0)[0] == "likely_starter"
    assert ROT.classify_tier(1, 0.85, 1)[0] == "likely_starter", "one game can't confirm"
    assert ROT.classify_tier(4, 0.1, 10)[0] == "third_team"
    assert ROT.classify_tier(None, None, 0, on_roster=True)[0] == "fringe"
    assert ROT.classify_tier(None, None, 0, on_roster=False)[0] == "unknown"


def test_rank_2_splits_into_first_team_rotation_vs_second_team_on_real_evidence():
    """The tier a rank-2 depth-chart player lands in now depends on whether he has a real,
    qualifying prior-season workload behind him — collapsing both into one "second_team"
    tier (the old 6-tier model) threw away real depth-chart+history information."""
    tier, why = ROT.classify_tier(2, 0.40, 10)
    assert tier == "first_team_rotation" and "40%" in why
    assert ROT.classify_tier(2, 0.40, 2)[0] == "second_team", "not enough games to confirm"
    assert ROT.classify_tier(2, 0.10, 10)[0] == "second_team", "share too low to be rotational"
    assert ROT.classify_tier(2, None, 0)[0] == "second_team"


def test_each_rotation_tier_has_its_own_distribution_shape_and_fringe_is_widest():
    widths = {}
    for tier in ROT.TIERS:
        pt = ROT.tier_playing_time(tier, expected_team_snaps=60.0)
        lo, hi = pt.snap_share_range()
        widths[tier] = hi - lo
        assert pt.basis == "preseason_tier" and pt.confidence == "low"
    assert widths["confirmed_starter"] < widths["second_team"] < widths["unknown"]
    assert widths["confirmed_starter"] < widths["fringe"]
    assert len(ROT.TIERS) == 7, "TIER1..TIER7 from confirmed_starter through unknown"


def test_seven_tiers_produce_seven_distinct_expected_snaps_for_the_same_position():
    """The actual bug this module was rebuilt to fix: the old model's confirmed_starter and
    third_team tiers shared the same mean snap SHARE (0.30), so multiplying by the one
    constant preseason team-snaps number (no schedule ever varies it) produced the literal
    identical expected_snaps for both tiers. Drives x position-snaps-per-drive structurally
    can't collide that way."""
    snaps = [ROT.tier_playing_time(t, expected_team_snaps=65.3, position="WR").expected_snaps
            for t in ROT.TIERS]
    assert len(set(snaps)) == len(ROT.TIERS), f"expected 7 distinct values, got {snaps}"


def test_preseason_risk_rises_as_the_tier_gets_less_certain():
    risks = [ROT.preseason_risk(t, ROT.tier_playing_time(t)) for t in
             ("confirmed_starter", "second_team", "third_team", "unknown")]
    assert risks == sorted(risks)
    assert 0.0 <= risks[0] <= 1.0 and risks[-1] <= 1.0


def test_team_tendency_reports_unmeasurable_preseason_fields_as_none_with_a_reason():
    weeks, _ = F.full_season("QB One", "QB", "CIN", n=10, pass_attempts=35.0, carries=4.0,
                             sacks=2.0)
    t = ROT.team_tendency("CIN", weeks, [F.game()])
    assert t.preseason_pass_rate is not None and t.preseason_rush_rate is not None
    assert "regular_season_measured" in t.basis["preseason_pass_rate"]
    assert t.coach == "Home Coach"
    for f in ("starter_snap_rate", "avg_first_team_drives", "qb_rotation_pattern",
              "second_team_usage"):
        assert getattr(t, f) is None
        assert "preseason" in t.reasons[f]


def test_position_changes_expected_snaps_within_the_same_tier():
    """A QB in the game plays every snap of his own drives (no within-drive rotation is
    possible at the position); RB/WR/TE corps rotate by personnel package even within a
    single drive they're nominally part of — so the SAME tier must produce a DIFFERENT
    expected_snaps for a QB than for a skill position."""
    qb = ROT.tier_playing_time("likely_starter", expected_team_snaps=65.0, position="QB")
    wr = ROT.tier_playing_time("likely_starter", expected_team_snaps=65.0, position="WR")
    rb = ROT.tier_playing_time("likely_starter", expected_team_snaps=65.0, position="RB")
    assert qb.expected_snaps != wr.expected_snaps != rb.expected_snaps
    assert qb.expected_snaps > wr.expected_snaps, "a QB plays every snap of his drives"


def test_team_rotation_nudge_is_signed_capped_and_position_scoped():
    cap = float(cfg("preseason_tiers", "team_pass_rate_nudge_cap"))
    lg = float(cfg("preseason_tiers", "team_pass_rate_league_mean"))
    # A pass-heavy team nudges WR/TE UP and RB DOWN; a run-heavy team the reverse.
    assert ROT.team_rotation_nudge("WR", lg + 0.30) > 0
    assert ROT.team_rotation_nudge("RB", lg + 0.30) < 0
    assert abs(ROT.team_rotation_nudge("WR", lg + 5.0)) == pytest.approx(cap)
    assert ROT.team_rotation_nudge("QB", lg + 0.30) == 0.0, "doesn't change a QB's own snaps"
    assert ROT.team_rotation_nudge("WR", None) == 0.0, "no signal -> no nudge, not a guess"


def test_historical_preseason_snap_share_is_real_and_tested_but_always_empty_from_nflverse():
    assert ROT.historical_preseason_snap_share(None) is None
    assert ROT.historical_preseason_snap_share([]) is None
    mean, n = ROT.historical_preseason_snap_share([0.40, 0.30, 0.50])
    assert n == 3 and 0.30 < mean < 0.50   # recency-weighted toward the most-recent game


def test_own_preseason_history_overrides_the_tier_prior_when_it_exists():
    """Top of the hierarchy: a player's own real preseason usage — when a source publishes
    it — must move the projection, not just sit next to the tier estimate unused."""
    no_history = ROT.tier_playing_time("second_team", expected_team_snaps=65.0, position="WR")
    high_history = ROT.tier_playing_time("second_team", expected_team_snaps=65.0, position="WR",
                                         own_preseason_snap_sample=[0.55, 0.60, 0.58, 0.52])
    assert high_history.basis == "preseason_own_history"
    assert high_history.confidence in ("medium", "high")
    assert high_history.expected_snaps > no_history.expected_snaps
    assert high_history.n_games == 4


def test_prior_influence_weights_sum_to_about_one_and_shift_toward_historical_when_present():
    no_hist = ROT.prior_influence_weights(depth_rank=2, own_preseason_n=0,
                                          team_rotation_nudge_magnitude=0.0)
    assert sum(no_hist.values()) == pytest.approx(1.0, abs=0.01)
    assert no_hist["historical_usage_weight"] == 0.0
    with_hist = ROT.prior_influence_weights(depth_rank=2, own_preseason_n=8,
                                            team_rotation_nudge_magnitude=0.0)
    assert sum(with_hist.values()) == pytest.approx(1.0, abs=0.01)
    assert with_hist["historical_usage_weight"] > no_hist["historical_usage_weight"]
    no_depth = ROT.prior_influence_weights(depth_rank=None, own_preseason_n=0,
                                           team_rotation_nudge_magnitude=0.0)
    assert no_depth["depth_chart_weight"] == 0.0, "no real depth-chart signal -> zero weight"


def test_snap_diagnostic_traces_every_field_to_a_real_input():
    pt = ROT.tier_playing_time("first_team_rotation", expected_team_snaps=65.0, position="RB")
    tendency = ROT.TeamTendency(team="CIN", preseason_pass_rate=0.58,
                                basis={"preseason_pass_rate": "regular_season_measured (own team)"})
    diag = ROT.snap_diagnostic("first_team_rotation", "listed second with real usage", "RB", pt,
                               tendency, {"p10": 10.0, "p90": 30.0}, "low", 2, 0, 0.05)
    assert diag["depth_chart_tier"] == "first_team_rotation"
    assert diag["expected_snaps"] == pt.expected_snaps
    assert diag["snap_percentiles"] == {"p10": 10.0, "p90": 30.0}
    assert "58" in diag["team_rotation"] or "0.58" in diag["team_rotation"] \
        or "58.0" in diag["team_rotation"]
    assert diag["historical_usage"].startswith("None")
    assert set(diag["prior_influence"]) == {"historical_usage_weight", "depth_chart_weight",
                                            "team_rotation_weight", "position_prior_weight",
                                            "league_prior_weight"}


def test_snap_clustering_report_detects_a_degenerate_distribution_vs_a_healthy_one():
    degenerate = [19.6] * 25 + [26.1] * 8 + [18.3]     # the OLD model's real live 2026-08 output
    report = ROT.snap_clustering_report(degenerate)
    assert report["n_distinct_values"] == 3
    assert report["pct_within_2_of_target"] == pytest.approx(76.5, abs=0.5)

    healthy = [12.3, 13.9, 20.5, 26.7, 21.3, 11.5, 15.6, 9.8, 24.1, 17.2, 30.0, 6.5]
    report2 = ROT.snap_clustering_report(healthy)
    assert report2["n_distinct_values"] == len(healthy)
    assert report2["pct_within_2_of_target"] < report["pct_within_2_of_target"]
    assert ROT.snap_clustering_report([]) == {}


def test_tier_share_measurement_pipeline_returns_nothing_when_no_preseason_rows_exist():
    """The pipeline that WOULD replace the assumed preseason table. Today it correctly
    produces {} because nflverse publishes no PRE snap rows — the honest answer."""
    from nfl.data.base import DepthChartEntry
    _, snaps = F.full_season(n=40)
    depth = [DepthChartEntry(season=F.SEASON, team="CIN", position="WR", rank=1,
                             player="Star Receiver")]
    assert ROT.measure_tier_shares(snaps, depth, season_type="PRE") == {}
    measured = ROT.measure_tier_shares(snaps, depth, season_type="REG")
    assert measured["likely_starter"][0] == pytest.approx(0.85)
    assert measured["likely_starter"][2] == 40


# ── matchup ──────────────────────────────────────────────────────────────────────────────

def _defense_weeks(team_allowed_ypc, n=40):
    return [PlayerWeek(season=F.SEASON, week=w % 17 + 1, season_type="REG",
                       game_id=f"g{w}", player_id="x", player="X", position="RB",
                       team="BUF", opponent="NYJ", carries=25.0,
                       rush_yards=25.0 * team_allowed_ypc, pass_attempts=35.0,
                       pass_yards=35.0 * 7.0533, completions=22.0, sacks=2.0)
            for w in range(n)]


def test_matchup_multiplier_is_shrunk_by_measured_reliability_not_applied_at_face_value():
    tough = MU.fit_defense(_defense_weeks(3.5))["NYJ"]
    soft = MU.fit_defense(_defense_weeks(5.5))["NYJ"]
    m_tough = MU.matchup_multipliers(tough)["yards_per_carry"]
    m_soft = MU.matchup_multipliers(soft)["yards_per_carry"]
    assert m_tough < 1.0 < m_soft
    rel = cfg("defense", "reliability", "yards_per_carry")
    raw = 5.5 / cfg("defense", "league", "yards_per_carry")
    assert m_soft == pytest.approx(1.0 + rel * (raw - 1.0), abs=1e-3)
    assert m_soft < raw, "the raw ratio must never be applied at face value"


def test_matchup_is_neutral_when_the_opponent_is_unknown_or_too_thin():
    assert MU.matchup_multipliers(None) == {}
    thin = MU.fit_defense(_defense_weeks(5.5, n=2))["NYJ"]
    assert "yards_per_carry" not in MU.matchup_multipliers(thin)
    assert MU.pass_rush_pressure(None) is None


def test_matchup_multiplier_is_clamped_to_the_measured_observed_range():
    absurd = MU.fit_defense(_defense_weeks(20.0))["NYJ"]
    lo, hi = cfg("defense", "clamp", "yards_per_carry")
    assert MU.matchup_multipliers(absurd)["yards_per_carry"] <= hi


# ── environment ──────────────────────────────────────────────────────────────────────────

def test_environment_reads_spread_and_total_off_the_schedule_from_both_sides():
    g = F.game(home="CIN", away="CLE", spread=-3.5, total=45.5)
    home = ENV.game_environment(g, "CIN")
    away = ENV.game_environment(g, "CLE")
    assert home.spread == -3.5 and away.spread == 3.5
    assert home.team_total == pytest.approx(45.5 / 2 - 3.5 / 2)
    assert away.team_total == pytest.approx(45.5 / 2 + 3.5 / 2)
    assert home.team_total + away.team_total == pytest.approx(45.5)
    assert away.expected_team_snaps > home.expected_team_snaps      # the favourite runs more
    assert away.expected_dropback_share < home.expected_dropback_share
    assert home.basis == "market_priced"


def test_environment_degrades_to_the_league_baseline_with_no_schedule_row():
    e = ENV.game_environment(None, "CIN", season_type="preseason")
    assert e.game_total is None and e.team_total is None and e.spread is None
    assert e.expected_team_snaps == pytest.approx(cfg("environment", "league_snaps_mean"),
                                                  abs=0.01)
    assert e.expected_scrimmage_plays == pytest.approx(cfg("environment", "league_plays_mean"),
                                                       abs=0.01)
    assert e.basis == "league_baseline" and e.weather is None


def test_offensive_snaps_and_scrimmage_plays_are_kept_as_separate_quantities():
    """Conflating them under-counts every per-snap projection by ~4.5% — the snap feed counts
    penalties/kneels/spikes that a scrimmage play doesn't. MEASURED ratio 1.0475."""
    e = ENV.game_environment(F.game(spread=0.0, total=44.0), "CIN")
    assert e.expected_team_snaps > e.expected_scrimmage_plays
    ratio = e.expected_team_snaps / e.expected_scrimmage_plays
    assert ratio == pytest.approx(1.0475, abs=0.02)


def test_high_wind_shifts_the_pass_rate_by_the_measured_gap_only_outdoors():
    calm = ENV.game_environment(F.game(roof="outdoors", wind=3.0), "CIN")
    windy = ENV.game_environment(F.game(roof="outdoors", wind=22.0), "CIN")
    dome = ENV.game_environment(F.game(roof="dome", wind=22.0), "CIN")
    gap = cfg("environment", "high_wind_dropback_share") - cfg("environment", "normal_dropback_share")
    assert windy.expected_dropback_share == pytest.approx(calm.expected_dropback_share + gap, abs=1e-3)
    assert dome.expected_dropback_share == pytest.approx(calm.expected_dropback_share, abs=1e-3)
    assert dome.weather == "indoors"


# ── simulation ───────────────────────────────────────────────────────────────────────────

def _sim_inputs(position="WR", n_games=10, offense_pct=0.85, **stats):
    priors = U.prior_for(position)
    weeks, snaps = F.full_season("Star", position, n=n_games, offense_pct=offense_pct, **stats)
    fit = U.fit_usage(weeks, U.snap_index(snaps), position, priors)
    env = ENV.game_environment(F.game(), "CIN")
    pt = PT.project_playing_time(fit.snap_sample, position, 1, env.expected_team_snaps)
    return fit, pt, env


def test_simulation_emits_every_supported_market_with_a_full_quantile_summary():
    fit, pt, env = _sim_inputs(targets=8.0, receptions=5.0, rec_yards=70.0)
    sim = E.simulate(fit, pt, env, n=6000, rng=np.random.default_rng(0))
    for key in ("targets", "receptions", "rec_yards", "rush_attempts", "rush_yards"):
        assert key in sim and len(sim[key]) == 6000
    s = E.summary(sim["rec_yards"])
    assert set(s) == {"mean", "median", "std_dev", "p10", "p25", "p50", "p75", "p90"}
    assert s["p10"] <= s["p25"] <= s["p50"] <= s["p75"] <= s["p90"]
    combo = E.market_array(sim, "rush_rec_yards")
    assert combo.mean() == pytest.approx(sum(sim[p].mean() for p in COMBOS["rush_rec_yards"]),
                                         abs=1e-6)


def test_counts_are_discrete_and_yards_are_continuous_and_non_negative():
    fit, pt, env = _sim_inputs(targets=8.0, receptions=5.0, rec_yards=70.0)
    sim = E.simulate(fit, pt, env, n=4000, rng=np.random.default_rng(1))
    assert np.all(sim["targets"] == np.rint(sim["targets"]))
    assert np.all(sim["receptions"] <= sim["targets"]), "receptions can never exceed targets"
    assert np.all(sim["rec_yards"] >= 0.0)
    assert np.any(sim["rec_yards"] % 1 != 0), "yards must be continuous, not a count"
    assert np.all(sim["rec_yards"][sim["targets"] == 0] == 0.0)


def test_two_stage_uncertainty_widens_a_thin_sample_without_moving_the_mean():
    """The codebase's two-stage pattern (MLB/WNBA/tennis) applied to NFL: a per-trial
    parameter draw scaled by real evidence, BEFORE the outcome draw. Identical point estimate,
    different evidence -> different spread, same mean."""
    priors = U.prior_for("WR")
    weeks, snaps = F.full_season(n=14, targets=8.0, receptions=5.0, rec_yards=70.0)
    fit = U.fit_usage(weeks, U.snap_index(snaps), "WR", priors)
    env = ENV.game_environment(F.game(), "CIN")
    pt = PT.project_playing_time([0.85] * 14, "WR", 1, env.expected_team_snaps)

    thin = U.UsageRates(player="X", player_id="x", position="WR", team="CIN",
                        opportunity=dict(fit.opportunity), efficiency=dict(fit.efficiency),
                        eff_n={k: 12.0 for k in fit.eff_n})
    deep = U.UsageRates(player="X", player_id="x", position="WR", team="CIN",
                        opportunity=dict(fit.opportunity), efficiency=dict(fit.efficiency),
                        eff_n={k: 8000.0 for k in fit.eff_n})
    kw = dict(playing_time=pt, environment=env, n=40000)
    a = E.simulate(thin, rng=np.random.default_rng(3), **kw)["rec_yards"]
    b = E.simulate(deep, rng=np.random.default_rng(3), **kw)["rec_yards"]
    assert a.std() > b.std() * 1.05, "a thin sample must carry more spread"
    assert a.mean() == pytest.approx(b.mean(), rel=0.06), "spread fix, not a bias shift"


def test_game_script_moves_carries_and_targets_in_opposite_directions():
    """nflverse's spread_line is from the HOME team's view, so a positive spread makes CIN the
    favourite. The measured coefficients say a favourite runs more plays AND throws less, so
    carries must rise and targets must fall — not both move together."""
    fit, pt, _ = _sim_inputs("RB", carries=16.0, rush_yards=70.0, targets=3.0,
                             receptions=2.0, rec_yards=18.0)
    favourite = ENV.game_environment(F.game(spread=9.5, total=48.5), "CIN")
    underdog = ENV.game_environment(F.game(spread=-9.5, total=48.5), "CIN")
    assert favourite.expected_team_snaps > underdog.expected_team_snaps
    assert favourite.expected_dropback_share < underdog.expected_dropback_share
    rng = lambda: np.random.default_rng(7)
    fav = E.simulate(fit, pt, favourite, n=25000, rng=rng())
    dog = E.simulate(fit, pt, underdog, n=25000, rng=rng())
    assert fav["rush_attempts"].mean() > dog["rush_attempts"].mean()
    assert fav["targets"].mean() < dog["targets"].mean()


def test_matchup_multiplier_moves_yards_but_not_opportunity():
    fit, pt, env = _sim_inputs("RB", carries=16.0, rush_yards=70.0)
    rng = lambda: np.random.default_rng(11)
    base = E.simulate(fit, pt, env, n=20000, rng=rng())
    soft = E.simulate(fit, pt, env, {"yards_per_carry": 1.20}, n=20000, rng=rng())
    assert soft["rush_yards"].mean() > base["rush_yards"].mean() * 1.1
    assert soft["rush_attempts"].mean() == pytest.approx(base["rush_attempts"].mean(), rel=0.02)


# ── correlation ──────────────────────────────────────────────────────────────────────────

def test_combo_corr_skips_a_thin_sample_and_shrinks_toward_the_measured_default():
    weeks, _ = F.full_season("Back", "RB", n=5, rush_yards=70.0, rec_yards=20.0)
    assert empirical_combo_corr(weeks) is None, "5 games is too thin to measure"
    rng = np.random.default_rng(0)
    varied = [F.week("Back", "RB", wk=i + 1, rush_yards=float(40 + 8 * i),
                     rec_yards=float(10 + 3 * ((i * 7) % 5))) for i in range(30)]
    m = empirical_combo_corr(varied)
    assert m is not None and m.shape == (2, 2)
    assert m[0, 0] == 1.0 and m[0, 1] == pytest.approx(m[1, 0])
    flat = [F.week("Back", "RB", wk=i + 1, rush_yards=50.0, rec_yards=10.0) for i in range(30)]
    assert empirical_combo_corr(flat) is None, "a constant column has no defined correlation"


def test_induced_correlation_preserves_every_marginal_and_widens_the_combo():
    fit, pt, env = _sim_inputs("RB", carries=16.0, rush_yards=70.0, targets=4.0,
                               receptions=3.0, rec_yards=28.0)
    r = pooled_default()
    target = np.array([[1.0, r], [r, 1.0]])
    plain = E.simulate(fit, pt, env, n=30000, rng=np.random.default_rng(5))
    corr = E.simulate(fit, pt, env, n=30000, rng=np.random.default_rng(5), combo_corr=target)
    for k in ("rush_yards", "rec_yards"):
        assert np.array_equal(np.sort(plain[k]), np.sort(corr[k]))
    before = np.corrcoef(plain["rush_yards"], plain["rec_yards"])[0, 1]
    after = np.corrcoef(corr["rush_yards"], corr["rec_yards"])[0, 1]
    assert after > before
    a, b = E.market_array(plain, "rush_rec_yards"), E.market_array(corr, "rush_rec_yards")
    assert b.mean() == pytest.approx(a.mean(), rel=0.02)
    assert b.std() > a.std()


def test_optimizer_tags_only_measured_same_team_nfl_relationships():
    import optimizer
    qb = {"sport": "NFL", "player": "QB One", "team": "CIN", "stat": "Passing Yards"}
    wr = {"sport": "NFL", "player": "WR One", "team": "CIN", "stat": "Receiving Yards"}
    rb = {"sport": "NFL", "player": "RB One", "team": "CIN", "stat": "Rushing Attempts"}
    other = {"sport": "NFL", "player": "WR Two", "team": "BUF", "stat": "Receiving Yards"}
    assert optimizer.correlation_tag(qb, wr) == "nfl_qb_pass_yards__receiver_rec_yards"
    assert optimizer._NFL_MEASURED_RHO[optimizer.correlation_tag(qb, wr)] == 0.3156
    assert optimizer._NFL_MEASURED_RHO[optimizer.correlation_tag(qb, rb)] < 0, \
        "dropbacks displace carries — the measured relationship is negative"
    # a cross-team pair is never given a measured relationship
    assert optimizer.correlation_tag(qb, other) == "independent"
    # an unmapped same-team pair keeps the generic categorical tag
    assert optimizer.correlation_tag(
        {"sport": "NFL", "team": "CIN", "stat": "Receptions", "player": "A"},
        {"sport": "NFL", "team": "CIN", "stat": "Targets", "player": "B"}) == "teammate_stack"
    # a non-NFL same-team pair is untouched
    assert optimizer.correlation_tag(
        {"sport": "WNBA", "team": "LV", "stat": "Points", "player": "A"},
        {"sport": "WNBA", "team": "LV", "stat": "Rebounds", "player": "B"}) == "teammate_stack"


def test_optimizer_correlated_simulation_uses_the_measured_rho():
    import optimizer
    legs = [{"sport": "NFL", "player": "QB One", "team": "CIN", "stat": "Passing Yards"},
            {"sport": "NFL", "player": "WR One", "team": "CIN", "stat": "Receiving Yards"}]
    m = optimizer._corr_matrix(legs, allow_correlated=True)
    assert m[0, 1] == pytest.approx(0.3156)
    assert optimizer._corr_matrix(legs, allow_correlated=False)[0, 1] == 0.0


# ── market resolution ────────────────────────────────────────────────────────────────────

def test_market_resolution():
    assert P._resolve_market("Passing Yards") == "pass_yards"
    assert P._resolve_market("Pass Attempts") == "pass_attempts"
    assert P._resolve_market("Completions") == "completions"
    assert P._resolve_market("Passing TDs") == "pass_tds"
    assert P._resolve_market("Rushing Yards") == "rush_yards"
    assert P._resolve_market("Rush Attempts") == "rush_attempts"
    assert P._resolve_market("Carries") == "rush_attempts"
    assert P._resolve_market("Receptions") == "receptions"
    assert P._resolve_market("Receiving Yards") == "rec_yards"
    assert P._resolve_market("Targets") == "targets"
    assert P._resolve_market("Rush+Rec Yards") == "rush_rec_yards"
    assert P._resolve_market("Rushing + Receiving Yards") == "rush_rec_yards"
    # deliberately unsupported: rushing/receiving TD props (probability model unproven) and
    # partial-game markets must resolve to None, never onto a modelled market
    for label in ("Rushing TDs", "Receiving TDs", "Anytime Touchdown", "1H Passing Yards",
                  "Longest Reception", "Fantasy Score", "Sacks Taken"):
        assert P._resolve_market(label) is None, label


def test_season_long_futures_never_resolve_to_a_per_game_market():
    """Found live 2026-08 in production verification: Underdog ships rest-of-season futures
    ("Season Receiving Yards", "Season Pass Tds", ...) under the same "NFL" league as real
    per-game props, distinguished only by this "season" qualifier in the label text -- and
    four of them (the yardage ones + "Season Pass Tds") were silently matching the per-game
    market before this guard, producing a projection for a single game's model run against a
    rest-of-season total. 110 of 600 live NFL lines were affected at the time this was found."""
    for label in ("Season Receiving Yards", "Season Rush Yards", "Season Pass Yards",
                  "Season Pass Tds", "Season Rec Tds", "Season Rush Tds", "Season Sacks",
                  "Regular Season Games Started"):
        assert P._resolve_market(label) is None, label
    # a preseason GAME's own per-game props are a different thing entirely and must still
    # resolve normally -- "preseason" is not the token "season"
    assert P._resolve_market("Receiving Yards") == "rec_yards"


def test_position_gating_stops_a_receiver_being_projected_for_pass_attempts():
    assert P.supports("QB", "pass_attempts") and not P.supports("WR", "pass_attempts")
    assert P.supports("RB", "rush_rec_yards") and not P.supports("QB", "rush_rec_yards")
    assert P.supports("TE", "receptions") and not P.supports("TE", "rush_attempts")


def test_team_alias_normalisation_matches_the_nflverse_codes():
    assert P.norm_team("LAR") == "LA" and P.norm_team("JAC") == "JAX"
    assert P.norm_team("WSH") == "WAS" and P.norm_team("CIN") == "CIN"
    assert P.norm_team(None) is None


def test_current_season_labels_january_as_the_previous_season():
    assert P.current_season("2027-01-15T00:00:00+00:00") == 2026
    assert P.current_season("2026-08-15T00:00:00+00:00") == 2026
    assert P.current_season("2026-12-01T00:00:00+00:00") == 2026
