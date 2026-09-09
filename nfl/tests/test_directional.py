"""
Directional and extreme-value regression tests for the NFL engine.

WHAT THESE PROVE AND WHAT THEY DO NOT. Every assertion here is a MECHANISM check: an input is
moved in a known direction against a synthetic fixture (nfl/tests/fakes.py) and the projection
is asserted to respond in the direction the engine's own measured coefficients say it should.
None of it is a claim about the engine's live accuracy, calibration or edge against real NFL
outcomes — that claim can only come from graded rows in the ledger scoped to a model_version
(backtest.py), never from a synthetic fixture. What these catch is a future regression: a sign
flip, a dropped multiplier, a rate wired to the wrong denominator.

Directions are read off the modules' own fitted numbers rather than guessed — model/
environment.py's coefficients (a favourite runs MORE plays and throws LESS; a higher total
means more plays and more passing) and model/matchup.py's reliability-shrunk multipliers.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nfl import projections as P
from nfl.config import cfg, position_cfg
from nfl.data import espn as ESPN
from nfl.data import set_source
from nfl.data.base import PlayerWeek
from nfl.model import environment as ENV
from nfl.model import usage as U
from nfl.tests import fakes as F


@pytest.fixture(autouse=True)
def _isolated_source():
    ESPN.set_test_override({}, {})
    yield
    P.clear_cache()
    set_source(None)
    ESPN.set_test_override(None, None)


def _install(specs, schedule, extra_weeks=(), filler=True):
    weeks, snaps, depth, rosters = F.offense(specs)
    if filler:
        fw, fs = F.league_filler(n_players=40, weeks_per=6)
        weeks, snaps = weeks + fw, snaps + fs
    P.clear_cache()
    set_source(F.FakeSource(weeks + list(extra_weeks), snaps, list(schedule), depth, rosters))


def _project(name, position, game, team="CIN", n=20000, seed=0):
    """Common random numbers across scenarios: the same seed for both sides of a directional
    comparison, so the difference asserted is the mechanism and not Monte-Carlo noise."""
    proj = P.project_player(name, team=team, position=position, game=game, n=n,
                            rng=np.random.default_rng(seed))
    assert proj is not None, name
    return proj


def _wr(name, offense_pct, targets, yards_per_target=8.0, depth_rank=1):
    return dict(player=name, position="WR", offense_pct=offense_pct, depth_rank=depth_rank,
                targets=targets, receptions=round(0.65 * targets, 2),
                rec_yards=round(yards_per_target * targets, 2))


# ── playing time ─────────────────────────────────────────────────────────────────────────

def test_a_bigger_recent_snap_share_sample_raises_expected_snaps_and_volume():
    """Two receivers with an IDENTICAL per-snap target rate (0.13) and an identical depth-chart
    rank, differing only in the snap share their own recent games show. Direction check: more
    playing time must mean more snaps and more targets, and the fitted per-snap rate must not
    move, or playing time and opportunity are not actually separable the way usage.py claims."""
    rate, snaps_per_game = 0.13, 62
    specs = [_wr("Rotational Wideout", 0.45, round(rate * 0.45 * snaps_per_game, 2)),
             _wr("Every Down Wideout", 0.85, round(rate * 0.85 * snaps_per_game, 2))]
    game = F.game(home="CIN", away="CLE")
    _install(specs, [game])

    low = _project("Rotational Wideout", "WR", game)
    high = _project("Every Down Wideout", "WR", game)

    assert low["playing_time"].expected_snap_share < high["playing_time"].expected_snap_share
    assert low["expected_snaps"] < high["expected_snaps"]
    assert low["expected_targets"] < high["expected_targets"]
    assert low["expected_routes"] < high["expected_routes"]
    assert (low["usage"].opportunity["targets_per_snap"]
            == pytest.approx(high["usage"].opportunity["targets_per_snap"], rel=0.02)), \
        "only playing time changed, so the per-snap opportunity rate must not have moved"


# ── opportunity ──────────────────────────────────────────────────────────────────────────

def test_a_bigger_recent_target_share_sample_raises_expected_targets_and_receiving_volume():
    """Two receivers on the same snap share and the same yards-per-target, differing only in
    how many targets their own recent games show. Direction check on the opportunity half of
    the opportunity x efficiency split."""
    specs = [_wr("Decoy Wideout", 0.82, 3.0), _wr("Volume Wideout", 0.82, 10.0)]
    game = F.game(home="CIN", away="CLE")
    _install(specs, [game])

    low = _project("Decoy Wideout", "WR", game)
    high = _project("Volume Wideout", "WR", game)

    assert low["usage"].opportunity["targets_per_snap"] < high["usage"].opportunity["targets_per_snap"]
    assert low["expected_targets"] < high["expected_targets"]
    assert low["sim"]["receptions"].mean() < high["sim"]["receptions"].mean()
    assert low["sim"]["rec_yards"].mean() < high["sim"]["rec_yards"].mean()
    assert low["expected_snaps"] == pytest.approx(high["expected_snaps"], rel=0.02), \
        "only the target rate changed, so playing time must not have moved"


# ── game environment ─────────────────────────────────────────────────────────────────────

def test_a_higher_game_total_raises_team_plays_the_pass_rate_and_a_receivers_volume():
    """environment.py's own fitted coefficients: plays_per_total = +0.0983 and
    dropback_per_total = +0.002373, so a bigger game means more plays AND more passing. Both
    must reach the receiver's projected volume, not stop at the GameEnvironment dataclass."""
    specs = [_wr("Alpha Receiver", 0.85, 8.0)]
    low_total = F.game(home="CIN", away="CLE", wk=1, total=37.5, spread=-2.5)
    high_total = F.game(home="CIN", away="CLE", wk=2, total=55.5, spread=-2.5)
    _install(specs, [low_total, high_total])

    e_low = ENV.game_environment(low_total, "CIN")
    e_high = ENV.game_environment(high_total, "CIN")
    assert e_low.expected_scrimmage_plays < e_high.expected_scrimmage_plays
    assert e_low.expected_dropback_share < e_high.expected_dropback_share

    low = _project("Alpha Receiver", "WR", low_total)
    high = _project("Alpha Receiver", "WR", high_total)
    assert low["expected_targets"] < high["expected_targets"]
    assert low["sim"]["rec_yards"].mean() < high["sim"]["rec_yards"].mean()


def test_being_favoured_raises_team_plays_and_carries_while_lowering_the_pass_rate():
    """The other half of the same fit, and the sign that is easy to get backwards:
    plays_per_spread = +0.17825 but dropback_per_spread = -0.002641, so a favourite runs more
    plays and throws LESS of them. nflverse's spread_line is from the HOME team's view, so a
    positive spread on a CIN home game makes CIN the favourite."""
    specs = [dict(player="Bell Cow", position="RB", offense_pct=0.65, depth_rank=1,
                  carries=15.0, rush_yards=64.0, targets=4.0, receptions=3.0, rec_yards=26.0)]
    favoured = F.game(home="CIN", away="CLE", wk=1, spread=9.5, total=46.5)
    underdog = F.game(home="CIN", away="CLE", wk=2, spread=-9.5, total=46.5)
    _install(specs, [favoured, underdog])

    e_fav = ENV.game_environment(favoured, "CIN")
    e_dog = ENV.game_environment(underdog, "CIN")
    assert e_fav.expected_scrimmage_plays > e_dog.expected_scrimmage_plays
    assert e_fav.expected_dropback_share < e_dog.expected_dropback_share

    fav = _project("Bell Cow", "RB", favoured)
    dog = _project("Bell Cow", "RB", underdog)
    assert fav["expected_carries"] > dog["expected_carries"]
    assert fav["expected_targets"] < dog["expected_targets"]


def test_an_unpriced_game_lands_between_the_two_priced_extremes_rather_than_at_one_of_them():
    """A game the schedules release has not priced falls back to the measured league baseline,
    which must sit between a big-total and a small-total game — not default to either end."""
    specs = [_wr("Alpha Receiver", 0.85, 8.0)]
    low_total = F.game(home="CIN", away="CLE", wk=1, total=37.5, spread=0.0)
    high_total = F.game(home="CIN", away="CLE", wk=2, total=55.5, spread=0.0)
    _install(specs, [low_total, high_total])

    baseline = ENV.game_environment(None, "CIN")
    assert baseline.basis == "league_baseline"
    for e in (ENV.game_environment(low_total, "CIN"), ENV.game_environment(high_total, "CIN")):
        assert e.basis == "market_priced"
    assert (ENV.game_environment(low_total, "CIN").expected_scrimmage_plays
            < baseline.expected_scrimmage_plays
            < ENV.game_environment(high_total, "CIN").expected_scrimmage_plays)

    low = _project("Alpha Receiver", "WR", low_total)
    P.clear_cache()
    unpriced = _project("Alpha Receiver", "WR", None)
    P.clear_cache()
    high = _project("Alpha Receiver", "WR", high_total)
    assert low["expected_targets"] < unpriced["expected_targets"] < high["expected_targets"]


# ── matchup ──────────────────────────────────────────────────────────────────────────────

def _defense_weeks(team, yards_per_att, yards_per_carry, n=40):
    """Player-weeks attributed to `team`'s defense — fit_defense reads them off the OPPONENT
    field of the offensive player who produced them. 40 games x 35 attempts clears
    config.defense.min_plays, below which no adjustment is applied at all."""
    return [PlayerWeek(season=F.SEASON, week=w % 17 + 1, season_type="REG",
                       game_id=f"{team}_def_{w}", player_id=f"def_{team}_{w}",
                       player=f"Opponent Passer {team} {w}", position="QB",
                       team="BUF", opponent=team,
                       pass_attempts=35.0, pass_yards=35.0 * yards_per_att, completions=22.0,
                       carries=25.0, rush_yards=25.0 * yards_per_carry, sacks=2.0)
            for w in range(n)]


def test_a_measurably_weaker_pass_defense_raises_receiving_yards_but_not_targets():
    """matchup.py scales EFFICIENCY, never opportunity — a soft defense means more yards per
    target, not more targets. Both defenses are fit from real per-play rates here, not handed
    a multiplier directly, so this covers fit_defense -> matchup_multipliers -> simulate."""
    specs = [_wr("Alpha Receiver", 0.85, 8.0)]
    soft_game = F.game(home="CIN", away="CLE", wk=1, spread=-2.5, total=45.5)
    tough_game = F.game(home="CIN", away="PIT", wk=2, spread=-2.5, total=45.5)
    league = cfg("defense", "league")
    _install(specs, [soft_game, tough_game],
             extra_weeks=(_defense_weeks("CLE", league["yards_per_pass_att"] * 1.16,
                                         league["yards_per_carry"] * 1.16)
                          + _defense_weeks("PIT", league["yards_per_pass_att"] * 0.84,
                                           league["yards_per_carry"] * 0.84)))

    soft = _project("Alpha Receiver", "WR", soft_game)
    P.clear_cache()
    tough = _project("Alpha Receiver", "WR", tough_game)

    assert soft["matchup"]["yards_per_target"] > 1.0 > tough["matchup"]["yards_per_target"]
    assert soft["sim"]["rec_yards"].mean() > tough["sim"]["rec_yards"].mean()
    assert soft["expected_targets"] == pytest.approx(tough["expected_targets"], rel=0.03), \
        "a matchup must move efficiency, never opportunity"


def test_an_unmeasurable_opponent_leaves_the_projection_completely_neutral():
    """Week 1, or any opponent with less than config.defense.min_plays of history: no
    multiplier at all rather than one fit on a handful of plays."""
    specs = [_wr("Alpha Receiver", 0.85, 8.0)]
    game = F.game(home="CIN", away="CLE", wk=1)
    _install(specs, [game], extra_weeks=_defense_weeks("CLE", 9.0, 6.0, n=2))
    assert _project("Alpha Receiver", "WR", game)["matchup"] == {}


# ── extreme-value guard ──────────────────────────────────────────────────────────────────

# (position, opportunity market, yards market, the efficiency rate that relates them, the
#  config.defense.clamp key bounding how far a matchup may move that rate)
_EFFICIENCY_MARKETS = (
    ("QB", "pass_attempts", "pass_yards", "yards_per_attempt", "yards_per_pass_att"),
    ("RB", "rush_attempts", "rush_yards", "yards_per_carry", "yards_per_carry"),
    ("WR", "targets", "rec_yards", "yards_per_target", "yards_per_pass_att"),
    ("TE", "targets", "rec_yards", "yards_per_target", "yards_per_pass_att"),
)

# How far each slate player's RAW per-attempt efficiency is planted from its positional prior.
# The band the guard asserts is derived from these and from the engine's own measured matchup
# clamp — no threshold in this test is chosen.
_SLATE_SCALES = (0.7, 1.0, 1.4)


def _slate_specs():
    specs = []
    for pos, _, _, rate_key, _ in _EFFICIENCY_MARKETS:
        prior = position_cfg("positional_priors", pos)[rate_key]
        for scale in _SLATE_SCALES:
            name = f"{pos} Slate {scale}"
            if pos == "QB":
                specs.append(dict(player=name, position=pos, offense_pct=0.99, depth_rank=1,
                                  pass_attempts=34.0, completions=22.0,
                                  pass_yards=round(34.0 * prior * scale, 2),
                                  carries=4.0, rush_yards=18.0))
            elif pos == "RB":
                specs.append(dict(player=name, position=pos, offense_pct=0.60, depth_rank=1,
                                  carries=15.0, rush_yards=round(15.0 * prior * scale, 2),
                                  targets=4.0, receptions=3.0, rec_yards=26.0))
            else:
                tg = 8.0 if pos == "WR" else 5.0
                specs.append(dict(player=name, position=pos, offense_pct=0.80, depth_rank=1,
                                  targets=tg, receptions=round(0.65 * tg, 2),
                                  rec_yards=round(tg * prior * scale, 2)))
    return specs


def test_no_slate_player_projects_an_implied_efficiency_outside_the_engines_own_baseline():
    """The guard against "1 target, 90 receiving yards" and "30 carries, 2 yards" coming out of
    ordinary inputs.

    The band is DERIVED, not chosen. Shrinkage is (observed*n + prior*k)/(n+k), so a fitted
    rate always lies between the positional prior and the player's own raw rate; this slate
    plants raw rates at 0.7x/1.0x/1.4x of each position's measured prior, so every fitted rate
    is inside [0.7x, 1.4x]. The only thing that may then move it is the matchup multiplier,
    which matchup.py clamps to the observed per-team-season range in config.defense.clamp. The
    simulated distribution's implied efficiency (mean yards / mean opportunity) must therefore
    land inside [0.7 * clamp_lo, 1.4 * clamp_hi] x prior.

    This is a check on the CENTRE of the distribution, deliberately. A single trial of 1 target
    for 90 yards is a real football outcome and the Gamma tail is supposed to produce it; a
    player whose MEAN implies 90 yards a target is a broken projection.
    """
    specs = _slate_specs()
    game = F.game(home="CIN", away="CLE")
    _install(specs, [game], filler=False)
    data = P.league_data()
    assert data["priors"] == {}, \
        "the slate must stay too thin to redefine a league prior, or the band's reference moves"

    checked = 0
    for pos, opp_market, yards_market, rate_key, clamp_key in _EFFICIENCY_MARKETS:
        prior = position_cfg("positional_priors", pos)[rate_key]
        clamp_lo, clamp_hi = cfg("defense", "clamp", clamp_key)
        lo = min(_SLATE_SCALES) * clamp_lo * prior
        hi = max(_SLATE_SCALES) * clamp_hi * prior
        for scale in _SLATE_SCALES:
            name = f"{pos} Slate {scale}"
            proj = _project(name, pos, game, n=30000)
            fitted = proj["usage"].efficiency[rate_key]
            assert min(_SLATE_SCALES) * prior <= fitted <= max(_SLATE_SCALES) * prior, \
                (name, rate_key, fitted, prior)

            opportunity = proj["sim"][opp_market].mean()
            yards = proj["sim"][yards_market].mean()
            assert opportunity > 0.5, f"{name} projects essentially no {opp_market}"
            implied = yards / opportunity
            assert lo <= implied <= hi, (name, rate_key, implied, lo, hi)
            checked += 1
    assert checked == len(_EFFICIENCY_MARKETS) * len(_SLATE_SCALES)


def test_the_efficiency_band_is_tight_enough_to_reject_the_failures_it_exists_to_catch():
    """The band above is only worth asserting if it actually excludes the shapes the product
    owner named. 90 receiving yards on 1 target implies 90 yards per target; 2 rushing yards on
    30 carries implies 0.067 a carry. Both must sit far outside the derived band."""
    wr_prior = position_cfg("positional_priors", "WR")["yards_per_target"]
    wr_hi = max(_SLATE_SCALES) * cfg("defense", "clamp", "yards_per_pass_att")[1] * wr_prior
    assert 90.0 / 1.0 > wr_hi * 4

    rb_prior = position_cfg("positional_priors", "RB")["yards_per_carry"]
    rb_lo = min(_SLATE_SCALES) * cfg("defense", "clamp", "yards_per_carry")[0] * rb_prior
    assert 2.0 / 30.0 < rb_lo / 10


def test_a_player_with_no_history_at_all_still_projects_a_sane_positional_baseline():
    """The widest, least-informed input the engine accepts: no stat history, no depth chart, no
    roster row for this player's own team. Every rate is the positional prior, so the implied
    efficiency must be the prior itself — not zero, not a runaway."""
    specs = [_wr("Alpha Receiver", 0.85, 8.0)]
    game = F.game(home="CIN", away="CLE")
    _install(specs, [game], filler=False)

    priors = U.prior_for("WR")
    fit = U.prior_only_usage("WR", priors)
    assert fit.sample_weight == 0.0 and fit.n_games == 0
    from nfl.model import playing_time as PT
    from nfl.sim import engine as E
    env = ENV.game_environment(game, "CIN")
    pt = PT.project_playing_time([], "WR", None, env.expected_team_snaps)
    sim = E.simulate(fit, pt, env, n=30000, rng=np.random.default_rng(0))
    implied = sim["rec_yards"].mean() / sim["targets"].mean()
    assert implied == pytest.approx(priors["yards_per_target"], rel=0.05)
    assert 0 < sim["targets"].mean() < env.expected_scrimmage_plays
