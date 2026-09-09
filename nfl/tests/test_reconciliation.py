"""
Team-level reconciliation tests — MECHANISM checks, not accuracy claims.

Every number asserted here is a code-correctness result: a team-game is built out of synthetic
players (nfl/tests/fakes.py) whose planted per-game production the fit recovers exactly, and
the check is asserted to flag the boards that genuinely cannot add up and to leave the ones
that can alone. None of this says anything about how often a real NFL board violates the
constraint — that is what running reconcile_board against the live board reports, and it is
deliberately reported rather than corrected (see nfl/model/reconciliation.py).
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nfl import board as B
from nfl import projections as P
from nfl.config import cfg
from nfl.data import espn as ESPN
from nfl.data import set_source
from nfl.model import environment as ENV
from nfl.model import reconciliation as R
from nfl.tests import fakes as F


@pytest.fixture(autouse=True)
def _isolated_source():
    ESPN.set_test_override({}, {})
    yield
    P.clear_cache()
    set_source(None)
    ESPN.set_test_override(None, None)


def _project_team(specs, game=None, team="CIN"):
    """Run a whole synthetic offence through the REAL pipeline and return its projections."""
    game = game if game is not None else F.game(home=team, away="CLE")
    weeks, snaps, depth, rosters = F.offense(specs, team=team)
    fw, fs = F.league_filler(n_players=40, weeks_per=6)
    P.clear_cache()
    set_source(F.FakeSource(weeks + fw, snaps + fs, [game], depth, rosters))
    out = []
    for spec in specs:
        proj = P.project_player(spec["player"], team=team, position=spec["position"], game=game,
                                n=8000)
        assert proj is not None, spec["player"]
        out.append(proj)
    return out


# ── the team side of the comparison ──────────────────────────────────────────────────────

def test_the_team_attempt_pool_is_derived_from_the_environments_own_plays_and_pass_rate():
    """Attempts, not dropbacks: dropback_share is a share of SCRIMMAGE plays, which counts
    sacks, and neither a target nor a pass attempt does. The conversion uses the measured
    league sack rate already in config, and the three pools must add back to the plays the
    environment started from."""
    env = ENV.game_environment(F.game(spread=0.0, total=44.0), "CIN")
    pool = R.team_attempt_pool(env)
    sack_rate = cfg("defense", "league", "sack_rate")
    assert pool["dropbacks"] == pytest.approx(env.expected_scrimmage_plays
                                              * env.expected_dropback_share)
    assert pool["pass_attempts"] == pytest.approx(pool["dropbacks"] * (1 - sack_rate))
    assert pool["pass_attempts"] < pool["dropbacks"]
    sacks = pool["dropbacks"] - pool["pass_attempts"]
    assert (pool["pass_attempts"] + sacks + pool["rush_attempts"]
            == pytest.approx(env.expected_scrimmage_plays))


def test_the_tolerance_is_the_engines_own_measured_team_volume_spread_not_a_chosen_number():
    plays_cv = cfg("sim", "snaps_sd_frac")
    dropback_sd = cfg("sim", "dropback_sd")
    league_share = cfg("environment", "league_dropback_share")
    assert R.pool_relative_sd("pass_attempts") == pytest.approx(
        math.sqrt(plays_cv ** 2 + (dropback_sd / league_share) ** 2))
    assert R.pool_relative_sd("rush_attempts") == pytest.approx(
        math.sqrt(plays_cv ** 2 + (dropback_sd / (1 - league_share)) ** 2))
    # the rush pool is the noisier of the two, because the same absolute pass-rate spread is a
    # bigger fraction of the smaller share
    assert R.pool_relative_sd("rush_attempts") > R.pool_relative_sd("pass_attempts")


# ── the healthy case must not false-positive ─────────────────────────────────────────────

def test_a_realistic_full_offence_produces_no_violation():
    """The whole point of the tolerance: a QB, a two-man backfield, three receivers and a tight
    end at ordinary per-game production must add up, or the check would fire on every board and
    mean nothing."""
    projections = _project_team(F.STARTER_OFFENSE)
    env = projections[0]["environment"]
    pool = R.team_attempt_pool(env)
    targets = sum(p["expected_targets"] for p in projections)
    carries = sum(p["expected_carries"] for p in projections)
    assert R.reconcile_team(projections, env) == []
    assert targets < pool["pass_attempts"], (targets, pool["pass_attempts"])
    assert carries < pool["rush_attempts"], (carries, pool["rush_attempts"])


def test_a_partial_board_never_flags_however_far_it_undershoots():
    """A board only carries the players a book posted a prop for, so summed projections sitting
    well under the team total is the normal state and carries no information. The check is
    one-sided by design."""
    specs = [s for s in F.STARTER_OFFENSE if s["position"] == "WR"][:1]
    projections = _project_team(specs)
    env = projections[0]["environment"]
    assert sum(p["expected_targets"] for p in projections) < 0.4 * R.team_attempt_pool(env)["pass_attempts"]
    assert R.reconcile_team(projections, env) == []


# ── the failures it exists to catch ──────────────────────────────────────────────────────

_HUNGRY_RECEIVERS = tuple(
    dict(player=f"Target Hog {i}", position="WR", offense_pct=0.88, depth_rank=1,
         targets=11.0, receptions=7.0, rec_yards=95.0)
    for i in range(6))


def test_six_alpha_receivers_on_one_team_are_flagged_against_the_teams_pass_attempts():
    """The integrity failure the module exists for: every one of these six projections is
    individually defensible (each player really did average 11 targets on an 88% snap share),
    and nothing per-player is wrong — they only become impossible together."""
    projections = _project_team(_HUNGRY_RECEIVERS)
    env = projections[0]["environment"]
    violations = R.reconcile_team(projections, env)
    by_q = {v.quantity: v for v in violations}
    assert "targets" in by_q, [v.message() for v in violations]
    v = by_q["targets"]
    assert v.projected_total > v.tolerance_high > v.team_expected
    assert v.overshoot > 0 and v.n_players == 6
    assert v.contributors[0][0].startswith("Target Hog")
    assert v.team == "CIN" and v.game_id == env.game_id
    assert "targets" in v.message() and "CIN" in v.message()
    # the receivers barely rush, so the carry pool is untouched — the check is per-pool
    assert "carries" not in by_q


def test_a_backfield_of_three_bell_cows_is_flagged_against_the_teams_rush_attempts():
    specs = tuple(dict(player=f"Bell Cow {i}", position="RB", offense_pct=0.70, depth_rank=1,
                       carries=17.0, rush_yards=74.0) for i in range(3))
    projs = _project_team(specs)
    violations = {v.quantity: v for v in R.reconcile_team(projs, projs[0]["environment"])}
    assert "carries" in violations, list(violations)
    v = violations["carries"]
    assert v.projected_total > v.tolerance_high
    assert v.contributors[0][1] == "RB"


def test_two_starting_quarterbacks_are_flagged_against_the_same_pass_attempt_pool():
    """Targets and pass attempts are drawn from one pool, so double-counting a starting QB is
    the same integrity failure as double-counting a receiver's targets."""
    specs = tuple(dict(player=f"Gun Slinger {i}", position="QB", offense_pct=0.99, depth_rank=1,
                       pass_attempts=36.0, completions=23.0, pass_yards=258.0)
                  for i in range(2))
    projs = _project_team(specs)
    violations = {v.quantity: v for v in R.reconcile_team(projs, projs[0]["environment"])}
    assert "pass_attempts" in violations, list(violations)
    assert violations["pass_attempts"].projected_total > violations["pass_attempts"].tolerance_high


# ── grouping ─────────────────────────────────────────────────────────────────────────────

def test_reconcile_board_groups_by_team_game_and_counts_a_duplicated_player_once():
    projections = _project_team(_HUNGRY_RECEIVERS)
    assert [v.quantity for v in R.reconcile_board(projections)] == ["targets"]
    # the same six projections again is a duplicate resolution, not twelve receivers
    doubled = R.reconcile_board(projections + list(projections))
    assert len(doubled) == 1
    assert doubled[0].projected_total == pytest.approx(
        R.reconcile_board(projections)[0].projected_total)
    assert doubled[0].n_players == 6


def test_two_teams_are_reconciled_against_their_own_environments_not_pooled():
    healthy = _project_team(F.STARTER_OFFENSE)
    hungry = _project_team(_HUNGRY_RECEIVERS, team="BUF",
                           game=F.game(home="BUF", away="MIA"))
    violations = R.reconcile_board(healthy + hungry)
    assert {v.team for v in violations} == {"BUF"}


def test_a_projection_with_no_team_or_no_environment_is_skipped_rather_than_crashing():
    assert R.reconcile_board([]) == []
    assert R.reconcile_board([{"player": "X", "team": None}]) == []
    assert R.reconcile_board([{"player": "X", "team": "CIN", "environment": None}]) == []


# ── wiring ───────────────────────────────────────────────────────────────────────────────

def test_attach_nfl_runs_the_check_and_reports_a_violating_board(capsys):
    """The check has to run on a REAL board build, not only when a test calls it directly —
    attach_nfl is the only point at which a whole team-game's projections exist together."""
    game = F.game(home="CIN", away="CLE", gameday="2026-09-13")
    weeks, snaps, depth, rosters = F.offense(_HUNGRY_RECEIVERS)
    fw, fs = F.league_filler(n_players=40, weeks_per=6)
    P.clear_cache()
    set_source(F.FakeSource(weeks + fw, snaps + fs, [game], depth, rosters))

    lines = [F.line(player=s["player"], stat="Targets", value=7.5) for s in _HUNGRY_RECEIVERS]
    assert B.attach_nfl(lines) == len(lines)
    out = capsys.readouterr().out
    assert "[nfl.board] team volume violation: CIN" in out
    assert "targets" in out
    # and every line still carries its projection — the check reports, it never rewrites
    assert all(l.get("model_proj") is not None for l in lines)


def test_attach_nfl_stays_quiet_on_a_healthy_board(capsys):
    game = F.game(home="CIN", away="CLE", gameday="2026-09-13")
    weeks, snaps, depth, rosters = F.offense()
    fw, fs = F.league_filler(n_players=40, weeks_per=6)
    P.clear_cache()
    set_source(F.FakeSource(weeks + fw, snaps + fs, [game], depth, rosters))

    lines = ([F.line(player=s["player"], stat="Targets", value=5.5)
              for s in F.STARTER_OFFENSE if s["position"] in ("WR", "TE", "RB")]
             + [F.line(player="Bell Cow", stat="Rush Attempts", value=14.5),
                F.line(player="Gun Slinger", stat="Pass Attempts", value=33.5)])
    assert B.attach_nfl(lines) == len(lines)
    assert "team volume violation" not in capsys.readouterr().out
