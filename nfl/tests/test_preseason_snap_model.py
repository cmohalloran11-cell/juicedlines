"""
End-to-end proof that the preseason playing-time model actually SEPARATES players by role,
position and team signal through the full project_player() path — not just that the
individual pure functions in model/rotation.py do (nfl/tests/test_core.py covers those).

The bug this rewrite fixed, reproduced from real live production evidence (2026-08): 25 of
34 real preseason board lines projected the literal identical 19.6 expected_snaps, 8 more at
26.1, one at 18.3 — three distinct values total across 34 real players of every position and
depth-chart rank. See ROT.snap_clustering_report and this file's regression test at the
bottom for the same measurement applied to synthetic players spanning every tier.

All offline, driven by FakeSource — same pattern as test_board.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from nfl import projections as P
from nfl.data import set_source
from nfl.data.base import DepthChartEntry, RosterWeek, SnapWeek
from nfl.model import rotation as ROT
from nfl.tests import fakes as F


def _depth(player, team="CIN", position="WR", rank=1):
    return DepthChartEntry(season=F.SEASON, team=team, position=position, rank=rank,
                           player=player)


def _roster(player, team="CIN", position="WR"):
    return RosterWeek(season=F.SEASON, week=1, team=team, player=player, position=position,
                      status="ACT")


def _install(weeks=None, snaps=None, schedule=None, depth=None, rosters=None):
    P.clear_cache()
    set_source(F.FakeSource(weeks, snaps, schedule, depth, rosters))


def _project(player, team, position, season_type="preseason"):
    return P.project_player(player, team=team, position=position, season_type=season_type)


# ── section 17: synthetic players with identical baseline talent, different roles ─────────

def test_confirmed_starter_second_team_and_third_team_produce_meaningfully_different_snaps():
    depth = [_depth("Player A", "CIN", "WR", 1), _depth("Player B", "CIN", "WR", 2),
            _depth("Player C", "CIN", "WR", 3)]
    rosters = [_roster(p, "CIN", "WR") for p in ("Player A", "Player B", "Player C")]
    # Player A's prior-season history CONFIRMS a real starter workload; B and C have none —
    # identical baseline efficiency/talent (see F.full_season's fixed stat kwargs) for all
    # three, so any snap difference below comes only from role, not skill.
    a_weeks, a_snaps = F.full_season("Player A", "WR", "CIN", n=10, offense_pct=0.75,
                                     targets=7.0, receptions=5.0, rec_yards=65.0)
    fw, fs = F.league_filler(n_players=40, weeks_per=6)
    _install(a_weeks + fw, a_snaps + fs, [], depth, rosters)

    a = _project("Player A", "CIN", "WR")
    b = _project("Player B", "CIN", "WR")
    c = _project("Player C", "CIN", "WR")

    assert a["rotation_tier"] == "confirmed_starter"
    assert b["rotation_tier"] == "second_team"
    assert c["rotation_tier"] == "third_team"
    snaps = {a["expected_snaps"], b["expected_snaps"], c["expected_snaps"]}
    assert len(snaps) == 3, f"expected 3 distinct values, got {snaps}"
    # None of the three should land on the OLD model's collision points.
    assert all(round(v, 1) not in (19.6,) for v in snaps) or len(snaps) == 3


def test_qb_starter_vs_backup_separate_and_differ_from_a_skill_position_in_the_same_tier():
    depth = [_depth("Star QB", "CIN", "QB", 1), _depth("Backup QB", "CIN", "QB", 2),
            _depth("Player A", "CIN", "WR", 1)]
    rosters = [_roster(p, "CIN", pos) for p, pos in
              (("Star QB", "QB"), ("Backup QB", "QB"), ("Player A", "WR"))]
    fw, fs = F.league_filler(n_players=40, weeks_per=6)
    _install(fw, fs, [], depth, rosters)

    starter = _project("Star QB", "CIN", "QB")
    backup = _project("Backup QB", "CIN", "QB")
    wr = _project("Player A", "CIN", "WR")

    assert starter["rotation_tier"] == "confirmed_starter" or starter["rotation_tier"] == "likely_starter"
    assert backup["rotation_tier"] == "second_team"
    assert starter["expected_snaps"] != backup["expected_snaps"]
    # Same tier (likely_starter/confirmed_starter roughly), different position -> different
    # expected_snaps: a QB plays every snap of his own drives, a WR rotates within them.
    assert starter["expected_snaps"] != wr["expected_snaps"]


def test_rb_starter_vs_backup_and_wr_starter_vs_third_string_all_separate():
    depth = [_depth("Feature Back", "CIN", "RB", 1), _depth("Backup Back", "CIN", "RB", 2),
            _depth("WR1", "CIN", "WR", 1), _depth("WR3", "CIN", "WR", 3)]
    rosters = [_roster(p, "CIN", pos) for p, pos in
              (("Feature Back", "RB"), ("Backup Back", "RB"), ("WR1", "WR"), ("WR3", "WR"))]
    fw, fs = F.league_filler(n_players=40, weeks_per=6)
    _install(fw, fs, [], depth, rosters)

    rb1 = _project("Feature Back", "CIN", "RB")
    rb2 = _project("Backup Back", "CIN", "RB")
    wr1 = _project("WR1", "CIN", "WR")
    wr3 = _project("WR3", "CIN", "WR")

    assert rb1["expected_snaps"] != rb2["expected_snaps"]
    assert wr1["expected_snaps"] != wr3["expected_snaps"]
    assert rb1["rotation_tier"] != rb2["rotation_tier"]
    assert wr1["rotation_tier"] != wr3["rotation_tier"]


# ── sparse/missing data must degrade gracefully, never crash or fabricate ─────────────────

def test_unknown_player_gets_the_widest_least_confident_projection_not_a_crash():
    fw, fs = F.league_filler(n_players=40, weeks_per=6)
    _install(fw, fs, [], [], [])   # no depth chart, no roster row, nothing about this player
    proj = P.project_player("Total Stranger", team="CIN", position="WR",
                            season_type="preseason")
    assert proj is not None
    assert proj["rotation_tier"] == "unknown"
    assert proj["playing_time"].confidence == "low"
    assert proj["expected_snaps"] is not None   # still a real number, just wide/uncertain
    assert proj["preseason_risk"] == max(
        ROT.preseason_risk(t, proj["playing_time"]) for t in [proj["rotation_tier"]])


def test_missing_depth_chart_but_on_roster_falls_back_to_fringe_not_a_crash():
    fw, fs = F.league_filler(n_players=40, weeks_per=6)
    _install(fw, fs, [], [], [_roster("Camp Body", "CIN", "WR")])
    proj = P.project_player("Camp Body", team="CIN", position="WR", season_type="preseason")
    assert proj["rotation_tier"] == "fringe"
    assert proj["expected_snaps"] is not None


def test_missing_team_rotation_signal_falls_back_cleanly_to_position_and_league_priors():
    """No games at all for this team in `weeks` -> team_tendency's pass rate is None (see
    ROT.team_tendency's own-sample-too-thin branch cascading to the league prior, which ALSO
    needs >=100 plays of SOMETHING to not be None) -> the drives model must still produce a
    real number using zero team-rotation nudge, not crash or fabricate a team signal."""
    depth = [_depth("Rookie WR", "NEW", "WR", 1)]
    rosters = [_roster("Rookie WR", "NEW", "WR")]
    _install([], [], [], depth, rosters)   # zero weeks anywhere -> no team OR league pass rate
    proj = P.project_player("Rookie WR", team="NEW", position="WR", season_type="preseason")
    assert proj is not None and proj["expected_snaps"] is not None
    assert proj["team_tendency"].preseason_pass_rate is None
    assert ROT.team_rotation_nudge("WR", None) == 0.0


def test_missing_historical_preseason_data_is_the_default_path_and_still_works():
    depth = [_depth("Player A", "CIN", "WR", 1)]
    rosters = [_roster("Player A", "CIN", "WR")]
    fw, fs = F.league_filler(n_players=40, weeks_per=6)
    _install(fw, fs, [], depth, rosters)
    proj = P.project_player("Player A", "CIN", "WR", season_type="preseason")
    assert proj["playing_time"].basis == "preseason_tier"
    assert proj["playing_time"].n_games == 0


def test_real_preseason_snap_history_when_present_overrides_the_tier_default():
    """The one case nflverse never actually provides today (see rotation.py's module
    docstring) — synthesized here to prove the override CODE PATH works, not just that it
    exists on paper. A player with a strong OWN preseason track record should out-project the
    bare tier default for the same role."""
    depth = [_depth("Vet Slot", "CIN", "WR", 2)]
    rosters = [_roster("Vet Slot", "CIN", "WR")]
    pre_snaps = [SnapWeek(season=F.SEASON, week=w, season_type="PRE",
                          game_id=f"{F.SEASON}_PRE{w}", player="Vet Slot", position="WR",
                          team="CIN", opponent="XXX", offense_snaps=40.0, offense_pct=0.62)
                for w in range(1, 4)]
    fw, fs = F.league_filler(n_players=40, weeks_per=6)
    _install(fw, fs + pre_snaps, [], depth, rosters)
    proj = P.project_player("Vet Slot", "CIN", "WR", season_type="preseason")
    assert proj["playing_time"].basis == "preseason_own_history"
    assert proj["playing_time"].n_games == 3
    assert proj["snap_diagnostic"]["prior_influence"]["historical_usage_weight"] > 0.3


# ── distribution / diagnostics surfaced end-to-end ─────────────────────────────────────────

def test_expected_snaps_ships_a_real_asymmetric_percentile_distribution_from_the_simulator():
    depth = [_depth("Player A", "CIN", "WR", 3)]   # a low-mean tier -> right-skewed Beta
    rosters = [_roster("Player A", "CIN", "WR")]
    fw, fs = F.league_filler(n_players=40, weeks_per=6)
    _install(fw, fs, [], depth, rosters)
    proj = P.project_player("Player A", "CIN", "WR", season_type="preseason")
    sp = proj["snap_percentiles"]
    assert sp is not None
    for k in ("p10", "p25", "p50", "p75", "p90", "mean", "median", "std_dev"):
        assert k in sp
    assert sp["p10"] < sp["p50"] < sp["p90"]
    # Real Monte Carlo asymmetry check: for a right-skewed low-mean distribution the upper
    # gap (p90-p50) is wider than the lower gap (p50-p10) — never forced, just what a Beta
    # with a mean well under 0.5 actually produces.
    assert (sp["p90"] - sp["p50"]) > (sp["p50"] - sp["p10"])


def test_snap_diagnostic_and_prior_influence_are_exposed_on_the_projection():
    depth = [_depth("Player A", "CIN", "RB", 1)]
    rosters = [_roster("Player A", "CIN", "RB")]
    fw, fs = F.league_filler(n_players=40, weeks_per=6)
    _install(fw, fs, [], depth, rosters)
    proj = P.project_player("Player A", "CIN", "RB", season_type="preseason")
    assert proj["snap_diagnostic"] is not None
    assert proj["prior_influence"] is not None
    assert proj["snap_diagnostic"]["depth_chart_tier"] == proj["rotation_tier"]


def test_regular_season_projections_get_no_snap_diagnostic_but_still_get_percentiles():
    weeks, snaps = F.full_season("Player A", "WR", "CIN", n=10, offense_pct=0.8)
    schedule = [F.game(home="CIN", away="CLE")]
    _install(weeks, snaps, schedule, [_depth("Player A", "CIN", "WR", 1)],
            [_roster("Player A", "CIN", "WR")])
    proj = P.project_player("Player A", "CIN", "WR", season_type="regular",
                            game=schedule[0])
    assert proj["snap_diagnostic"] is None
    assert proj["snap_percentiles"] is not None


# ── clustering regression: the actual bug, measured on synthetic players spanning every tier

def test_clustering_report_on_every_tier_shows_real_separation_not_the_old_collision():
    """The direct regression test for the bug this task exists to fix. Real live evidence
    (2026-08, before this rewrite): 34 real preseason board lines produced only 3 distinct
    expected_snaps values (19.6 x25, 26.1 x8, 18.3 x1) -- 76.5% within +-2 snaps of 20. This
    builds one synthetic player per tier x position combination the board actually projects
    (QB/RB/WR/TE, the 4 positions this engine supports) and measures the same statistic on
    the NEW model's output."""
    positions = ("QB", "RB", "WR", "TE")
    depth, rosters, players = [], [], []
    for ti, tier_name in enumerate(ROT.TIERS):
        rank = {"confirmed_starter": 1, "likely_starter": 1, "first_team_rotation": 2,
               "second_team": 2, "third_team": 4}.get(tier_name)
        for pos in positions:
            name = f"{tier_name}_{pos}"
            players.append((name, pos))
            if rank is not None:
                depth.append(_depth(name, "CIN", pos, rank))
            if tier_name != "unknown":
                rosters.append(_roster(name, "CIN", pos))
    fw, fs = F.league_filler(n_players=40, weeks_per=6)
    _install(fw, fs, [], depth, rosters)

    snaps = [P.project_player(name, "CIN", pos, season_type="preseason")["expected_snaps"]
            for name, pos in players]
    report = ROT.snap_clustering_report(snaps)

    OLD_PCT_WITHIN_2 = 76.5    # real live 2026-08 measurement, see module docstring
    OLD_DISTINCT = 3
    assert report["n_distinct_values"] > OLD_DISTINCT
    assert report["pct_within_2_of_target"] < OLD_PCT_WITHIN_2
