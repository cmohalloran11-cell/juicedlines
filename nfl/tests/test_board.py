"""End-to-end tests for the entry points the integration layer calls — attach_nfl(lines) and
analytics.analyze(line) — plus the characterization tests that missing data degrades
gracefully instead of crashing or fabricating. All offline, driven by a synthetic source."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import valuation
from nfl import analytics as A
from nfl import board as B
from nfl import projections as P
from nfl.data import espn as ESPN
from nfl.data import set_source
from nfl.data.base import DepthChartEntry, RosterWeek
from nfl.tests import fakes as F


def _depth(player="Star Receiver", team="CIN", position="WR", rank=1):
    return DepthChartEntry(season=F.SEASON, team=team, position=position, rank=rank,
                           player=player)


def _roster(player="Star Receiver", team="CIN", position="WR", status="ACT", wk=1):
    return RosterWeek(season=F.SEASON, week=wk, team=team, player=player, position=position,
                      status=status)


def _install(weeks=None, snaps=None, schedule=None, depth=None, rosters=None):
    P.clear_cache()
    set_source(F.FakeSource(weeks, snaps, schedule, depth, rosters))


@pytest.fixture(autouse=True)
def _isolated_source():
    # Empty-but-not-None ESPN override: every test gets zero real HTTP calls by default (see
    # nfl.data.espn.set_test_override) without having to opt in individually. A test that
    # wants to exercise the ESPN enrichment itself calls set_test_override again with real
    # fixture data.
    ESPN.set_test_override({}, {})
    yield
    P.clear_cache()
    set_source(None)
    ESPN.set_test_override(None, None)


def _rich_board(season_type="regular"):
    weeks, snaps = F.full_season("Star Receiver", "WR", "CIN", n=10, offense_pct=0.86,
                                 targets=9.0, receptions=6.0, rec_yards=84.0)
    fw, fs = F.league_filler(n_players=40, weeks_per=6)
    schedule = [F.game(home="CIN", away="CLE", gameday="2026-09-13")]
    _install(weeks + fw, snaps + fs, schedule if season_type == "regular" else [],
             [_depth()], [_roster()])
    return weeks, snaps


# ── attach_nfl ───────────────────────────────────────────────────────────────────────────

def test_attach_nfl_writes_the_shared_valuation_fields_and_the_nfl_contract():
    _rich_board()
    lines = [F.line(stat="Receiving Yards", value=72.5)]
    assert B.attach_nfl(lines) == 1
    l = lines[0]
    for f in ("model_proj", "model_prob", "model_edge", "model_floor", "model_ceiling",
              "model_median", "model_n", "proj_kind"):
        assert l.get(f) is not None, f
    assert l["proj_kind"] == "nfl_regular"
    assert l["proj_kind"] in valuation._FULL_ENGINE_KINDS
    assert 0.0 <= l["model_prob"] <= 1.0
    assert l["model_floor"] <= l["model_median"] <= l["model_ceiling"]
    assert l["model_edge"] == pytest.approx(l["model_proj"] - l["line"], abs=0.06)
    for f in ("season_type", "expected_snaps", "snap_range", "expected_targets",
              "expected_carries", "playing_time_confidence", "role_confidence",
              "game_total", "team_total", "spread", "role", "depth_chart_position"):
        assert f in l, f
    assert l["season_type"] == "regular" and l["season_type_confirmed"] is True
    assert l["role"] == "starter" and l["depth_chart_position"] == 1
    assert l["game_total"] == 45.5 and l["spread"] == -3.5
    assert l["nfl_confidence"] is not None and l["nfl_confidence_factors"]
    # the shared, sport-agnostic engine must work off these fields unmodified
    assert 0 <= valuation.confidence_score(l) <= 100
    assert 0 <= valuation.juice_score(l) <= 100


def test_attach_nfl_resolves_opponent_and_is_home_from_the_real_matched_schedule_game():
    _rich_board()   # CIN (team on the line) hosts CLE, per _rich_board's game()
    lines = [F.line(stat="Receiving Yards", value=72.5, team="CIN")]
    B.attach_nfl(lines)
    assert lines[0]["opponent"] == "CLE"
    assert lines[0]["is_home"] is True


def test_attach_nfl_opponent_is_unknown_for_preseason_by_construction():
    """Preseason games are never in the matched schedule (see resolve_season_type's
    docstring) -- opponent must be honestly None, not guessed."""
    _rich_board(season_type="preseason")
    lines = [F.line(stat="Receiving Yards", value=72.5, team="CIN")]
    B.attach_nfl(lines)
    assert lines[0]["season_type"] == "preseason"
    assert lines[0]["opponent"] is None
    assert lines[0]["is_home"] is None


def test_attach_nfl_uses_espn_team_logo_and_overrides_the_books_headshot():
    """ESPN's real photo wins over whatever the book's own image_url was -- found live
    2026-08: PrizePicks/Underdog's image_url for NFL is sometimes a team crest instead of a
    real face."""
    _rich_board()
    ESPN.set_test_override(
        {"cin": {"id": "4", "abbr": "CIN", "logo": "https://espn.example/cin.png",
                 "name": "Cincinnati Bengals"}},
        {"star receiver": "https://espn.example/headshots/star-receiver.png"},
    )
    lines = [F.line(stat="Receiving Yards", value=72.5, team="CIN",
                    headshot="https://static.prizepicks.com/images/teams/nfl/team-crest.webp")]
    B.attach_nfl(lines)
    assert lines[0]["team_logo"] == "https://espn.example/cin.png"
    assert lines[0]["headshot"] == "https://espn.example/headshots/star-receiver.png"


def test_attach_nfl_keeps_the_books_headshot_when_espn_has_no_match():
    """A player ESPN's active roster doesn't (yet) carry must keep the book's own image
    rather than being blanked -- an unmatched name is not evidence the image is wrong."""
    _rich_board()
    ESPN.set_test_override({}, {})   # no ESPN match for anyone
    original = "https://static.prizepicks.com/images/players/star-receiver.webp"
    lines = [F.line(stat="Receiving Yards", value=72.5, team="CIN", headshot=original)]
    B.attach_nfl(lines)
    assert lines[0]["headshot"] == original
    assert "team_logo" not in lines[0] or lines[0].get("team_logo") is None


def test_attach_nfl_ignores_other_sports_and_unmodelled_markets():
    _rich_board()
    lines = [
        F.line(stat="Receiving Yards", value=70.5),
        F.line(stat="Receiving TDs", value=0.5),
        F.line(stat="1H Receiving Yards", value=35.5),
        {"sport": "MLB", "player": "Someone", "stat_type": "Hits", "line": 0.5},
    ]
    assert B.attach_nfl(lines) == 1
    assert lines[0].get("model_proj") is not None
    assert all(l.get("model_proj") is None for l in lines[1:])


def test_attach_nfl_returns_zero_with_no_nfl_lines():
    _install()
    assert B.attach_nfl([{"sport": "MLB", "player": "X", "stat_type": "Hits", "line": 1.5}]) == 0


def test_preseason_lines_are_tagged_separately_and_trusted_less():
    """A game absent from the schedule release is a preseason game — see
    board.resolve_season_type. It must get its own proj_kind and a capped market trust."""
    weeks, snaps = F.full_season("Star Receiver", "WR", "CIN", n=10, offense_pct=0.86,
                                 targets=9.0, receptions=6.0, rec_yards=84.0)
    fw, fs = F.league_filler(40, weeks_per=6)
    _install(weeks + fw, snaps + fs, [F.game(gameday="2026-09-13")], [_depth()], [_roster()])

    # BOTH in the same call: a board carrying a preseason and a regular-season line for the
    # same player+market must not collapse them into one group and project both with
    # whichever season type the first line happened to resolve to.
    board = [F.line(stat="Receiving Yards", value=72.5, start="2026-09-13T17:00:00Z"),
             F.line(stat="Receiving Yards", value=72.5, start="2026-08-16T17:00:00Z")]
    board[1]["id"] = "pp_preseason"
    assert B.attach_nfl(board) == 2
    reg, pre = [board[0]], [board[1]]
    assert reg[0]["proj_kind"] == "nfl_regular"
    assert pre[0]["proj_kind"] == "nfl_preseason"
    assert pre[0]["season_type"] == "preseason" and pre[0]["season_type_confirmed"] is False
    assert pre[0]["trust_weight"] <= 0.35 < reg[0]["trust_weight"]
    assert pre[0]["preseason_risk"] is not None and pre[0]["rotation_tier"]
    assert pre[0]["playing_time_confidence"] == "low"
    assert pre[0]["game_total"] is None and pre[0]["spread"] is None
    # "trusted less" means the projection is pulled a larger FRACTION of the way from the
    # model's own raw mean back to the posted line — the raw means themselves differ, so
    # comparing raw edges would compare two different questions.
    def pull(l):
        return abs(l["model_proj"] - l["line"]) / abs(l["model_raw"] - l["line"])
    assert pull(pre[0]) < pull(reg[0])


# ── book_season_type (PrizePicks' NFLP league code) — confirmed live 2026-08 ────────────

def test_resolve_season_type_trusts_book_preseason_hint_over_a_real_schedule_match():
    """PrizePicks' own NFLP classification (pullers._pp_nfl_season_type) must win even when
    our OWN schedule lookup would otherwise resolve the game as regular season -- the book's
    real classification is more trustworthy than our absence-based inference, not less."""
    game = F.game(home="CIN", away="CLE", gameday="2026-09-13")
    idx = B._schedule_index([game])
    line = F.line(stat="Receiving Yards", value=72.5, team="CIN",
                  start="2026-09-13T17:00:00Z", book_season_type="preseason")
    season_type, matched_game, confirmed = B.resolve_season_type(line, idx)
    assert season_type == "preseason"
    assert matched_game is None
    assert confirmed is True


def test_resolve_season_type_trusts_book_regular_hint_even_with_no_schedule_match():
    """The book saying "NFL" (regular) must be trusted even if our schedule lookup fails to
    match (e.g. a team-alias/date edge case) -- better a real book classification with no
    environment enrichment than a wrongly-inferred preseason label."""
    idx = B._schedule_index([])   # no games at all -- lookup will find nothing
    line = F.line(stat="Receiving Yards", value=72.5, team="CIN",
                  start="2026-09-13T17:00:00Z", book_season_type="regular")
    season_type, matched_game, confirmed = B.resolve_season_type(line, idx)
    assert season_type == "regular"
    assert matched_game is None
    assert confirmed is True


def test_resolve_season_type_falls_back_to_inference_with_no_book_hint():
    """Unchanged behavior for lines with no book_season_type (Underdog, or any PrizePicks
    league string the detector doesn't recognize) -- still infers from schedule presence."""
    game = F.game(home="CIN", away="CLE", gameday="2026-09-13")
    idx = B._schedule_index([game])
    matched = F.line(stat="Receiving Yards", value=72.5, team="CIN", start="2026-09-13T17:00:00Z")
    assert B.resolve_season_type(matched, idx) == ("regular", game, True)
    unmatched = F.line(stat="Receiving Yards", value=72.5, team="CIN", start="2026-08-16T17:00:00Z")
    assert B.resolve_season_type(unmatched, idx) == ("preseason", None, False)


def test_a_thin_sample_defers_to_the_market_line_instead_of_shipping_a_noisy_edge():
    weeks, snaps = F.full_season("Rookie WR", "WR", "CIN", n=1, offense_pct=0.30,
                                 targets=14.0, receptions=11.0, rec_yards=180.0)
    fw, fs = F.league_filler(40, weeks_per=6)
    _install(weeks + fw, snaps + fs, [F.game(gameday="2026-09-13")], [], [])
    lines = [F.line("Rookie WR", "Receiving Yards", 55.5)]
    assert B.attach_nfl(lines) == 1
    assert lines[0]["trust_weight"] < 0.5
    assert abs(lines[0]["model_edge"]) < 20.0, "one huge game must not become a huge edge"


# ── graceful degradation ─────────────────────────────────────────────────────────────────

def test_no_snap_counts_at_all_widens_the_band_and_lowers_playing_time_confidence():
    weeks, snaps = F.full_season("Star Receiver", "WR", "CIN", n=10, offense_pct=0.86,
                                 targets=9.0, receptions=6.0, rec_yards=84.0)
    fw, fs = F.league_filler(40, weeks_per=6)
    sched = [F.game(gameday="2026-09-13")]

    _install(weeks + fw, snaps + fs, sched, [_depth()], [_roster()])
    with_snaps = [F.line(stat="Receiving Yards", value=72.5)]
    B.attach_nfl(with_snaps)

    _install(weeks + fw, fs, sched, [_depth()], [_roster()])   # this player's snaps removed
    without = [F.line(stat="Receiving Yards", value=72.5)]
    assert B.attach_nfl(without) == 1
    a, b = with_snaps[0], without[0]
    assert b["playing_time_confidence"] == "low" and a["playing_time_confidence"] == "high"
    # RELATIVE spread, not absolute: with no snap rows the per-snap rates collapse toward the
    # (lower) positional prior, so the absolute band narrows simply because the projection
    # itself is smaller. The honest comparison is the distribution's coefficient of variation
    # around the model's own raw mean.
    assert (b["model_std_dev"] / b["model_raw"]) > (a["model_std_dev"] / a["model_raw"])
    assert b["expected_snaps"] is not None    # the depth chart still supports an estimate
    assert b["snap_range"][1] - b["snap_range"][0] > a["snap_range"][1] - a["snap_range"][0]


def test_no_depth_chart_and_no_snaps_yields_role_unknown_not_a_fabricated_role():
    fw, fs = F.league_filler(40, weeks_per=6)
    weeks = [F.week("Ghost Player", "WR", "CIN", wk=w) for w in range(1, 4)]
    _install(weeks + fw, fs, [F.game(gameday="2026-09-13")], [], [])
    lines = [F.line("Ghost Player", "Receiving Yards", 40.5)]
    assert B.attach_nfl(lines) == 1
    assert lines[0]["role"] == "unknown"
    assert lines[0]["depth_chart_position"] is None
    assert lines[0]["playing_time_confidence"] == "low"


def test_no_schedule_row_leaves_the_game_environment_fields_null_not_defaulted():
    weeks, snaps = F.full_season("Star Receiver", "WR", "CIN", n=8, targets=8.0,
                                 receptions=5.0, rec_yards=70.0)
    fw, fs = F.league_filler(40, weeks_per=6)
    _install(weeks + fw, snaps + fs, [], [_depth()], [_roster()])
    lines = [F.line(stat="Receiving Yards", value=68.5)]
    assert B.attach_nfl(lines) == 1
    l = lines[0]
    assert l["game_total"] is None and l["team_total"] is None and l["spread"] is None
    assert l["weather"] is None
    assert l["model_proj"] is not None, "a missing schedule must not stop the projection"


def test_an_entirely_empty_data_source_projects_nothing_rather_than_inventing_a_player():
    _install()
    lines = [F.line()]
    assert B.attach_nfl(lines) == 0
    assert lines[0].get("model_proj") is None


def test_unmeasurable_fields_are_reported_as_none_never_as_a_plausible_number():
    _rich_board()
    lines = [F.line(stat="Receiving Yards", value=72.5)]
    B.attach_nfl(lines)
    l = lines[0]
    assert l["red_zone_opportunities"] is None, "no red-zone source is pulled"
    assert l["expected_routes_basis"] == "pass_play_snaps_upper_bound", \
        "routes run are not published anywhere free — the field must say what it really is"
    assert l["expected_routes"] is not None and l["expected_routes"] <= l["expected_snaps"]


# ── analyze ──────────────────────────────────────────────────────────────────────────────

def test_analyze_returns_real_drivers_built_from_the_model_fields():
    _rich_board()
    lines = [F.line(stat="Receiving Yards", value=72.5)]
    B.attach_nfl(lines)
    out = A.analyze(lines[0])
    assert out["available"] is True
    assert out["sport"] == "NFL" and out["player_type"] == "WR"
    assert out["season_type"] == "regular" and out["role"] == "starter"
    assert out["hit_rate"]["n"] == 10 and out["hit_rate"]["over_pct"] == 100
    assert len(out["recent"]) == 10 and out["recent"][0]["cells"]["TGT"] == 9
    assert out["view_cols"] == ["SNP", "TGT", "REC", "RECYD", "CAR", "RYD"]
    drivers = " ".join(out["drivers"])
    assert "snaps" in drivers and "targets per snap" in drivers
    assert "45.5 total" in drivers, "the real market total must appear in the reasoning"
    assert out["red_zone_opportunities"] is None and out["red_zone_note"]
    assert out["expected_routes_note"]


def test_analyze_preseason_explains_the_rotation_tier_and_its_limits():
    weeks, snaps = F.full_season("Star Receiver", "WR", "CIN", n=10, offense_pct=0.86,
                                 targets=9.0, receptions=6.0, rec_yards=84.0)
    qw, qs = F.full_season("CIN Passer", "QB", "CIN", n=10, offense_pct=0.98,
                           pass_attempts=34.0, completions=22.0, pass_yards=245.0,
                           sacks=2.0, carries=4.0, rush_yards=18.0)
    fw, fs = F.league_filler(40, weeks_per=6)
    _install(weeks + qw + fw, snaps + qs + fs, [], [_depth()], [_roster()])
    lines = [F.line(stat="Receiving Yards", value=45.5, start="2026-08-16T17:00:00Z")]
    B.attach_nfl(lines)
    out = A.analyze(lines[0])
    assert out["season_type"] == "preseason"
    assert out["rotation_tier"] == "confirmed_starter"
    drivers = " ".join(out["drivers"])
    assert "rotation tier" in drivers and "confirmed starter" in drivers
    assert "no data source publishes preseason snap counts" in drivers
    t = out["team_tendency"]
    assert t["preseason_pass_rate"] is not None
    assert t["starter_snap_rate"] is None and t["avg_first_team_drives"] is None
    assert "preseason" in t["unavailable"]["starter_snap_rate"]


def test_analyze_says_so_when_the_player_cannot_be_resolved():
    _install()
    out = A.analyze(F.line("Nobody At All"))
    assert out["available"] is False and "nothing to project" in out["reason"]


# ── confidence ───────────────────────────────────────────────────────────────────────────

def test_nfl_confidence_uses_the_right_weight_set_and_drops_missing_factors():
    from nfl.confidence import nfl_confidence
    _rich_board()
    lines = [F.line(stat="Receiving Yards", value=72.5)]
    B.attach_nfl(lines)
    proj = P.project_player("Star Receiver", team="CIN", season_type="regular",
                            game=F.game(gameday="2026-09-13"))
    reg = nfl_confidence(proj, lines[0])
    assert reg["season_type"] == "regular"
    assert {f["factor"] for f in reg["factors"]} >= {"Playing Time", "Usage Stability",
                                                     "Matchup", "Game Environment"}
    # historical accuracy has no graded NFL rows yet -> dropped, not scored at a default
    hist = next(f for f in reg["factors"] if f["factor"] == "Historical Model Accuracy")
    assert hist["value"] is None and "no graded NFL history" in hist["detail"]
    assert 0 <= reg["score"] <= 100 and reg["level"] in ("high", "medium", "low")

    pre = nfl_confidence(dict(proj, season_type="preseason"), lines[0])
    assert {f["factor"] for f in pre["factors"]} >= {"Rotation Certainty", "Role Stability"}
    assert "Usage Stability" not in {f["factor"] for f in pre["factors"]}


def test_nfl_confidence_is_none_when_nothing_real_can_be_scored():
    from nfl.confidence import nfl_confidence
    empty = {"season_type": "regular", "playing_time": None, "usage": None,
             "environment": None, "opponent_defense": None, "matchup": {}}
    out = nfl_confidence(empty, {})
    assert out["score"] is None and out["level"] is None
    assert "not enough real signal" in out["note"]
