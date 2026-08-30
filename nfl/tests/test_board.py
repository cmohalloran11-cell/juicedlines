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


def _rich_board(schedule=None):
    weeks, snaps = F.full_season("Star Receiver", "WR", "CIN", n=10, offense_pct=0.86,
                                 targets=9.0, receptions=6.0, rec_yards=84.0)
    fw, fs = F.league_filler(n_players=40, weeks_per=6)
    if schedule is None:
        schedule = [F.game(home="CIN", away="CLE", gameday="2026-09-13")]
    _install(weeks + fw, snaps + fs, schedule, [_depth()], [_roster()])
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
    for f in ("expected_snaps", "snap_range", "expected_targets",
              "expected_carries", "playing_time_confidence", "role_confidence",
              "game_total", "team_total", "spread", "role", "depth_chart_position"):
        assert f in l, f
    assert l["role"] == "starter" and l["depth_chart_position"] == 1
    assert l["game_total"] == 45.5 and l["spread"] == -3.5
    assert l["nfl_confidence"] is not None and l["nfl_confidence_factors"]
    # the shared, sport-agnostic engine must work off these fields unmodified
    assert 0 <= valuation.confidence_score(l) <= 100
    # juice_score's range depends on valuation.JUICE_VERSION: v1 is unsigned 0-100, v2 is
    # signed [-100, +100] and None when the model has no opinion to score. Assert the contract
    # both versions share, so this stays a real check under either.
    js = valuation.juice_score(l)
    assert js is None or -100 <= js <= 100


def test_attach_nfl_resolves_opponent_and_is_home_from_the_real_matched_schedule_game():
    _rich_board()   # CIN (team on the line) hosts CLE, per _rich_board's game()
    lines = [F.line(stat="Receiving Yards", value=72.5, team="CIN")]
    B.attach_nfl(lines)
    assert lines[0]["opponent"] == "CLE"
    assert lines[0]["is_home"] is True


def test_attach_nfl_opponent_is_unknown_when_no_schedule_row_matches():
    """A line the schedules release carries no row for still projects, but its opponent must
    be honestly None rather than guessed from anything but the real schedule."""
    _rich_board(schedule=[])
    lines = [F.line(stat="Receiving Yards", value=72.5, team="CIN")]
    assert B.attach_nfl(lines) == 1
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


def test_espn_team_assets_maps_nflverses_la_alias_to_the_rams(monkeypatch):
    """nflverse (and this repo's own P.norm_team) normalizes the Rams' real abbreviation
    "LAR" down to the legacy code "LA" -- but ESPN itself has no team keyed "LA", only
    "LAR", so a lookup by the aliased code silently missed the Rams' crest. Found live
    2026-08: every NFL team's logo populated except the Rams. This bypasses
    set_test_override (which would skip the code under test) and instead fakes the ESPN
    HTTP layer directly."""
    import json as _json
    payload = _json.dumps({"sports": [{"leagues": [{"teams": [
        {"team": {"id": "14", "abbreviation": "LAR", "displayName": "Los Angeles Rams",
                  "shortDisplayName": "Rams", "logos": [{"href": "https://espn.example/lar.png"}]}},
        {"team": {"id": "4", "abbreviation": "CIN", "displayName": "Cincinnati Bengals",
                  "shortDisplayName": "Bengals", "logos": [{"href": "https://espn.example/cin.png"}]}},
    ]}]}]})
    ESPN.set_test_override(None, None)   # let the real team_assets() body run
    ESPN._cache.clear()
    monkeypatch.setattr(ESPN.cache, "fetch_text", lambda *a, **k: payload)
    assets = ESPN.team_assets()
    ESPN._cache.clear()
    assert assets["lar"]["logo"] == "https://espn.example/lar.png"
    assert assets["la"]["logo"] == "https://espn.example/lar.png"
    assert assets["la"] == assets["lar"]


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


def test_zero_trust_line_reads_as_a_coinflip_not_a_systematic_under():
    """The mean-vs-median anchoring bug, reproduced and fixed. Real live evidence (2026-08):
    94% of the live board recommended Under, even on lines where model_proj sat
    EXACTLY on the market line (zero visible edge). Root cause: the old code blended the
    model's MEAN toward the market anchor and shifted the whole (right-skewed, Gamma-shaped)
    simulated array by (blended_mean - raw_mean) to match — which recenters the array so its
    MEAN sits on the line while its MEDIAN (what model_prob is actually computed from) sits
    BELOW it, because a right-skewed distribution's median is always below its mean. At zero
    trust the market line IS the anchor, so this produced model_prob well under 50% on a
    line with no real disagreement at all. Fixed by blending/shifting on the MEDIAN — a
    market line is definitionally the book's implied 50/50 point, so a zero-trust line
    should land close to a coinflip."""
    fw, fs = F.league_filler(40, weeks_per=6)
    # A player with ZERO usable game history -> sample_weight 0.0 -> trust 0.0 < min_trust,
    # so model_proj is anchored fully to the market line (the exact zero-disagreement case
    # the live production bug was found on).
    _install(fw, fs, [], [_depth("Nobody Yet", rank=2)], [_roster("Nobody Yet")])
    lines = [F.line(player="Nobody Yet", stat="Receiving Yards", value=45.5)]
    assert B.attach_nfl(lines) == 1
    l = lines[0]
    assert l["trust_weight"] == 0.0
    assert 0.40 <= l["model_prob"] <= 0.60, (
        f"zero-trust line should read near a coinflip, got model_prob={l['model_prob']}")
    # The invariant the fix restores: median direction and probability direction must agree
    # (dataos.validate_direction's own rule) -- trivially true here since the line landed
    # dead center, but the real point is model_prob is no longer skew-biased toward Under.


def test_low_trust_edge_still_reflects_a_right_skewed_stats_true_mean_not_the_bare_line():
    """A near-zero-trust line still gets an informative "Proj" that differs from the bare
    market line — the fix changes HOW the recentering works, it doesn't collapse "Proj" back
    to just echoing the line. For a right-skewed stat (Gamma yards), the honest mean sits
    ABOVE a line that represents the median, so model_proj > line is the expected,
    correct-by-construction result at zero trust, not a bug."""
    fw, fs = F.league_filler(40, weeks_per=6)
    _install(fw, fs, [], [_depth("Nobody Yet", rank=2)], [_roster("Nobody Yet")])
    lines = [F.line(player="Nobody Yet", stat="Receiving Yards", value=45.5)]
    B.attach_nfl(lines)
    l = lines[0]
    assert l["trust_weight"] == 0.0
    assert l["model_proj"] > l["line"], (
        "a right-skewed stat's mean sits above the market's implied median")
    assert l["model_median"] == pytest.approx(l["line"], abs=1.0)


# ── schedule matching ────────────────────────────────────────────────────────────────────

def test_match_game_finds_the_scheduled_game_and_returns_none_when_there_is_no_row():
    game = F.game(home="CIN", away="CLE", gameday="2026-09-13")
    idx = B._schedule_index([game])
    matched = F.line(stat="Receiving Yards", value=72.5, team="CIN", start="2026-09-13T17:00:00Z")
    assert B.match_game(matched, idx) is game
    unmatched = F.line(stat="Receiving Yards", value=72.5, team="CIN", start="2026-08-16T17:00:00Z")
    assert B.match_game(unmatched, idx) is None
    assert B.match_game(F.line(team=None), idx) is None
    assert B.match_game(F.line(start=""), idx) is None


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
    assert out["role"] == "starter"
    assert out["hit_rate"]["n"] == 10 and out["hit_rate"]["over_pct"] == 100
    assert len(out["recent"]) == 10 and out["recent"][0]["cells"]["TGT"] == 9
    assert out["view_cols"] == ["SNP", "TGT", "REC", "RECYD", "CAR", "RYD"]
    drivers = " ".join(out["drivers"])
    assert "snaps" in drivers and "targets per snap" in drivers
    assert "45.5 total" in drivers, "the real market total must appear in the reasoning"
    assert out["red_zone_opportunities"] is None and out["red_zone_note"]
    assert out["expected_routes_note"]


def test_projection_ships_a_real_monte_carlo_snap_distribution():
    """`snap_percentiles` comes from the simulator's own snaps array, so it carries BOTH
    stages of playing-time uncertainty and is asymmetric wherever the underlying Beta is."""
    _rich_board(schedule=[])
    proj = P.project_player("Star Receiver", team="CIN", position="WR")
    sp = proj["snap_percentiles"]
    assert sp is not None
    for k in ("p10", "p25", "p50", "p75", "p90", "mean", "median", "std_dev"):
        assert k in sp, k
    assert sp["p10"] < sp["p50"] < sp["p90"]


def test_analyze_says_so_when_the_player_cannot_be_resolved():
    _install()
    out = A.analyze(F.line("Nobody At All"))
    assert out["available"] is False and "nothing to project" in out["reason"]


# ── confidence ───────────────────────────────────────────────────────────────────────────

def test_nfl_confidence_scores_every_real_factor_and_drops_missing_ones():
    from nfl.confidence import nfl_confidence
    _rich_board()
    lines = [F.line(stat="Receiving Yards", value=72.5)]
    B.attach_nfl(lines)
    proj = P.project_player("Star Receiver", team="CIN", game=F.game(gameday="2026-09-13"))
    reg = nfl_confidence(proj, lines[0])
    assert {f["factor"] for f in reg["factors"]} >= {"Playing Time", "Usage Stability",
                                                     "Matchup", "Game Environment"}
    # historical accuracy has no graded NFL rows yet -> dropped, not scored at a default
    hist = next(f for f in reg["factors"] if f["factor"] == "Historical Model Accuracy")
    assert hist["value"] is None and "no graded NFL history" in hist["detail"]
    assert 0 <= reg["score"] <= 100 and reg["level"] in ("high", "medium", "low")


def test_nfl_confidence_is_none_when_nothing_real_can_be_scored():
    from nfl.confidence import nfl_confidence
    empty = {"playing_time": None, "usage": None,
             "environment": None, "opponent_defense": None, "matchup": {}}
    out = nfl_confidence(empty, {})
    assert out["score"] is None and out["level"] is None
    assert "not enough real signal" in out["note"]
