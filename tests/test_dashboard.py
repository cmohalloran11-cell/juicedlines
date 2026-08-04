"""
Tests for dashboard.py's demon/goblin handling.

Demon/goblin lines are structurally decisive (the boosted payout comes from setting the line
at an extreme threshold), so their juice_score is inflated by design, not earned. These tests
guard against them silently swamping the priced-play surfaces (Daily Juice Drops, Juice Leader,
Best Value, and the default Projections feed) — they must stay a separate, opt-in lane.
"""
from __future__ import annotations

import os
import tempfile

import pytest

import db as _db
import dashboard


@pytest.fixture(autouse=True)
def _temp_db(monkeypatch, tmp_path):
    # dashboard.build() reads db.recent_prop_moves(), which queries the prop_clv table
    # directly -- this test file never called db.init_db(), so it only ever "passed" because
    # SOME prior test in the same session (or a leftover history.db from a manual local run)
    # happened to create the schema first. On a genuinely clean checkout with no pre-existing
    # history.db (exactly what CI's runners have), prop_clv doesn't exist yet and this raised
    # sqlite3.OperationalError: no such table: prop_clv -- confirmed by reproducing a truly
    # fresh venv + fresh checkout locally. Isolate this file with its own temp DB, matching
    # the pattern already used in tests/test_diagnostics.py and tests/test_auth_and_user_api.py.
    monkeypatch.setattr(_db, "DB_PATH", tmp_path / "history_test.db", raising=False)
    _db.init_db()


def _line(id_, player, odds_type="standard", model_prob=0.6, model_edge=0.5):
    return {
        "id": id_, "player": player, "team": "NYY", "sport": "MLB", "stat_type": "Hits",
        "source": "prizepicks", "line": 1.5, "model_proj": 2.0, "model_prob": model_prob,
        "model_edge": model_edge, "model_n": 20, "proj_kind": "engine",
        "odds_type": odds_type, "lineup_status": None,
        # real PrizePicks standard legs always carry a flat pick'em price (595/595 in
        # production); demon/goblin never do (valuation.is_unpriced gates those separately).
        "pickem_price": None if odds_type in ("demon", "goblin") else -137,
    }


def _slate():
    # One modest, honestly-priced standard prop and a pile of demon props whose extreme
    # model_prob (near 1.0) gives them a much higher raw juice_score than the standard prop —
    # the scenario that would crowd standard out of every top-N surface if not guarded against.
    demons = [_line(f"d{i}", f"Demon Guy {i}", odds_type="demon", model_prob=0.98, model_edge=3.0)
              for i in range(20)]
    return [_line("s1", "Standard Guy", odds_type="standard", model_prob=0.6, model_edge=0.5)] + demons


def test_projected_includes_demon_with_real_projection():
    pool = dashboard._projected(_slate())
    assert len(pool) == 21   # all projected — demon/goblin are no longer filtered out


def test_build_picks_surfaces_exclude_unpriced_demon_goblin():
    d = dashboard.build(_slate(), updated_at="2026-01-01T00:00:00Z")
    # drops (Daily Juice Drops) must be the standard prop only — no demon, despite their
    # far higher raw juice_score
    assert [row["player"] for row in d["drops"]] == ["Standard Guy"]
    # juice_leader / top_edge / best_value tiles must all be the standard prop, not a demon
    for tile_key in ("juice_leader", "top_edge", "best_value"):
        tile = d["tiles"][tile_key]
        assert tile is not None and tile["player"] == "Standard Guy"


def test_drop_marks_demon_goblin_unpriced_with_no_edge_pct():
    row = dashboard._drop(_line("d1", "Demon Guy", odds_type="demon", model_prob=0.98))
    assert row["unpriced"] is True
    assert row["edgePct"] is None
    assert row["oddsType"] == "demon"
    # projection/probability still real and present
    assert row["projection"] is not None and row["probability"] is not None


def test_drop_side_is_over_for_demon_even_when_model_favors_under():
    # PrizePicks doesn't offer Under on a boosted leg — the displayed side must never be
    # "under" for demon/goblin, regardless of what the raw model probability says.
    row = dashboard._drop(_line("d1", "Demon Guy", odds_type="demon", model_prob=0.05))
    assert row["side"] == "over"


def test_projected_excludes_props_with_no_priced_available_side():
    # Market-availability audit: a standard/boosted prop where the book offers neither side
    # priced (no pickem_price, no over/under_implied — e.g. thin Underdog/Sleeper data) isn't
    # a bet a user can actually place, so it must be excluded from the board entirely rather
    # than shown with a fabricated recommendation.
    unpriced_standard = {
        "id": "u1", "player": "Nobody", "team": "NYY", "sport": "MLB", "stat_type": "Hits",
        "source": "underdog", "line": 1.5, "model_proj": 2.0, "model_prob": 0.7,
        "model_edge": 0.5, "model_n": 20, "proj_kind": "engine",
        "odds_type": "standard", "lineup_status": None, "pickem_price": None,
    }
    assert dashboard._projected([unpriced_standard]) == []
    # same prop, but with a real price on one side → included, recommending only that side
    priced = {**unpriced_standard, "over_implied": 0.5}
    out = dashboard._projected([priced])
    assert len(out) == 1
    assert dashboard._drop(out[0])["side"] == "over"


def test_new_hero_tiles_scoped_correctly():
    # best_demon/best_goblin must come from the unpriced lane, best_standard from the
    # standard-only subset, highest_confidence/safest_play from the full priced pool —
    # never leak a demon/goblin into the priced-only tiles (same guard as the existing tests).
    lines = _slate()
    d = dashboard.build(lines, updated_at="2026-01-01T00:00:00Z")
    assert d["tiles"]["best_demon"]["player"] == "Demon Guy 0"
    assert d["tiles"]["best_standard"]["player"] == "Standard Guy"
    assert d["tiles"]["highest_confidence"]["player"] == "Standard Guy"  # only priced prop here
    assert d["tiles"]["best_goblin"] is None  # no goblin props in this slate


def test_coach_context_has_only_real_computed_numbers():
    lines = _slate()
    d = dashboard.build(lines, updated_at="2026-01-01T00:00:00Z")
    ctx = d["coach_context"]
    assert ctx["demon_count"] == 20
    assert ctx["total_priced_props_today"] == 1


def test_projections_default_excludes_demon_goblin_opt_in_includes():
    lines = _slate()
    default_rows = dashboard.projections(lines, limit=50)
    assert all(r["oddsType"] == "standard" for r in default_rows)
    assert len(default_rows) == 1

    boosted_rows = dashboard.projections(lines, limit=50, odds_types=("demon", "goblin"))
    assert len(boosted_rows) == 20
    assert all(r["oddsType"] == "demon" for r in boosted_rows)
    assert all(r["unpriced"] for r in boosted_rows)
