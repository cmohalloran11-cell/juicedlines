"""
Tests for dashboard.py's demon/goblin handling.

Demon/goblin lines are structurally decisive (the boosted payout comes from setting the line
at an extreme threshold), so their juice_score is inflated by design, not earned. These tests
guard against them silently swamping the priced-play surfaces (Daily Juice Drops, Juice Leader,
Best Value, and the default Projections feed) — they must stay a separate, opt-in lane.
"""
from __future__ import annotations

import dashboard


def _line(id_, player, odds_type="standard", model_prob=0.6, model_edge=0.5):
    return {
        "id": id_, "player": player, "team": "NYY", "sport": "MLB", "stat_type": "Hits",
        "source": "prizepicks", "line": 1.5, "model_proj": 2.0, "model_prob": model_prob,
        "model_edge": model_edge, "model_n": 20, "proj_kind": "engine",
        "odds_type": odds_type, "lineup_status": None,
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


def test_projections_default_excludes_demon_goblin_opt_in_includes():
    lines = _slate()
    default_rows = dashboard.projections(lines, limit=50)
    assert all(r["oddsType"] == "standard" for r in default_rows)
    assert len(default_rows) == 1

    boosted_rows = dashboard.projections(lines, limit=50, odds_types=("demon", "goblin"))
    assert len(boosted_rows) == 20
    assert all(r["oddsType"] == "demon" for r in boosted_rows)
    assert all(r["unpriced"] for r in boosted_rows)
