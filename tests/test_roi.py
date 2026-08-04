"""
Tests for db.roi_summary — the realized-ROI figure behind the dashboard's "Model ROI" tile.
Isolated with its own temp DB (pattern from tests/test_dashboard.py).
"""
from __future__ import annotations

import pytest

import db


@pytest.fixture(autouse=True)
def _temp_db(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "roi_test.db", raising=False)
    db.init_db()


def _seed(rows):
    with db._lock, db._conn() as c:
        for r in rows:
            c.execute("""
                INSERT INTO prop_clv (line_id, game_date, sport, player, stat_type,
                    close_line, close_proj, actual, odds_type,
                    close_over_price, close_under_price, model_raw_prob)
                VALUES (:line_id, :game_date, :sport, :player, :stat_type,
                    :close_line, :close_proj, :actual, :odds_type,
                    :close_over_price, :close_under_price, :model_raw_prob)
            """, r)
        c.commit()


def _row(id_, close_line, close_proj, actual, over_price="-110", under_price="-110",
        model_raw_prob=None, odds_type="standard", days_ago=1):
    import datetime
    gd = (datetime.date.today() - datetime.timedelta(days=days_ago)).isoformat()
    return {"line_id": id_, "game_date": gd, "sport": "MLB", "player": f"P{id_}",
            "stat_type": "Hits", "close_line": close_line, "close_proj": close_proj,
            "actual": actual, "odds_type": odds_type, "close_over_price": over_price,
            "close_under_price": under_price, "model_raw_prob": model_raw_prob}


def test_roi_summary_no_rows_returns_null_roi():
    r = db.roi_summary(days=7)
    assert r["priced_plays"] == 0
    assert r["roi_pct"] is None


def test_roi_summary_counts_a_winning_over_play():
    # model projects well above the line (a "play", edge>=0.5), picks over (mean>line, no
    # raw prob given), actual clears it → win at -110 (decimal ~1.909).
    _seed([_row("a", close_line=1.5, close_proj=2.5, actual=3.0)])
    r = db.roi_summary(days=7, edge=0.5)
    assert r["priced_plays"] == 1
    assert r["profit"] == pytest.approx(1.0 * (1.0 + 100 / 110 - 1.0), abs=0.005)
    assert r["roi_pct"] > 0


def test_roi_summary_counts_a_losing_play():
    _seed([_row("a", close_line=1.5, close_proj=2.5, actual=1.0)])  # picked over, missed
    r = db.roi_summary(days=7, edge=0.5)
    assert r["priced_plays"] == 1
    assert r["profit"] == -1.0
    assert r["roi_pct"] == -100.0


def test_roi_summary_excludes_below_edge_threshold():
    # |proj - line| = 0.2 < default edge 0.5 → not a "play", excluded entirely.
    _seed([_row("a", close_line=1.5, close_proj=1.7, actual=5.0)])
    r = db.roi_summary(days=7, edge=0.5)
    assert r["plays"] == 0
    assert r["priced_plays"] == 0


def test_roi_summary_excludes_unpriced_rows_from_staking_but_not_fabricated():
    # A real play (edge met) but no recorded price → counted as a "play" for volume, but
    # NOT staked (never invent a price).
    _seed([_row("a", close_line=1.5, close_proj=2.5, actual=3.0,
                over_price=None, under_price=None)])
    r = db.roi_summary(days=7, edge=0.5)
    assert r["plays"] == 1
    assert r["priced_plays"] == 0
    assert r["roi_pct"] is None


def test_roi_summary_excludes_demon_goblin():
    _seed([_row("a", close_line=1.5, close_proj=2.5, actual=3.0, odds_type="demon")])
    r = db.roi_summary(days=7, edge=0.5)
    assert r["priced_plays"] == 0


def test_roi_summary_excludes_rows_outside_the_day_window():
    _seed([_row("a", close_line=1.5, close_proj=2.5, actual=3.0, days_ago=30)])
    r = db.roi_summary(days=7, edge=0.5)
    assert r["priced_plays"] == 0


def test_roi_summary_uses_model_raw_prob_for_the_pick_on_skewed_stats():
    # Mean sits above the line (would naively read as "over"), but the real pick (from
    # model_raw_prob) is under — and the actual comes in under, so this should be a WIN,
    # not a loss (mirrors db.scorecard's skewed-stat handling).
    _seed([_row("a", close_line=0.5, close_proj=0.6, actual=0.0, model_raw_prob=0.3)])
    r = db.roi_summary(days=7, edge=0.05)
    assert r["priced_plays"] == 1
    assert r["profit"] > 0
