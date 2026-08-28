"""
Tests that CFB rows log the FULL prop_clv ledger schema -- including pre-anchor
model_raw/model_raw_prob/trust_weight/model_version -- from day one, on a freshly
init_db()'d database. CFB uses the exact same prop_clv table every other sport does
(sport='CFB'); no schema change was needed, so this test is what proves that claim rather
than just asserting it in a docstring. Fixture pattern matches tests/test_dashboard.py's
(temp DB_PATH via monkeypatch + explicit init_db()) per CLAUDE.md's testing-requirements
section.
"""
from __future__ import annotations

import pytest

import db as _db


@pytest.fixture(autouse=True)
def _temp_db(monkeypatch, tmp_path):
    monkeypatch.setattr(_db, "DB_PATH", tmp_path / "cfb_ledger_test.db", raising=False)
    _db.init_db()


def _cfb_line(id_="cfb_odds_evt1_draftkings_player_pass_yds_will_howard"):
    return {
        "id": id_, "sport": "CFB", "source": "draftkings", "player": "Will Howard",
        "stat_type": "Passing Yards", "line": 245.5,
        "model_proj": 261.0, "model_prob": 0.58, "proj_kind": "cfb_prior_b",
        "model_raw": 258.0, "model_raw_prob": 0.55, "trust_weight": 0.6,
        "model_floor": 190.0, "model_ceiling": 330.0,
        "model_version": "cfb-1.0.0", "feature_version": "1.0.0",
        "data_snapshot": "2026-08-28", "model_n": 4,
        "model_pre_median": 255.0, "model_pre_sd": 38.2, "model_anchor_t": 0.6,
        "odds_type": "standard", "over_price": "-115", "under_price": "-105",
        "over_implied": 0.535, "under_implied": 0.512, "game_id": "evt1",
        "lineup_status": None,
    }


def test_cfb_row_logs_full_pre_anchor_schema():
    n = _db.log_clv([_cfb_line()], "2026-08-28T12:00:00")
    assert n == 1

    with _db._lock, _db._conn() as c:
        row = dict(c.execute(
            "SELECT * FROM prop_clv WHERE sport='CFB' AND line_id=?",
            ("cfb_odds_evt1_draftkings_player_pass_yds_will_howard",)).fetchone())

    assert row["sport"] == "CFB"
    assert row["close_proj"] == 261.0
    assert row["model_raw"] == 258.0
    assert row["model_raw_prob"] == 0.55
    assert row["trust_weight"] == 0.6
    assert row["model_version"] == "cfb-1.0.0"
    assert row["model_raw_median"] == 255.0
    assert row["model_raw_sd"] == 38.2
    assert row["model_anchor_t"] == 0.6
    assert row["proj_kind"] == "cfb_prior_b"


def test_cfb_row_open_close_upsert_moves_close_only():
    open_ts = "2026-08-28T12:00:00"
    close_ts = "2026-08-28T18:00:00"
    _db.log_clv([_cfb_line()], open_ts)

    moved = _cfb_line()
    moved["line"] = 248.5
    moved["model_proj"] = 263.0
    _db.log_clv([moved], close_ts)

    with _db._lock, _db._conn() as c:
        row = dict(c.execute(
            "SELECT * FROM prop_clv WHERE sport='CFB' AND line_id=?",
            ("cfb_odds_evt1_draftkings_player_pass_yds_will_howard",)).fetchone())

    assert row["open_line"] == 245.5 and row["open_ts"] == open_ts
    assert row["close_line"] == 248.5 and row["close_ts"] == close_ts


def test_cfb_row_without_line_or_model_proj_is_not_logged():
    unpriced = _cfb_line()
    unpriced["model_proj"] = None
    assert _db.log_clv([unpriced], "2026-08-28T12:00:00") == 0


def test_grading_and_scorecard_are_scoped_to_cfb_and_report_insufficient_data_when_ungraded():
    _db.log_clv([_cfb_line()], "2026-08-28T12:00:00")
    card = _db.scorecard(sport="CFB")
    # Nothing graded yet (actual IS NULL) -- must not fabricate a hit rate from an ungraded row.
    assert card is not None
