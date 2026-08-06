"""
Tests for fantasy.player_matching — the fuzzy-match fallback + review-log requirement. Exact
mapping short-circuits fuzzy matching; high-confidence fuzzy matches auto-link; anything below
threshold is logged for human review, never silently mis-mapped.
"""
from __future__ import annotations

import pytest

import fantasy
from store.database import SQLiteDatabase
from fantasy.repositories import PlayerRepository, PlayerMappingRepository, UnmatchedPlayerRepository
from fantasy import player_matching


@pytest.fixture()
def db(tmp_path):
    d = SQLiteDatabase(tmp_path / "fantasy_test.db")
    fantasy.init_schema(d)
    return d


def test_normalize_name_strips_punctuation_and_suffixes():
    assert player_matching.normalize_name("Ja'Marr Chase") == "jamarr chase"
    assert player_matching.normalize_name("Odell Beckham Jr.") == "odell beckham"
    assert player_matching.normalize_name("Michael Pittman III") == "michael pittman"


def test_find_best_match_prefers_exact_normalized_name():
    candidates = [
        {"id": "p1", "full_name": "Justin Jefferson", "position": "WR", "team": "MIN"},
        {"id": "p2", "full_name": "Justin Fields", "position": "QB", "team": "PIT"},
    ]
    best, score = player_matching.find_best_match("Justin Jefferson", "WR", "MIN", candidates)
    assert best["id"] == "p1"
    assert score >= player_matching.MATCH_THRESHOLD


def test_find_best_match_position_tiebreak():
    # Two similarly-named players; only the position match should push the right one ahead.
    candidates = [
        {"id": "p1", "full_name": "Josh Allen", "position": "QB", "team": "BUF"},
        {"id": "p2", "full_name": "Josh Allen", "position": "LB", "team": "JAX"},
    ]
    best, _ = player_matching.find_best_match("Josh Allen", "QB", "BUF", candidates)
    assert best["id"] == "p1"


def test_resolve_or_log_uses_existing_exact_mapping(db):
    players = PlayerRepository(db)
    mapper = PlayerMappingRepository(db)
    player = players.create(full_name="CeeDee Lamb", position="WR", team="DAL")
    mapper.map(player["id"], "nflverse_gsis", "00-1111111")

    resolved = player_matching.resolve_or_log(
        db, "nflverse_gsis", "00-1111111", "CeeDee Lamb", position="WR", team="DAL")
    assert resolved == player["id"]


def test_resolve_or_log_auto_links_high_confidence_fuzzy_match(db):
    players = PlayerRepository(db)
    player = players.create(full_name="Amon-Ra St. Brown", position="WR", team="DET")

    # nflverse's display name spells it slightly differently -- close enough to auto-link.
    resolved = player_matching.resolve_or_log(
        db, "nflverse_gsis", "00-2222222", "Amon-Ra St Brown", position="WR", team="DET")
    assert resolved == player["id"]

    # second call for the same source_id hits the exact-mapping path now, not fuzzy again
    resolved_again = player_matching.resolve_or_log(
        db, "nflverse_gsis", "00-2222222", "Amon-Ra St Brown", position="WR", team="DET")
    assert resolved_again == player["id"]


def test_resolve_or_log_logs_low_confidence_instead_of_guessing(db):
    players = PlayerRepository(db)
    players.create(full_name="Marvin Harrison Jr.", position="WR", team="ARI")

    resolved = player_matching.resolve_or_log(
        db, "nflverse_gsis", "00-3333333", "Completely Different Name", position="WR", team="SEA")
    assert resolved is None

    pending = UnmatchedPlayerRepository(db).list_pending()
    assert len(pending) == 1
    assert pending[0]["source_id"] == "00-3333333"
    assert pending[0]["status"] == "pending"


def test_resolve_or_log_no_candidates_still_logs_not_raises(db):
    resolved = player_matching.resolve_or_log(
        db, "nflverse_gsis", "00-4444444", "Nobody In The Table", position="TE")
    assert resolved is None
    assert len(UnmatchedPlayerRepository(db).list_pending()) == 1
