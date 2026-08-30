"""
Tests for cfb.player_matching — same fuzzy-match + review-log requirement as
fantasy.player_matching (see tests/test_fantasy_player_matching.py), applied to cfb_players.
Exact mapping short-circuits fuzzy matching; high-confidence fuzzy matches auto-link;
anything below threshold is logged for human review, never silently mis-mapped.
"""
from __future__ import annotations

import pytest

import cfb
from store.database import SQLiteDatabase
from cfb.repositories import PlayerRepository, PlayerMappingRepository, UnmatchedPlayerRepository
from cfb import player_matching


@pytest.fixture()
def db(tmp_path):
    d = SQLiteDatabase(tmp_path / "cfb_test.db")
    cfb.init_schema(d)
    return d


def test_reuses_fantasy_normalize_and_match_functions():
    from fantasy.player_matching import normalize_name, find_best_match
    assert player_matching.normalize_name is normalize_name
    assert player_matching.find_best_match is find_best_match


def test_resolve_or_log_uses_existing_exact_mapping(db):
    players = PlayerRepository(db)
    mapper = PlayerMappingRepository(db)
    player = players.create(full_name="Will Howard", position="QB", team="Ohio State")
    mapper.map(player["id"], "cfbd", "101")

    resolved = player_matching.resolve_or_log(
        db, "cfbd", "101", "Will Howard", position="QB", team="Ohio State")
    assert resolved == player["id"]


def test_resolve_or_log_auto_links_high_confidence_fuzzy_match(db):
    players = PlayerRepository(db)
    player = players.create(full_name="TreVeyon Henderson", position="RB", team="Ohio State")

    # the-odds-api's player description spells it slightly differently -- close enough to
    # auto-link even with no team hint (Odds API player props carry no team).
    resolved = player_matching.resolve_or_log(
        db, "the_odds_api", "treveyon henderson", "TreVeyon Henderson")
    assert resolved == player["id"]

    resolved_again = player_matching.resolve_or_log(
        db, "the_odds_api", "treveyon henderson", "TreVeyon Henderson")
    assert resolved_again == player["id"]


def test_resolve_or_log_logs_low_confidence_instead_of_guessing(db):
    players = PlayerRepository(db)
    players.create(full_name="Julian Sayin", position="QB", team="Ohio State")

    resolved = player_matching.resolve_or_log(
        db, "the_odds_api", "completely different name", "Completely Different Name")
    assert resolved is None

    pending = UnmatchedPlayerRepository(db).list_pending()
    assert len(pending) == 1
    assert pending[0]["source"] == "the_odds_api"
    assert pending[0]["status"] == "pending"


def test_resolve_or_log_no_candidates_still_logs_not_raises(db):
    resolved = player_matching.resolve_or_log(
        db, "cfbd", "999", "Nobody In The Table", position="TE")
    assert resolved is None
    assert len(UnmatchedPlayerRepository(db).list_pending()) == 1


def test_unmatched_resolve_links_the_source_id(db):
    players = PlayerRepository(db)
    mapper = PlayerMappingRepository(db)
    unmatched = UnmatchedPlayerRepository(db)
    player = players.create(full_name="Jeremiah Smith", position="WR", team="Ohio State")

    row = unmatched.log(source="the_odds_api", source_id="j smith", raw_name="J. Smith")
    ok = unmatched.resolve(row["id"], player["id"])
    assert ok is True
    assert mapper.resolve("the_odds_api", "j smith") == player["id"]
    assert unmatched.list_pending() == []
