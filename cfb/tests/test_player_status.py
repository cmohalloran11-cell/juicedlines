"""
Tests for cfb.player_status — the manual admin-editable status override + the staleness rule
that flags a projection near kickoff when status is unknown or stale.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import cfb
from store.database import SQLiteDatabase
from cfb import player_status
from cfb.repositories import PlayerRepository


@pytest.fixture()
def db(tmp_path):
    d = SQLiteDatabase(tmp_path / "cfb_status_test.db")
    cfb.init_schema(d)
    return d


@pytest.fixture()
def player_id(db):
    return PlayerRepository(db).create(full_name="Will Howard", position="QB",
                                       team="Ohio State")["id"]


def test_get_status_none_when_never_set(db, player_id):
    assert player_status.get_status(db, player_id) is None


def test_set_status_then_get_roundtrips(db, player_id):
    row = player_status.set_status(db, player_id, "questionable", note="ankle",
                                   set_by="admin@example.com")
    assert row["status"] == "questionable"
    fetched = player_status.get_status(db, player_id)
    assert fetched["status"] == "questionable" and fetched["note"] == "ankle"


def test_set_status_is_upsert_not_append(db, player_id):
    player_status.set_status(db, player_id, "out")
    player_status.set_status(db, player_id, "probable")
    row = player_status.get_status(db, player_id)
    assert row["status"] == "probable"


def test_is_stale_true_when_never_set():
    assert player_status.is_stale(None, kickoff_iso="2026-08-30T19:30:00+00:00") is True


def test_is_stale_false_when_no_kickoff_known(db, player_id):
    row = player_status.set_status(db, player_id, "available")
    assert player_status.is_stale(row, kickoff_iso=None) is False


def test_is_stale_false_when_confirmed_recently_near_kickoff():
    kickoff = datetime.now(timezone.utc) + timedelta(hours=2)
    row = {"as_of": (kickoff - timedelta(hours=10)).isoformat(timespec="seconds")}
    assert player_status.is_stale(row, kickoff_iso=kickoff.isoformat(timespec="seconds")) is False


def test_is_stale_true_when_confirmation_too_old_and_near_kickoff():
    kickoff = datetime.now(timezone.utc) + timedelta(hours=2)
    row = {"as_of": (kickoff - timedelta(hours=72)).isoformat(timespec="seconds")}
    assert player_status.is_stale(row, kickoff_iso=kickoff.isoformat(timespec="seconds")) is True


def test_is_stale_false_when_far_from_kickoff_even_if_old():
    kickoff = datetime.now(timezone.utc) + timedelta(days=5)
    row = {"as_of": (kickoff - timedelta(hours=200)).isoformat(timespec="seconds")}
    assert player_status.is_stale(row, kickoff_iso=kickoff.isoformat(timespec="seconds")) is False
