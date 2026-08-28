"""
Tests for cfb.schema / cfb.repositories on a fresh temp DB — CLAUDE.md's testing-requirements
rule: every DB-touching test isolates itself with a temp DB and calls init_schema explicitly,
never assumes a table exists because it does on the developer's own machine.
"""
from __future__ import annotations

import pytest

import cfb
from store.database import SQLiteDatabase
from cfb.repositories import (TeamRepository, PlayerRepository, SyncLogRepository)


@pytest.fixture()
def db(tmp_path):
    d = SQLiteDatabase(tmp_path / "cfb_schema_test.db")
    cfb.init_schema(d)
    return d


def test_init_schema_is_idempotent(db):
    cfb.init_schema(db)   # second call must not raise (CREATE TABLE IF NOT EXISTS)
    cfb.init_schema(db)


def test_team_repository_upsert_dedupes_by_cfbd_id(db):
    repo = TeamRepository(db)
    a = repo.upsert("Ohio State", "Big Ten", "fbs", "OSU", cfbd_id="1")
    b = repo.upsert("Ohio State", "Big Ten", "fbs", "OSU", cfbd_id="1")
    assert a["id"] == b["id"]
    assert len(repo.all()) == 1


def test_team_repository_filters_by_classification(db):
    repo = TeamRepository(db)
    repo.upsert("Ohio State", "Big Ten", "fbs", "OSU", cfbd_id="1")
    repo.upsert("Some FCS School", "FCS Conf", "fcs", "SFC", cfbd_id="2")
    assert [t["school"] for t in repo.all(classification="fbs")] == ["Ohio State"]


def test_player_repository_create_and_update(db):
    repo = PlayerRepository(db)
    row = repo.create(full_name="Will Howard", position="QB", team="Ohio State")
    assert row["full_name"] == "Will Howard"
    repo.update(row["id"], team="Kansas State")
    assert repo.get(row["id"])["team"] == "Kansas State"


def test_player_repository_find_by_team_and_position(db):
    repo = PlayerRepository(db)
    repo.create(full_name="Will Howard", position="QB", team="Ohio State")
    repo.create(full_name="TreVeyon Henderson", position="RB", team="Ohio State")
    repo.create(full_name="Julian Sayin", position="QB", team="Alabama")
    assert len(repo.find_by_team("Ohio State")) == 2
    assert len(repo.find_by_position("QB")) == 2


def test_sync_log_records_and_reports_staleness(db):
    repo = SyncLogRepository(db)
    assert repo.is_stale("cfbd_fbs_rosters") is True
    repo.record("cfbd_fbs_rosters", 134)
    assert repo.is_stale("cfbd_fbs_rosters") is False
    assert repo.last_synced("cfbd_fbs_rosters") is not None
