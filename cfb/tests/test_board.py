"""
cfb.board.attach_cfb is a deliberate no-op stub (Phase 4 part A -- no projection math yet, see
its module docstring). These tests pin that contract down: it must never write model_proj (so
the ledger correctly refuses to log an unpriced CFB line, per cfb/tests/test_ledger.py), never
touch a non-CFB line, and never raise on a malformed/missing field.
"""
from __future__ import annotations

from cfb.board import attach_cfb


def test_attach_cfb_does_not_write_model_proj_yet():
    lines = [{"id": "cfb_1", "sport": "CFB", "player": "Will Howard",
             "stat_type": "Passing Yards", "line": 245.5}]
    attach_cfb(lines)
    assert lines[0].get("model_proj") is None


def test_attach_cfb_ignores_non_cfb_lines():
    lines = [{"id": "mlb_1", "sport": "MLB", "model_proj": 2.0}]
    attach_cfb(lines)
    assert lines[0]["model_proj"] == 2.0   # untouched


def test_attach_cfb_does_not_raise_on_missing_fields():
    attach_cfb([{"sport": "CFB"}, {}])
