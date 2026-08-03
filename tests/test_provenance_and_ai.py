"""
Tests for the provenance stamp and AI Juice's grounding/degradation contract.

The AI tests deliberately do NOT make a network call — they verify the safety contract
(no key → clean unavailable response; grounding context contains only real fields).
"""
from __future__ import annotations

import os

import provenance
import ai_juice


# ── provenance ────────────────────────────────────────────────────────────────

def test_stamp_only_projected_lines_and_mirror_ledger_fields():
    lines = [
        {"id": "a", "sport": "MLB", "model_proj": 1.7, "proj_kind": "engine", "model_n": 20},
        {"id": "b", "sport": "MLB", "line": 1.5},  # no projection → not stamped
    ]
    n = provenance.stamp_lines(lines, data_snapshot="2026-07-26T00:00:00+00:00")
    assert n == 1
    a, b = lines
    assert "provenance" in a and "provenance" not in b
    # ledger-persisted flat fields mirror the nested provenance
    assert (a["model_version"] == a["provenance"]["model_version"]
            == provenance.model_version("MLB"))
    assert a["feature_version"] == provenance.FEATURE_VERSION
    assert a["data_snapshot"] == "2026-07-26"
    assert a["provenance"]["sims"] == 5000 and a["provenance"]["method"] == "engine"


def test_versions_endpoint_payload_shape():
    v = provenance.versions()
    for key in ("model_versions", "feature_version", "simulation_version",
                "calibration_version", "schema_version"):
        assert key in v
    assert v["model_versions"]["MLB"] == provenance.model_version("MLB")


# ── AI Juice ──────────────────────────────────────────────────────────────────

def test_ai_unavailable_without_key(monkeypatch):
    # Clear every provider key (a loaded .env may have set GEMINI_API_KEY process-wide).
    for k in ("ANTHROPIC_API_KEY", "AI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert ai_juice.available() is False
    r = ai_juice.explain({"player": "X", "stat_type": "Hits", "model_proj": 1.7})
    assert r["available"] is False and "not configured" in r["reason"]


def test_ai_explain_refuses_line_without_projection(monkeypatch):
    # With a key present, a line with no projection is refused BEFORE any network call.
    monkeypatch.setattr(ai_juice, "PROVIDER", "gemini", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert ai_juice.available() is True
    r = ai_juice.explain({"player": "X", "stat_type": "Hits"})   # no model_proj
    assert r["available"] is False and "No projection" in r["reason"]


def test_ai_context_block_contains_only_supplied_facts():
    line = {"player": "Judge", "stat_type": "Hits", "line": 1.5, "model_proj": 1.7,
            "model_n": 22, "proj_kind": "engine", "drivers": ["park"],
            "provenance": {"model_version": "mlb-1.0.0", "data_snapshot": "2026-07-26"}}
    ctx = ai_juice._context_block(line)
    assert "Judge" in ctx and "park" in ctx and "mlb-1.0.0" in ctx
    # a fact NOT supplied must not appear
    assert "xBA" not in ctx and "barrel" not in ctx
