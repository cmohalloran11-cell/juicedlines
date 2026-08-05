"""
Tests for analytics.attach_stat_trust's 2026-08-05 shrinkage — applying db.stat_gammas'
already-documented formula (proj = line + gamma*(model-line)) to MLB's actual model_proj/
model_prob, instead of only ever discounting juice_score's EV component. Live audit found
the gap: MLB "Pitching Outs" measured gamma=0.0 (proven zero edge) but was still recommended
Under 91.4% of the time on live board data, hitting only 33.8%.
"""
from __future__ import annotations

import pytest

import analytics


def _line(sport, stat, line, proj, prob, **extra):
    return {"sport": sport, "stat_type": stat, "line": line,
           "model_proj": proj, "model_prob": prob, **extra}


def _no_losing(monkeypatch):
    monkeypatch.setattr(analytics.db, "stat_side_losing", lambda sport, **kw: {})


def test_zero_gamma_shrinks_fully_to_the_market_line(monkeypatch):
    monkeypatch.setattr(analytics.db, "stat_gammas", lambda sport, **kw: {"pitching outs": 0.0})
    _no_losing(monkeypatch)
    l = _line("MLB", "Pitching Outs", line=15.5, proj=17.0, prob=0.62)
    analytics.attach_stat_trust([l])
    assert l["stat_trust_gamma"] == 0.0
    assert l["model_proj"] == 15.5          # fully deferred to the line
    assert l["model_prob"] == 0.5           # fully deferred to a coin flip
    assert l["model_edge"] == 0.0
    assert l["model_raw"] == 17.0           # pre-shrinkage value preserved
    assert l["model_raw_prob"] == 0.62


def test_full_gamma_leaves_the_projection_unchanged(monkeypatch):
    monkeypatch.setattr(analytics.db, "stat_gammas", lambda sport, **kw: {"hits": 1.0})
    _no_losing(monkeypatch)
    l = _line("MLB", "Hits", line=1.5, proj=1.9, prob=0.58)
    analytics.attach_stat_trust([l])
    assert l["model_proj"] == 1.9
    assert l["model_prob"] == 0.58
    assert l["model_edge"] == pytest.approx(0.4)


def test_partial_gamma_blends_proportionally(monkeypatch):
    monkeypatch.setattr(analytics.db, "stat_gammas", lambda sport, **kw: {"walks": 0.4})
    _no_losing(monkeypatch)
    l = _line("MLB", "Walks", line=0.5, proj=0.9, prob=0.7)
    analytics.attach_stat_trust([l])
    # proj = 0.5 + 0.4*(0.9-0.5) = 0.66 ; prob = 0.5 + 0.4*(0.7-0.5) = 0.58
    assert l["model_proj"] == pytest.approx(0.66)
    assert l["model_prob"] == pytest.approx(0.58)


def test_shrinkage_never_flips_the_recommended_side(monkeypatch):
    # A shrink-toward-neutral transformation can soften confidence but can never move a
    # probability past 0.5 to the other side — this is a mathematical property of
    # 0.5 + gamma*(p-0.5) for gamma in [0,1], not a corner case. Locking it in because it's
    # the exact reason this fix alone doesn't change an over/under recommendation count.
    _no_losing(monkeypatch)
    for gamma in (0.0, 0.1, 0.3, 0.5, 0.7, 1.0):
        monkeypatch.setattr(analytics.db, "stat_gammas", lambda sport, g=gamma: {"runs": g})
        l = _line("MLB", "Runs", line=0.5, proj=0.6, prob=0.47)   # starts as "under"
        analytics.attach_stat_trust([l])
        assert l["model_prob"] <= 0.5, f"flipped sides at gamma={gamma}"


def test_wnba_and_tennis_are_not_shrunk(monkeypatch):
    # WNBA already anchors via basketball.board's own trust/sample_weight blend; tennis is
    # already fully market-deferred. Applying this shrinkage on top of either would
    # double-shrink — scoped to MLB only.
    monkeypatch.setattr(analytics.db, "stat_gammas", lambda sport, **kw: {"points": 0.0})
    wnba = _line("WNBA", "Points", line=18.5, proj=22.0, prob=0.7)
    tennis = _line("Tennis", "Points", line=18.5, proj=22.0, prob=0.7)
    analytics.attach_stat_trust([wnba, tennis])
    assert wnba["model_proj"] == 22.0 and wnba["model_prob"] == 0.7
    assert tennis["model_proj"] == 22.0 and tennis["model_prob"] == 0.7
    assert wnba["stat_trust_gamma"] == 0.0   # gamma is still attached for both


def test_missing_projection_or_line_is_skipped_gracefully(monkeypatch):
    monkeypatch.setattr(analytics.db, "stat_gammas", lambda sport, **kw: {"hits": 0.2})
    _no_losing(monkeypatch)
    no_proj = {"sport": "MLB", "stat_type": "Hits", "line": 1.5, "model_proj": None, "model_prob": None}
    no_line = {"sport": "MLB", "stat_type": "Hits", "line": None, "model_proj": 1.9, "model_prob": 0.6}
    analytics.attach_stat_trust([no_proj, no_line])
    assert "model_raw" not in no_proj
    assert "model_raw" not in no_line


def test_does_not_double_shrink_if_called_twice(monkeypatch):
    monkeypatch.setattr(analytics.db, "stat_gammas", lambda sport, **kw: {"walks": 0.4})
    _no_losing(monkeypatch)
    l = _line("MLB", "Walks", line=0.5, proj=0.9, prob=0.7)
    analytics.attach_stat_trust([l])
    first_raw, first_proj, first_prob = l["model_raw"], l["model_proj"], l["model_prob"]
    analytics.attach_stat_trust([l])
    assert l["model_raw"] == first_raw     # not overwritten with the already-shrunk value
    assert l["model_proj"] == first_proj   # re-shrinking the already-shrunk value would move it
    assert l["model_prob"] == first_prob


# ── stat_side_losing attachment (2026-08-05) ────────────────────────────────────

def test_confirmed_losing_side_is_flagged_on_the_line(monkeypatch):
    monkeypatch.setattr(analytics.db, "stat_gammas", lambda sport, **kw: {"pitching outs": 0.5})
    monkeypatch.setattr(analytics.db, "stat_side_losing", lambda sport, **kw:
                        {("pitching outs", "under"): {"hit_rate": 0.338, "shrunk_hit_rate": 0.38, "n": 139}})
    l = _line("MLB", "Pitching Outs", line=14.5, proj=13.0, prob=0.35)   # recommends "under"
    analytics.attach_stat_trust([l])
    assert l["stat_side_losing"] is True
    assert l["stat_side_losing_detail"]["side"] == "under"
    assert l["stat_side_losing_detail"]["n"] == 139


def test_winning_side_is_not_flagged_even_with_other_losing_entries(monkeypatch):
    monkeypatch.setattr(analytics.db, "stat_gammas", lambda sport, **kw: {"pitching outs": 0.5})
    monkeypatch.setattr(analytics.db, "stat_side_losing", lambda sport, **kw:
                        {("pitching outs", "under"): {"hit_rate": 0.338, "shrunk_hit_rate": 0.38, "n": 139}})
    l = _line("MLB", "Pitching Outs", line=14.5, proj=16.0, prob=0.65)   # recommends "over"
    analytics.attach_stat_trust([l])
    assert l["stat_side_losing"] is False
    assert "stat_side_losing_detail" not in l


def test_losing_side_check_uses_the_pre_shrinkage_lean_not_the_collapsed_tie(monkeypatch):
    # gamma=0.0 collapses model_prob to EXACTLY 0.5 for every prop on that stat (see
    # test_zero_gamma_shrinks_fully_to_the_market_line) — if the losing-side check read the
    # post-shrinkage probability, every single one of them would tie-break to the same side
    # regardless of each player's real original lean. Verified live 2026-08-05: this bug
    # made all 122 MLB Pitching Outs props resolve to "under" after the shrinkage fix
    # shipped; the fix here found 74 genuinely originally-under vs 48 originally-over.
    monkeypatch.setattr(analytics.db, "stat_gammas", lambda sport, **kw: {"pitching outs": 0.0})
    monkeypatch.setattr(analytics.db, "stat_side_losing", lambda sport, **kw:
                        {("pitching outs", "under"): {"hit_rate": 0.33, "shrunk_hit_rate": 0.38, "n": 145}})
    originally_over = _line("MLB", "Pitching Outs", line=14.5, proj=16.0, prob=0.55)
    originally_under = _line("MLB", "Pitching Outs", line=14.5, proj=13.0, prob=0.35)
    analytics.attach_stat_trust([originally_over, originally_under])
    assert originally_over["model_prob"] == 0.5 and originally_under["model_prob"] == 0.5
    assert originally_over["stat_side_losing"] is False
    assert originally_under["stat_side_losing"] is True


def test_wnba_is_never_flagged_losing(monkeypatch):
    # stat_side_losing is MLB-only, same scoping rationale as the shrinkage above.
    monkeypatch.setattr(analytics.db, "stat_gammas", lambda sport, **kw: {"points": 0.5})
    monkeypatch.setattr(analytics.db, "stat_side_losing", lambda sport, **kw:
                        {("points", "under"): {"hit_rate": 0.3, "shrunk_hit_rate": 0.35, "n": 200}})
    l = _line("WNBA", "Points", line=18.5, proj=15.0, prob=0.35)
    analytics.attach_stat_trust([l])
    assert "stat_side_losing" not in l
