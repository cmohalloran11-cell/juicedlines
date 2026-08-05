"""
Tests for optimizer.py — the Entry Optimizer.

Covers: the public PrizePicks payout table shape, the correlation heuristic (categorical,
not a fabricated coefficient), candidate generation's hard exclusion rules, Monte Carlo
sanity at the probability extremes (near-certain vs. near-impossible entries), Kelly bounds,
and an end-to-end today_report() smoke test over synthetic lines.
"""
from __future__ import annotations

import pytest

import db as _db
import optimizer


@pytest.fixture(autouse=True)
def _temp_db(monkeypatch, tmp_path):
    # optimizer.build_pool() calls db.stat_gammas()/backtest.current_accuracy(), both of
    # which query prop_clv — isolate with a fresh temp DB (same pattern as test_dashboard.py).
    monkeypatch.setattr(_db, "DB_PATH", tmp_path / "history_test.db", raising=False)
    _db.init_db()
    optimizer._CACHE.update(key=None, t=0.0, v=None)  # today_report() is TTL-cached; don't
    # let one test's result leak into the next test's (different DB / different lines).


def _line(id_, player, team, sport="MLB", model_prob=0.6, opponent=None, stat="Hits",
         start_time="2026-08-04T23:05:00Z", stat_side_losing=False):
    return {
        "id": id_, "player": player, "team": team, "opponent": opponent, "sport": sport,
        "stat": stat, "line": 1.5, "projection": 1.9, "median": 1.5, "edge": 0.4,
        "edgePct": 4.0, "probability": model_prob, "side": "over" if model_prob >= 0.5 else "under",
        "juiceScore": int(round(model_prob * 100)), "confidence": 70, "calibration": 0.5,
        "historicalAccuracy": 0.6, "modelAgreement": 0.6, "volatility": 0.6,
        "floor": 0.2, "ceiling": 3.2, "startTime": start_time, "source": "prizepicks",
        "oddsType": "standard", "headshot": None, "statSideLosing": stat_side_losing,
    }


def _pool(n=30, base_prob=0.60):
    # n distinct players on n distinct teams/games so combos never collide on correlation.
    return [_line(f"id{i}", f"Player {i}", f"T{i}", model_prob=base_prob) for i in range(n)]


# ── payout table ─────────────────────────────────────────────────────────────

def test_payout_table_shape():
    # every supported format pays MORE than the stake on a full hit
    for (picks, kind), table in optimizer.PAYOUTS.items():
        assert table[picks] > 1.0
        assert all(m >= 0 for m in table.values())
    # 2-pick Flex IS a real PrizePicks product (2026-08-04, confirmed): 2/2 pays 2.0x, 1/2
    # pays 0.5x — a partial-insurance payout even at the minimum pick count.
    assert optimizer.PAYOUTS[(2, "flex")] == {2: 2.0, 1: 0.5}
    assert (2, "power") in optimizer.PAYOUTS


# ── correlation heuristic ────────────────────────────────────────────────────

def test_correlation_tag_duplicate_player():
    a = _line("a", "Same Guy", "NYY")
    b = _line("b", "Same Guy", "NYY")
    assert optimizer.correlation_tag(a, b) == "duplicate_player"


def test_correlation_tag_teammates():
    a = _line("a", "P1", "NYY")
    b = _line("b", "P2", "NYY")
    assert optimizer.correlation_tag(a, b) == "teammate_stack"


def test_correlation_tag_opposing_same_game():
    a = _line("a", "P1", "NYY", opponent="BOS")
    b = _line("b", "P2", "BOS", opponent="NYY")
    assert optimizer.correlation_tag(a, b) == "same_game_opponents"


def test_correlation_tag_independent_by_default():
    a = _line("a", "P1", "NYY")
    b = _line("b", "P2", "LAD")
    assert optimizer.correlation_tag(a, b) == "independent"


def test_correlation_summary_levels():
    legs = [_line("a", "P1", "NYY"), _line("b", "P2", "LAD"), _line("c", "P3", "SEA")]
    summary = optimizer.correlation_summary(legs)
    assert summary["level"] == "low"
    assert summary["linkedPairs"] == 0


# ── candidate generation: hard exclusion rules ──────────────────────────────

def test_candidate_legsets_excludes_duplicate_players():
    pool = _pool(10)
    dupe = dict(pool[0]); dupe["id"] = "dup"
    combos = optimizer.candidate_legsets(pool + [dupe], picks=2, shortlist=200)
    for combo in combos:
        assert len({l["player"] for l in combo}) == 2


def test_candidate_legsets_excludes_correlated_pairs_by_default():
    pool = [_line("a", "P1", "NYY", model_prob=0.9), _line("b", "P2", "NYY", model_prob=0.9)]
    # both legs are on the same team (teammate_stack) — the ONLY pair available, and it must
    # be excluded, so no candidate should be generated at all.
    combos = optimizer.candidate_legsets(pool, picks=2, shortlist=10)
    assert combos == []


def test_candidate_legsets_too_few_eligible_legs_returns_empty():
    assert optimizer.candidate_legsets(_pool(1), picks=3) == []


def test_candidate_legsets_excludes_confirmed_losing_stat_sides():
    # analytics.attach_stat_trust's stat_side_losing flag (2026-08-05) — a stat/side the
    # model's own graded history shows losing (e.g. MLB Pitching Outs Under, 33.8% hit
    # rate). Still real, still visible on Projections; just not eligible for a
    # recommended/ranked entry — same treatment as duplicate players and correlated pairs.
    pool = _pool(10)
    pool[3] = dict(pool[3], statSideLosing=True)
    combos = optimizer.candidate_legsets(pool, picks=3, shortlist=200)
    losing_id = pool[3]["id"]
    assert combos, "should still find combos among the other 9 legs"
    for combo in combos:
        assert losing_id not in {l["id"] for l in combo}


# ── Monte Carlo sanity at the probability extremes ──────────────────────────

def test_simulate_entry_near_certain_legs():
    legs = [_line(f"id{i}", f"P{i}", f"T{i}", model_prob=0.995) for i in range(3)]
    sim = optimizer.simulate_entry(legs, picks=3, kind="power", n_sims=20_000)
    assert sim is not None
    assert sim["probabilityAllHit"] > 0.97
    # 3-pick power pays 5x on a hit -> EV should be close to +4.0 (5x - 1) when it's a lock
    assert sim["expectedROI"] == pytest.approx(4.0, abs=0.2)
    assert sim["chanceOfProfit"] > 0.97


def test_simulate_entry_near_impossible_legs():
    # _line()'s side auto-flips to whichever side is MORE likely, so a low model_prob alone
    # doesn't make the RECOMMENDED side unlikely — force side="over" explicitly here to build
    # a genuinely near-impossible recommended-side probability.
    legs = [_line(f"id{i}", f"P{i}", f"T{i}", model_prob=0.005) for i in range(3)]
    for l in legs:
        l["side"] = "over"
    sim = optimizer.simulate_entry(legs, picks=3, kind="power", n_sims=20_000)
    assert sim["probabilityAllHit"] < 0.01
    assert sim["expectedROI"] < -0.9
    assert sim["kellyFraction"] == 0.0
    assert sim["riskOfRuin"] == 0.0   # kelly=0 -> nothing staked -> can't be ruined


def test_simulate_entry_rejects_wrong_leg_count():
    legs = [_line("a", "P1", "T1")]
    assert optimizer.simulate_entry(legs, picks=2, kind="power") is None


def test_simulate_entry_rejects_unsupported_format():
    legs = [_line("a", "P1", "T1"), _line("b", "P2", "T2")]
    assert optimizer.simulate_entry(legs, picks=2, kind="jackpot") is None  # not a real format


def test_simulate_entry_2pick_flex_uses_partial_hit_insurance():
    # 2-pick Flex: near-certain legs should land near the 2/2 payout (2.0x -> EV ~+1.0);
    # this is the format's whole point (a 1/2 miss still pays 0.5x instead of losing the
    # stake entirely, unlike 2-pick Power).
    legs = [_line(f"id{i}", f"P{i}", f"T{i}", model_prob=0.995) for i in range(2)]
    sim = optimizer.simulate_entry(legs, picks=2, kind="flex", n_sims=20_000)
    assert sim is not None
    assert sim["expectedROI"] == pytest.approx(1.0, abs=0.1)


def test_kelly_fraction_is_bounded():
    legs = [_line(f"id{i}", f"P{i}", f"T{i}", model_prob=0.75) for i in range(2)]
    sim = optimizer.simulate_entry(legs, picks=2, kind="power", n_sims=30_000)
    assert 0.0 <= sim["kellyFraction"] <= 0.25


# ── extreme-EV review flag ───────────────────────────────────────────────────

def test_audit_entry_flags_extreme_roi_not_ordinary_ones():
    ordinary = {"expectedROI": 0.15}
    extreme = {"expectedROI": 1.75}
    assert optimizer.audit_entry(ordinary) is None
    flag = optimizer.audit_entry(extreme)
    assert flag is not None and flag["flagged"] is True and flag["expectedROI"] == 1.75


def test_build_entry_carries_review_flag_without_altering_the_number():
    legs = [_line(f"id{i}", f"P{i}", f"T{i}", model_prob=0.96) for i in range(2)]
    entry = optimizer.build_entry(legs, picks=2, kind="power", n_sims=20_000)
    assert entry is not None
    assert entry["simulation"]["expectedROI"] > optimizer.ENTRY_EV_REVIEW_THRESHOLD
    assert entry["reviewFlag"] is not None
    assert entry["reviewFlag"]["expectedROI"] == entry["simulation"]["expectedROI"]


# ── quality score ────────────────────────────────────────────────────────────

def test_quality_score_rewards_higher_ev_and_confidence():
    legs_strong = [_line(f"id{i}", f"P{i}", f"T{i}", model_prob=0.8) for i in range(2)]
    for l in legs_strong:
        l["confidence"] = 90
    legs_weak = [_line(f"idw{i}", f"Pw{i}", f"Tw{i}", model_prob=0.51) for i in range(2)]
    for l in legs_weak:
        l["confidence"] = 30
    sim_strong = optimizer.simulate_entry(legs_strong, picks=2, kind="power", n_sims=20_000)
    sim_weak = optimizer.simulate_entry(legs_weak, picks=2, kind="power", n_sims=20_000)
    corr = optimizer.correlation_summary(legs_strong)
    q_strong = optimizer.quality_score(sim_strong, legs_strong, corr)
    q_weak = optimizer.quality_score(sim_weak, legs_weak, optimizer.correlation_summary(legs_weak))
    assert q_strong > q_weak


def test_quality_score_penalizes_correlation_level():
    legs = [_line(f"id{i}", f"P{i}", f"T{i}", model_prob=0.8) for i in range(2)]
    sim = optimizer.simulate_entry(legs, picks=2, kind="power", n_sims=20_000)
    low = optimizer.quality_score(sim, legs, {"level": "low", "linkedPairs": 0, "flags": []})
    high = optimizer.quality_score(sim, legs, {"level": "high", "linkedPairs": 3, "flags": []})
    assert low > high


# ── end-to-end smoke test ────────────────────────────────────────────────────

def _raw_line(i, prob):
    return {
        "id": f"id{i}", "player": f"Player {i}", "team": f"T{i}", "sport": "MLB",
        "stat_type": "Hits", "line": 1.5, "model_proj": 1.7, "model_median": 1.5,
        "model_prob": prob, "model_n": 25, "proj_kind": "engine", "odds_type": "standard",
        "over_implied": 0.52, "under_implied": 0.52, "lineup_status": None,
        "start_time": f"2026-08-04T{18 + (i % 4)}:05:00Z",
        "model_floor": 0.2, "model_ceiling": 3.2, "source": "prizepicks",
    }


def test_today_report_end_to_end():
    lines = [_raw_line(i, 0.5 + ((i * 37) % 40 - 20) / 100.0) for i in range(40)]
    report = optimizer.today_report(lines, sport="MLB")
    assert report["poolSize"] == 40
    assert set(report["todaysBest"].keys()) == {f"{p}_{k}" for p, k in optimizer.FORMATS}
    for entries in report["todaysBest"].values():
        assert isinstance(entries, list)
        assert len(entries) <= optimizer.TODAYS_BEST_N
        seen_legsets = set()
        for entry in entries:
            assert 0 <= entry["qualityScore"] <= 100
            assert entry["simulation"]["simCount"] >= 100_000
            assert len(entry["legs"]) == entry["picks"]
            assert len({l["player"] for l in entry["legs"]}) == entry["picks"]
            legset = frozenset(l["id"] for l in entry["legs"])
            assert legset not in seen_legsets  # each entry in a format is a distinct combo
            seen_legsets.add(legset)
    assert isinstance(report["topEntries"], list)


def test_today_report_empty_pool_degrades_gracefully():
    report = optimizer.today_report([], sport="MLB")
    assert report["poolSize"] == 0
    assert all(v == [] for v in report["todaysBest"].values())
    assert report["topEntries"] == []
