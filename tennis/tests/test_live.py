"""
Unit tests for the 2026-07-29 tennis engine rebuild's live-data bridge: the Elo model's
name-keyed history+live composition, the rank-based cold-start prior, recency-weighted
rate fitting, per-match fatigue adjustment, and model-agreement confidence. All pure/
synthetic — no network calls.
"""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tennis.model.elo import EloModel
from tennis.model.rates import fit as rates_fit, PlayerRates
from tennis.projections import _apply_fatigue, _model_agreement


@dataclass
class _M:
    """Minimal historical-match stand-in — just what EloModel.fit()/rates.fit() read."""
    player: str
    opp: str
    surface: str
    date: str
    won: bool = True
    player_id: str = ""
    opp_id: str = ""
    serve_won: int = 0
    serve_played: int = 0
    return_won: int = 0
    return_played: int = 0
    aces: int = 0
    dfs: int = 0
    sv_games: int = 0


# ── Elo: name-keyed history + live composition ────────────────────────────────────

def test_elo_keys_by_name_not_source_id():
    # "Historical" fit uses one id scheme; a second fit call (simulating the live ESPN
    # layer, which has NO relation to the historical id scheme) must extend the SAME
    # rating, not create a disconnected entry — the whole point of the name bridge.
    elo = EloModel("ATP")
    hist = [_M(player="Novak Djokovic", opp="Random Player", surface="Hard",
              date="2020-01-01", player_id="104925", opp_id="999999")]
    elo.fit(hist)
    r_after_hist = elo.rating("Novak Djokovic")
    assert r_after_hist > elo.start   # won his historical match → rating above the 1500 start

    live = [_M(player="Novak Djokovic", opp="Random Player", surface="Hard", date="2026-01-01")]
    elo.fit(live)
    r_after_live = elo.rating("Novak Djokovic")
    assert r_after_live > r_after_hist   # a SECOND win further raises the SAME rating


def test_elo_rating_and_win_prob_accept_raw_names():
    elo = EloModel("ATP")
    elo.fit([_M(player="A Player", opp="B Player", surface="Hard", date="2025-01-01")])
    # rating()/win_prob() normalize internally — callers just pass the name as given.
    assert elo.rating("A Player") == elo.rating("a player")
    assert elo.win_prob("A Player", "B Player") > 0.5


def test_seed_from_rank_never_overwrites_a_real_rating():
    elo = EloModel("ATP")
    elo.fit([_M(player="Known Player", opp="Someone Else", surface="Hard", date="2025-01-01")])
    fitted = elo.rating("Known Player")
    elo.seed_from_rank({"known player": 1, "brand new player": 3})
    assert elo.rating("Known Player") == fitted           # untouched — a real fit exists
    seeded = elo.rating("Brand New Player")
    assert seeded > elo.start                              # rank 3 → above the flat 1500 start
    assert seeded < elo.rating("Known Player") + 1000       # sane magnitude, not a runaway value


def test_seed_from_rank_monotonic_in_rank():
    elo = EloModel("ATP")
    elo.seed_from_rank({"rank one": 1, "rank fifty": 50, "rank two hundred": 200})
    assert elo.rating("Rank One") > elo.rating("Rank Fifty") > elo.rating("Rank Two Hundred")


# ── rates.py: recency weighting ───────────────────────────────────────────────────

def test_recent_match_weighted_more_than_old_match():
    # Two matches for the same player, opposite serve outcomes, at very different dates.
    # If the RECENT one is weighted more, the fitted spw should sit closer to the recent
    # match's own rate than to a simple unweighted average of the two.
    old = _M(player="P", opp="X", surface="Hard", date="2016-01-01", player_id="1", opp_id="2",
             serve_won=50, serve_played=100, return_won=40, return_played=100)   # spw=0.50
    recent = _M(player="P", opp="Y", surface="Hard", date="2022-06-01", player_id="1", opp_id="3",
                serve_won=90, serve_played=100, return_won=40, return_played=100)  # spw=0.90
    base, by_id = rates_fit([old, recent], "ATP")
    p = by_id["1"]
    unweighted_avg = (50 + 90) / (100 + 100)   # 0.70
    assert p.spw > unweighted_avg   # recency weighting pulls it above the plain average, toward 0.90


def test_n_matches_stays_a_raw_unweighted_count():
    m1 = _M(player="P", opp="X", surface="Hard", date="2016-01-01", player_id="1", opp_id="2",
            serve_won=50, serve_played=100)
    m2 = _M(player="P", opp="Y", surface="Hard", date="2022-01-01", player_id="1", opp_id="3",
            serve_won=50, serve_played=100)
    _, by_id = rates_fit([m1, m2], "ATP")
    assert by_id["1"].n_matches == 2   # confidence bucketing needs a real count, not a decayed one


# ── projections.py: fatigue + model agreement ─────────────────────────────────────

def test_apply_fatigue_does_not_mutate_the_shared_rates_object():
    original = PlayerRates("1", "P", "ATP", spw=0.65, rpw=0.35, ace_rate=0.08, df_rate=0.03,
                           pts_per_svgame=6.7, surface_spw={"Hard": 0.66})
    shifted = _apply_fatigue(original, -0.02)
    assert shifted.spw == 0.63
    assert shifted.surface_spw["Hard"] == 0.64
    assert original.spw == 0.65                # the cached shared object is untouched
    assert original.surface_spw["Hard"] == 0.66
    assert _apply_fatigue(original, 0.0) is original   # no-op shortcut, no needless copy


def test_model_agreement_perfect_vs_wide_spread():
    assert _model_agreement(0.6, 0.6, 0.6) == 1.0
    assert _model_agreement(0.9, 0.5, 0.1) == 0.0
    mid = _model_agreement(0.55, 0.60, 0.50)
    assert 0.0 < mid < 1.0


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn(); print("ok", fn.__name__)
