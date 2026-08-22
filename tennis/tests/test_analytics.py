"""
tennis/analytics.py::analyze() never returned a "recent" (recently-completed matches) field
at all, for every player, even though a real live source already exists in the codebase
(tennis/data/espn.py::completed_matches — already used by board.py for the fatigue-penalty
and live-surface signals, just never surfaced in the drawer). Found via a live investigation,
2026-08: "recent games missing for some players" turned out to be "recent games missing for
EVERY player" for tennis specifically, a structural gap, not a per-player data-matching bug.

All offline — monkeypatches tennis.data.espn.completed_matches directly, no network.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tennis import analytics as A
from tennis.data import espn as _espn


def _match(a_name, b_name, a_winner, date="2026-08-15", tournament="Cincinnati Open",
          round_="Round of 16", surface="Hard", a_games=None, b_games=None, completed=True):
    return {
        "id": "1", "tournament": tournament, "date": date, "round": round_,
        "surface": surface, "completed": completed, "best_of": 3,
        "a_id": "100", "a_name": a_name, "a_winner": a_winner,
        "a_games": a_games or [6, 4, 6], "b_id": "200", "b_name": b_name,
        "b_winner": not a_winner, "b_games": b_games or [3, 6, 2],
    }


def test_recent_matches_finds_the_player_on_either_side_of_the_draw(monkeypatch):
    matches = [_match("Jannik Sinner", "Carlos Alcaraz", a_winner=True)]
    monkeypatch.setattr(_espn, "completed_matches", lambda tour, days_back=14: matches)
    won = A._recent_matches("Jannik Sinner", "ATP")
    assert len(won) == 1 and won[0]["won"] is True and won[0]["opponent"] == "Carlos Alcaraz"
    lost = A._recent_matches("Carlos Alcaraz", "ATP")
    assert len(lost) == 1 and lost[0]["won"] is False and lost[0]["opponent"] == "Jannik Sinner"


def test_recent_matches_ignores_matches_the_player_wasnt_in(monkeypatch):
    matches = [_match("Jannik Sinner", "Carlos Alcaraz", a_winner=True)]
    monkeypatch.setattr(_espn, "completed_matches", lambda tour, days_back=14: matches)
    assert A._recent_matches("Novak Djokovic", "ATP") == []


def test_recent_matches_sorts_most_recent_first_and_respects_limit(monkeypatch):
    matches = [_match("Jannik Sinner", "Player B", True, date="2026-07-01"),
              _match("Jannik Sinner", "Player C", True, date="2026-08-10"),
              _match("Jannik Sinner", "Player D", True, date="2026-07-20")]
    monkeypatch.setattr(_espn, "completed_matches", lambda tour, days_back=14: matches)
    out = A._recent_matches("Jannik Sinner", "ATP", limit=2)
    assert len(out) == 2
    assert [r["date"] for r in out] == ["2026-08-10", "2026-07-20"]


def test_recent_matches_score_reads_as_own_games_dash_opponent_games(monkeypatch):
    matches = [_match("Jannik Sinner", "Carlos Alcaraz", a_winner=True,
                      a_games=[6, 4, 6], b_games=[3, 6, 2])]
    monkeypatch.setattr(_espn, "completed_matches", lambda tour, days_back=14: matches)
    out = A._recent_matches("Jannik Sinner", "ATP")
    assert out[0]["score"] == "6/3-4/6-6/2"
    out2 = A._recent_matches("Carlos Alcaraz", "ATP")
    assert out2[0]["score"] == "3/6-6/4-2/6"


def test_recent_matches_degrades_to_empty_on_an_espn_failure(monkeypatch):
    def boom(tour, days_back=14):
        raise ConnectionError("no route to host")
    monkeypatch.setattr(_espn, "completed_matches", boom)
    assert A._recent_matches("Jannik Sinner", "ATP") == []


def test_analyze_surfaces_recent_matches_and_no_note_when_present(monkeypatch):
    monkeypatch.setattr(A.P, "_model", lambda tour: {"name": set(), "elo": None})
    matches = [_match("Star Player", "Someone Else", a_winner=True)]
    monkeypatch.setattr(_espn, "completed_matches", lambda tour, days_back=14: matches)
    out = A.analyze({"player": "Star Player", "stat_type": "Aces", "line": 8.5})
    assert out["recent"] == A._recent_matches("Star Player", "ATP")
    assert len(out["recent"]) == 1
    assert out["recent_note"] is None


def test_analyze_gives_an_honest_note_when_no_recent_matches_exist(monkeypatch):
    monkeypatch.setattr(A.P, "_model", lambda tour: {"name": set(), "elo": None})
    monkeypatch.setattr(_espn, "completed_matches", lambda tour, days_back=14: [])
    out = A.analyze({"player": "Nobody Played Lately", "stat_type": "Aces", "line": 8.5})
    assert out["recent"] == []
    assert out["recent_note"] and "no completed matches" in out["recent_note"].lower()


def test_analyze_tries_both_tours_when_the_sackmann_model_has_never_seen_the_player(monkeypatch):
    """A player with zero rows in the frozen historical mirror (a brand-new tour player,
    e.g. Sinner/Alcaraz as of when this adapter was written) must not also lose real
    recent-match history just because `tour` never resolved."""
    monkeypatch.setattr(A.P, "_model", lambda tour: {"name": set(), "elo": None})
    matches = [_match("New Player", "Someone", a_winner=True)]
    def fake_completed(tour, days_back=14):
        return matches if tour == "WTA" else []
    monkeypatch.setattr(_espn, "completed_matches", fake_completed)
    out = A.analyze({"player": "New Player", "stat_type": "Aces", "line": 8.5})
    assert len(out["recent"]) == 1
    assert out["tour"] == "WTA"
