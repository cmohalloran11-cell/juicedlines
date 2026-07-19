"""Soccer / World Cup projection regression tests.

Guards two fixes:
  * _prob_over — an integer line is a PUSH at X==line, so over = X >= line+1 (the old
    ceil(line) counted the push as a win and inflated P(over)/edge on whole-number lines).
  * analyze_soccer's note — must honestly describe the ESPN recent-form model, not the
    stale "no per-player game-log feed" text that contradicted the actual projection.
"""

import math
import tempfile
from pathlib import Path

import pytest

import analytics as A
import db


@pytest.fixture(autouse=True)
def _temp_db():
    # analyze_soccer reads line-movement history from SQLite; give it an empty temp store
    # so the note/field assertions run without a live history.db.
    db.DB_PATH = Path(tempfile.mktemp(suffix=".db"))
    db.init_db()
    yield
    try:
        db.DB_PATH.unlink()
    except OSError:
        pass


def test_prob_over_integer_line_excludes_push():
    # At lam == line, the mass sitting exactly on an integer line is a push, not an over,
    # so P(over) must be < 0.5 and strictly below the old ceil() value that double-counted it.
    lam = 3.0
    buggy = round(A._poisson_sf(max(1, math.ceil(3.0)), lam), 3)   # old: P(X>=3)
    fixed = A._prob_over(3.0, lam)                                 # new: P(X>=4)
    assert fixed < buggy
    assert fixed < 0.5


def test_prob_over_half_line_unchanged():
    # A .5 line can't push, so floor(line)+1 == ceil(line): behaviour is identical.
    for line, lam in [(2.5, 2.2), (0.5, 0.7), (4.5, 5.1)]:
        old = round(A._poisson_sf(max(1, math.ceil(line)), lam), 3)
        assert A._prob_over(line, lam) == old


def test_prob_over_monotonic_in_lambda():
    # More expected shots ⇒ higher P(over) for a fixed line.
    assert A._prob_over(2.5, 1.0) < A._prob_over(2.5, 2.0) < A._prob_over(2.5, 3.0)


def _soccer_line(**kw):
    base = {"id": "x1", "sport": "World Cup", "player": "Lionel Messi",
            "stat_type": "Shots", "line": 3, "position": "Attacker", "team": "Argentina"}
    base.update(kw)
    return base


def test_note_describes_recent_form_for_espn_kind():
    a = A.analyze_soccer(_soccer_line(proj_kind="espn", model_n=14, model_form=4.5,
                                      model_proj=3.4, model_edge=0.4, model_prob=0.43))
    note = a["note"].lower()
    assert "club matches" in note            # says what it actually used
    assert "market line" in note             # and that it blends toward the line
    assert "no free per-player" not in note  # the old, contradictory claim is gone
    assert a["model_form"] == 4.5            # raw recent rate surfaced for transparency


def test_note_is_honest_for_market_consensus_kind():
    a = A.analyze_soccer(_soccer_line(stat_type="Tackles", proj_kind="consensus",
                                      model_proj=2.5, model_edge=0.0))
    assert "market line" in a["note"].lower()
    assert "no independent model edge" in a["note"].lower()


def test_is_wc_comp():
    assert A._is_wc_comp("2026 FIFA World Cup")
    assert A._is_wc_comp("FIFA World Cup")
    assert not A._is_wc_comp("FIFA Club World Cup")   # a club tournament, not international
    assert not A._is_wc_comp("2025-26 Serie A")
    assert not A._is_wc_comp(None)


def _game(comp, shots):
    return {"_comp": comp, "totalShots": float(shots)}


def test_espn_projection_uses_wc_matches_plus_recent_club(monkeypatch):
    # 3 World Cup matches (shots 2,3,4) + 6 club games (all 5 shots). The WC games are the real
    # environment and must NOT be deflated; recent club form is deflated in and stabilises them.
    games = ([_game("2026 FIFA World Cup", s) for s in (2, 3, 4)]
             + [_game("2025-26 MLS", 5) for _ in range(6)])
    monkeypatch.setattr(A, "_espn_roster_map", lambda: {A.mlb._norm_name("Lionel Messi"): "123"})
    monkeypatch.setattr(A, "_espn_gamelog", lambda _aid: games)

    line = _soccer_line(id="m1", odds_type="standard")   # standard line → market anchor == 3
    done = A._attach_soccer_espn([line])

    assert "m1" in done
    assert line["proj_kind"] == "espn"
    assert line["model_n_wc"] == 3          # this World Cup's matches
    assert line["model_n_club"] == 6        # a few recent club games
    assert line["model_n"] == 9
    # raw WC shot rate is surfaced (recency-weighted 2,3,4 -> ~3.3), NOT the deflated value
    assert line["model_form"] == 3.3
    # projection sits between the WC-led form and the anchoring line, well under the old ~4
    assert 3.0 <= line["model_proj"] <= 3.6

    a = A.analyze_soccer(line)
    note = a["note"].lower()
    assert "3 world cup matches" in note
    assert "6 recent club games" in note


def test_wc_final_override_is_line_agnostic_and_wins():
    # The hand-set final projection must apply regardless of the exact line the book posts,
    # matching stat-label variants, and set proj_kind="final".
    lines = [
        {"id": "a", "sport": "World Cup", "player": "Lamine Yamal", "stat_type": "Shots", "line": 2.5},
        {"id": "b", "sport": "World Cup", "player": "Julián Álvarez", "stat_type": "Shots", "line": 2.5},
        {"id": "c", "sport": "World Cup", "player": "Lionel Messi", "stat_type": "Shots On Target", "line": 1.5},
        {"id": "d", "sport": "World Cup", "player": "Nobody Special", "stat_type": "Shots", "line": 2.5},
    ]
    A._attach_soccer_final(lines)

    yamal, alvarez, messi, other = lines
    assert yamal["proj_kind"] == "final" and yamal["model_proj"] == 3.6
    assert yamal["model_edge"] == 1.1                      # 3.6 vs 2.5
    # same projection, different line → edge/prob adapt (Álvarez 1.6 proj under a 2.5 line)
    assert alvarez["model_proj"] == 1.6 and alvarez["model_edge"] == -0.9
    assert alvarez["model_prob"] < 0.30
    # "Shots On Target" label maps to the sog projection
    assert messi["model_proj"] == 1.7
    # players not in the override are left untouched
    assert "model_proj" not in other

    # "Shots Assisted" (key passes) must NOT be caught by the assists projection
    assert A._final_stat_key("Shots Assisted") is None
    assert A._final_stat_key("Assists") == "assists"
    sa = [{"id": "sa", "sport": "World Cup", "player": "Lionel Messi",
           "stat_type": "Shots Assisted", "line": 1.5}]
    A._attach_soccer_final(sa)
    assert "model_proj" not in sa[0]

    a = A.analyze_soccer(lines[0])
    assert "final projection" in a["note"].lower()


def test_wc_final_override_self_expires(monkeypatch):
    import datetime as _dt

    class _Frozen(_dt.date):
        @classmethod
        def today(cls):
            return cls(2026, 7, 25)                        # after the final
    monkeypatch.setattr(A, "date", _Frozen)
    lines = [{"id": "a", "sport": "World Cup", "player": "Lamine Yamal", "stat_type": "Shots", "line": 2.5}]
    A._attach_soccer_final(lines)
    assert "model_proj" not in lines[0]                    # inert once the date passes


def test_espn_projection_falls_back_to_club_when_no_wc_games(monkeypatch):
    # No WC games in the log → recent club form (deflated), same as before this change.
    games = [_game("2025-26 Serie A", 4) for _ in range(8)]
    monkeypatch.setattr(A, "_espn_roster_map", lambda: {A.mlb._norm_name("Lionel Messi"): "123"})
    monkeypatch.setattr(A, "_espn_gamelog", lambda _aid: games)

    line = _soccer_line(id="m2", odds_type="standard")
    A._attach_soccer_espn([line])
    assert line["model_n_wc"] == 0
    assert line["model_n_club"] == 6        # capped at _CLUB_WINDOW
    a = A.analyze_soccer(line)
    assert "club matches" in a["note"].lower()
    assert "world cup match" not in a["note"].lower()
