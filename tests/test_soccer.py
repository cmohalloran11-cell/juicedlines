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
