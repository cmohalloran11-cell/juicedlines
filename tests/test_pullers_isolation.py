"""
2026-08 regression guard: one malformed record from a live feed must never drop the
WHOLE pull. Before this fix, `_pp_line`/`_ud_dedup` ran unguarded inside their loops,
so a single bad projection/prop out of ~12,000 raised out to the outer try/except and
fetch_prizepicks/fetch_underdog returned `[]` for the entire refresh cycle.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pullers


def _prop(sport="MLB", stat="Hits", player_name="X O/U", line=1.5, choice="over",
          american_price="-110", is_boosted=False, image_url=None, country=None,
          position=None, status="pre_game", line_id="l1", match_id="m1",
          over_under_id="ou1"):
    return types.SimpleNamespace(
        sport=sport, stat=stat, player_name=player_name, line=line, choice=choice,
        american_price=american_price, is_boosted=is_boosted, image_url=image_url,
        country=country, position=position, status=status, line_id=line_id,
        match_id=match_id, raw={"over_under_id": over_under_id, "options": []},
    )


def test_ud_dedup_isolates_one_malformed_prop_and_keeps_the_rest():
    good_over = _prop(choice="over", over_under_id="ou1", line_id="l1")
    good_under = _prop(choice="under", over_under_id="ou1", line_id="l1")
    broken = types.SimpleNamespace(is_boosted=False, stat="Hits", player_name="Y",
                                    raw=None)   # raw=None -> raw.get(...) raises AttributeError
    rows, skipped = pullers._ud_dedup([good_over, broken, good_under])
    assert skipped == 1
    assert len(rows) == 1
    assert rows[0]["over_price"] == "-110" and rows[0]["under_price"] == "-110"


def test_ud_dedup_all_clean_reports_zero_skipped():
    rows, skipped = pullers._ud_dedup([_prop()])
    assert skipped == 0
    assert len(rows) == 1


def test_fetch_prizepicks_isolates_one_malformed_projection(monkeypatch):
    pullers._pp_result_cache.clear()

    good = {
        "id": "1", "type": "projection",
        "attributes": {"stat_type": "Hits", "line_score": 1.5, "odds_type": "standard",
                        "description": None, "start_time": None, "status": "pre_game"},
        "relationships": {
            "new_player": {"data": {"type": "new_player", "id": "p1"}},
            "league": {"data": {"type": "league", "id": "lg1"}},
        },
    }
    included = [
        {"type": "new_player", "id": "p1",
         "attributes": {"display_name": "Player One", "team": "NYY", "position": "OF",
                        "league": "MLB", "image_url": None}},
        {"type": "league", "id": "lg1", "attributes": {"name": "MLB"}},
    ]
    payload = {"data": [good, None], "included": included}   # None = malformed record

    class FakeResp:
        status_code = 200
        def json(self):
            return payload

    monkeypatch.setattr(pullers, "_pp_get", lambda *a, **k: FakeResp())

    lines, err = pullers.fetch_prizepicks()
    assert len(lines) == 1
    assert lines[0]["player"] == "Player One"
    assert err is not None and "skipped 1" in err


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_") and callable(fn):
            fn(); print("ok", k)
