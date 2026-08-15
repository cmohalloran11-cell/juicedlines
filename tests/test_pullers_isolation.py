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


# ── NFL sport detection (2026-08 wiring: PrizePicks/Underdog NFL now maps to sport="NFL") ──

def test_pp_league_nfl_maps_to_nfl():
    assert pullers._sport_from_pp_league("NFL") == "NFL"
    assert pullers._sport_from_pp_league("nfl") == "NFL"


def test_pp_league_nfl_period_subleagues_excluded():
    for league in ("NFL1H", "NFL1Q", "NFLSZN", "NFL 1H"):
        assert pullers._sport_from_pp_league(league) == "other"


def test_pp_league_cfl_stays_other_not_nfl():
    # CFL carries no "nfl" substring and has no engine -- must not accidentally match.
    assert pullers._sport_from_pp_league("CFL") == "other"


def test_pp_league_nfl_does_not_regress_other_sports():
    assert pullers._sport_from_pp_league("MLB") == "MLB"
    assert pullers._sport_from_pp_league("WNBA") == "WNBA"
    assert pullers._sport_from_pp_league("MLBSZN2") == "other"
    assert pullers._sport_from_pp_league("WNBA1Q") == "other"
    assert pullers._sport_from_pp_league("ATP") == "Tennis"


def test_ud_sport_map_nfl():
    assert pullers._UD_SPORT_MAP.get("NFL") == "NFL"
    assert pullers._UD_SPORT_MAP.get("CFL") == "other"


def test_fetch_prizepicks_resolves_real_nfl_projection_to_sport_nfl(monkeypatch):
    pullers._pp_result_cache.clear()

    nfl_proj = {
        "id": "n1", "type": "projection",
        "attributes": {"stat_type": "Receiving Yards", "line_score": 44.5,
                       "odds_type": "standard", "description": None,
                       "start_time": "2026-08-16T17:00:00Z", "status": "pre_game"},
        "relationships": {
            "new_player": {"data": {"type": "new_player", "id": "np1"}},
            "league": {"data": {"type": "league", "id": "lgnfl"}},
        },
    }
    included = [
        {"type": "new_player", "id": "np1",
         "attributes": {"display_name": "Justin Jefferson", "team": "MIN",
                        "position": "WR", "league": "NFL", "image_url": None}},
        {"type": "league", "id": "lgnfl", "attributes": {"name": "NFL"}},
    ]
    payload = {"data": [nfl_proj], "included": included}

    class FakeResp:
        status_code = 200
        def json(self):
            return payload

    monkeypatch.setattr(pullers, "_pp_get", lambda *a, **k: FakeResp())

    lines, err = pullers.fetch_prizepicks()
    assert err is None
    assert len(lines) == 1
    assert lines[0]["sport"] == "NFL"
    assert lines[0]["player"] == "Justin Jefferson"


def test_fetch_underdog_resolves_real_nfl_prop_to_sport_nfl():
    over = _prop(sport="NFL", stat="receiving_yards", player_name="Justin Jefferson O/U",
                 choice="over", over_under_id="ounfl1", line_id="lnfl1")
    under = _prop(sport="NFL", stat="receiving_yards", player_name="Justin Jefferson O/U",
                  choice="under", over_under_id="ounfl1", line_id="lnfl1")
    rows, skipped = pullers._ud_dedup([over, under])
    assert skipped == 0
    assert len(rows) == 1
    assert rows[0]["sport"] == "NFL"


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_") and callable(fn):
            fn(); print("ok", k)
