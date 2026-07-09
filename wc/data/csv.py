"""
CSV adapters — drop files in wc/data/files/ and set the matching source to `csv`.

Expected files & headers (extra columns ignored; missing file → empty list):
  fixtures.csv : id,home,away,date,stage,neutral,knockout,rivalry,host_home
  strength.csv : team,gf_pg,ga_pg,xg_pg,xga_pg,shots_pg,possession
  players.csv  : name,team,position,minutes,shots90,sot90,xg90,xg_share,fouls90,
                 yellow90,red90,save_pct,start_prob
  odds.csv     : match_id,player,market,line,side,price,book
"""

from __future__ import annotations

import csv as _csv
from pathlib import Path

from .base import (Fixture, TeamStrength, Player, OddsLine,
                   FixtureSource, TeamStrengthSource, PlayerSource, OddsSource)

_DIR = Path(__file__).parent / "files"


def _rows(name: str) -> list[dict]:
    f = _DIR / name
    if not f.exists():
        return []
    with f.open(newline="", encoding="utf-8") as fh:
        return list(_csv.DictReader(fh))


def _f(r, k, d=0.0):
    try:
        return float(r.get(k, d) or d)
    except (TypeError, ValueError):
        return d


def _b(r, k):
    return str(r.get(k, "")).strip().lower() in ("1", "true", "yes", "y")


class CsvFixtures(FixtureSource):
    def fixtures(self):
        return [Fixture(r["id"], r["home"], r["away"], r.get("date", ""),
                        stage=r.get("stage", "group"), neutral=_b(r, "neutral"),
                        knockout=_b(r, "knockout"), rivalry=_b(r, "rivalry"),
                        host_home=(r.get("host_home") or None)) for r in _rows("fixtures.csv")]


class CsvStrength(TeamStrengthSource):
    def __init__(self):
        self._by = {r["team"]: r for r in _rows("strength.csv")}

    def strength(self, team):
        r = self._by.get(team)
        return TeamStrength(team, _f(r, "gf_pg"), _f(r, "ga_pg"), _f(r, "xg_pg"),
                            _f(r, "xga_pg"), _f(r, "shots_pg", 12), _f(r, "possession", 0.5)) if r else None


class CsvPlayers(PlayerSource):
    def __init__(self):
        self._rows = _rows("players.csv")

    def players(self, team):
        return [Player(r["name"], team, r.get("position", "MF"), _f(r, "minutes"),
                       _f(r, "shots90"), _f(r, "sot90"), _f(r, "xg90"), _f(r, "xg_share"),
                       _f(r, "fouls90"), _f(r, "yellow90"), _f(r, "red90"),
                       _f(r, "save_pct"), _f(r, "start_prob", 0.8))
                for r in self._rows if r.get("team") == team]


class CsvOdds(OddsSource):
    def __init__(self):
        self._rows = _rows("odds.csv")

    def odds(self, match_id):
        out = []
        for r in self._rows:
            if r.get("match_id") != match_id:
                continue
            line = r.get("line")
            out.append(OddsLine(match_id, r["player"], r.get("market", "goal"),
                                float(line) if line not in (None, "", "None") else None,
                                r.get("side", "over"), _f(r, "price"), r.get("book", "csv")))
        return out
