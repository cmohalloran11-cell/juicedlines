"""
Sackmann-format history adapter (prototype/backtest source).

Pulls ATP/WTA match CSVs from a Sackmann-format mirror (config `repos`), caches them
to tennis/data/cache/, and parses each match into TWO PlayerMatch records (winner +
loser) so serve and return are both first-class. Skips rows missing serve counts.

⚠ Sackmann-derived data is Creative Commons NON-COMMERCIAL — use to build and
validate the model only, never to power production. Swap for a licensed feed there.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from ..config import cfg
from .base import PlayerMatch, MatchHistorySource

_CACHE = Path(__file__).parent / "cache"
_RAW = "https://raw.githubusercontent.com/{repo}/master/{fname}"


def _int(v, d=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return d


def _fetch_csv(repo: str, fname: str) -> str | None:
    """Return CSV text for repo/fname, disk-cached. None if unavailable."""
    _CACHE.mkdir(exist_ok=True)
    cached = _CACHE / f"{repo.replace('/', '_')}__{fname}"
    if cached.exists():
        return cached.read_text(encoding="utf-8")
    try:
        import requests
        r = requests.get(_RAW.format(repo=repo, fname=fname), timeout=30)
    except Exception as exc:  # pragma: no cover
        print(f"[sackmann] {fname}: {exc}")
        return None
    if r.status_code != 200 or len(r.text) < 100:
        return None
    cached.write_text(r.text, encoding="utf-8")
    return r.text


def _rows_to_player_matches(rows: list[dict]) -> list[PlayerMatch]:
    out: list[PlayerMatch] = []
    for m in rows:
        # need both players' serve counts to derive serve+return
        if not (m.get("w_svpt") and m.get("l_svpt")):
            continue
        w_svpt, l_svpt = _int(m["w_svpt"]), _int(m["l_svpt"])
        if w_svpt <= 0 or l_svpt <= 0:
            continue
        w_swon = _int(m.get("w_1stWon")) + _int(m.get("w_2ndWon"))
        l_swon = _int(m.get("l_1stWon")) + _int(m.get("l_2ndWon"))
        common = dict(date=str(m.get("tourney_date", "")), tournament=m.get("tourney_name", ""),
                      surface=m.get("surface") or "Hard", best_of=_int(m.get("best_of"), 3),
                      round=m.get("round", ""), score=m.get("score", ""))
        # winner's line
        out.append(PlayerMatch(
            **common, player_id=str(m.get("winner_id", "")), player=m.get("winner_name", ""),
            opp_id=str(m.get("loser_id", "")), opp=m.get("loser_name", ""), won=True,
            serve_won=w_swon, serve_played=w_svpt,
            return_won=l_svpt - l_swon, return_played=l_svpt,
            aces=_int(m.get("w_ace")), dfs=_int(m.get("w_df")),
            bp_faced=_int(m.get("w_bpFaced")), bp_saved=_int(m.get("w_bpSaved")),
            sv_games=_int(m.get("w_SvGms"))))
        # loser's line
        out.append(PlayerMatch(
            **common, player_id=str(m.get("loser_id", "")), player=m.get("loser_name", ""),
            opp_id=str(m.get("winner_id", "")), opp=m.get("winner_name", ""), won=False,
            serve_won=l_swon, serve_played=l_svpt,
            return_won=w_svpt - w_swon, return_played=w_svpt,
            aces=_int(m.get("l_ace")), dfs=_int(m.get("l_df")),
            bp_faced=_int(m.get("l_bpFaced")), bp_saved=_int(m.get("l_bpSaved")),
            sv_games=_int(m.get("l_SvGms"))))
    return out


class SackmannHistory(MatchHistorySource):
    def player_matches(self, tour: str, years: list[int]) -> list[PlayerMatch]:
        repo = cfg("repos", tour) or cfg("repos", "ATP")
        prefix = "atp" if tour == "ATP" else "wta"
        out: list[PlayerMatch] = []
        for y in years:
            txt = _fetch_csv(repo, f"{prefix}_matches_{y}.csv")
            if not txt:
                continue
            rows = list(csv.DictReader(io.StringIO(txt)))
            out.extend(_rows_to_player_matches(rows))
        return out
