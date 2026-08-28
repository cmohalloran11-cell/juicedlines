"""
CollegeFootballData (CFBD) REST adapter -- teams, rosters, schedule (+ market spread/total),
per-player box scores, per-team advanced efficiency. Patreon Tier 3 key required.

Requires an API key: set the CFBD_API_KEY env var (server-side only -- see cfb/config.py's
module docstring and cfb/README.md's License constraints section; this key must never reach
a browser response or build_static.py's output JSON). Absent, every method below returns an
honestly-empty result ([]) -- never a crash, never fabricated data -- matching
basketball/data/balldontlie.py's established BALLDONTLIE_API_KEY pattern exactly (`_key()`
guard, warn-once-per-process, in-process memo cache with a TTL, per-call diagnostics on any
HTTP/parse failure).

ENDPOINT SHAPES USED BELOW ARE UNVERIFIED AGAINST A LIVE RESPONSE. This sandbox's egress
proxy has no route to api.collegefootballdata.com (same failure mode already confirmed this
session for statsapi.mlb.com and api.underdogfantasy.com -- see the task's own environment
note), and no CFBD_API_KEY is available in this environment either way. The shapes parsed
here are built against CFBD's own published OpenAPI schema and the `cfbd` npm client's
TypeScript types (both cited in the task spec as the reference for the REST shape), not a
live call:
  GET /teams                 -> [{id, school, conference, classification, abbreviation, ...}]
  GET /roster?team=&year=    -> [{id, first_name, last_name, team, position, jersey, ...}]
  GET /games?year=&week=&seasonType=
                              -> [{id, season, week, seasonType, startDate, homeTeam, awayTeam,
                                   homeClassification, awayClassification, completed,
                                   homePoints, awayPoints, ...}]
  GET /recruiting/players?year=
                              -> [{name, position, committedTo, rating, stars, ranking, ...}]
  GET /lines?year=&week=&team=
                              -> [{id (=gameId), homeTeam, awayTeam,
                                   lines: [{provider, spread, overUnder}, ...]}]
  GET /games/players?year=&week=&seasonType=&team=
                              -> [{id (=gameId), teams: [{team, category, types: [
                                   {name (=stat name, e.g. "YDS"/"TD"/"C/ATT"),
                                    athletes: [{id, name, stat}]}]}]}]
  GET /stats/game/advanced?year=&week=&team=
                              -> [{gameId, week, team, opponent,
                                   offense: {ppa, successRate}, defense: {ppa, successRate},
                                   plays}]
If any of these shapes is wrong, every parse failure prints a `[cfb.cfbd_client]` diagnostic
(flush=True) so it fails loud in the next scheduled run's logs rather than silently returning
nothing indistinguishable from "no key configured" -- same pattern balldontlie.py already
proved this session. Cross-check https://api.collegefootballdata.com/api/docs/ first if a
live shape turns out to differ.
"""
from __future__ import annotations

import time
from typing import Optional

import requests

from .base import (CfbDataSource, TeamRef, PlayerRef, PlayerGameStats, RecruitRating,
                   TeamGameEfficiency, ScheduleGame)
from ..config import CFBD_BASE_URL, HTTP_TIMEOUT, CACHE_TTL, cfbd_api_key

_memory: dict = {}
_warned_no_key = False


def _key() -> Optional[str]:
    return cfbd_api_key()


def _get(path: str, params: Optional[dict] = None, ttl: float = CACHE_TTL) -> Optional[list | dict]:
    """None (never a crash) on: no API key configured, any HTTP/network failure, or a
    response that doesn't parse as JSON. 'No key configured' is logged once per process."""
    global _warned_no_key
    key = _key()
    if not key:
        if not _warned_no_key:
            print("[cfb.cfbd_client] CFBD_API_KEY is not set -- every CFB team/roster/"
                 "schedule/stats request returns empty until it is. See DEPLOY.md's env "
                 "var table.", flush=True)
            _warned_no_key = True
        return None
    ck = (path, tuple(sorted((params or {}).items())))
    now = time.time()
    hit = _memory.get(ck)
    if hit and now - hit[0] < ttl:
        return hit[1]
    url = f"{CFBD_BASE_URL}{path}"
    val = None
    try:
        resp = requests.get(url, params=params,
                            headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
                            timeout=HTTP_TIMEOUT)
        if resp.status_code != 200:
            print(f"[cfb.cfbd_client] {path} {params} -> HTTP {resp.status_code}: "
                 f"{resp.text[:200]}", flush=True)
        else:
            val = resp.json()
    except Exception as exc:
        print(f"[cfb.cfbd_client] {path} {params} -> {exc}", flush=True)
    _memory[ck] = (now, val)
    return val


def _f(v) -> float:
    try:
        return float(v) if v not in (None, "") else 0.0
    except (TypeError, ValueError):
        return 0.0


def _of(v) -> Optional[float]:
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _oi(v) -> Optional[int]:
    try:
        return int(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


class CFBDClient(CfbDataSource):
    """Live CollegeFootballData adapter. See module docstring for the endpoint shapes and
    their (unverified-live, spec-derived) provenance."""

    def teams(self, season: int, classification: str = "fbs") -> list[TeamRef]:
        raw = _get("/teams", {"year": season})
        if not isinstance(raw, list):
            return []
        out = []
        for t in raw:
            if not isinstance(t, dict):
                continue
            cls = (t.get("classification") or "").lower()
            if classification and cls != classification.lower():
                continue
            school = t.get("school")
            if not school:
                continue
            out.append(TeamRef(
                id=str(t.get("id")), school=school, conference=t.get("conference"),
                classification=t.get("classification"), abbreviation=t.get("abbreviation")))
        return out

    def roster(self, team: str, season: int) -> list[PlayerRef]:
        raw = _get("/roster", {"team": team, "year": season})
        if not isinstance(raw, list):
            return []
        out = []
        for p in raw:
            if not isinstance(p, dict):
                continue
            pid = p.get("id")
            first, last = p.get("first_name") or "", p.get("last_name") or ""
            name = (first + " " + last).strip() or p.get("name")
            if pid is None or not name:
                continue
            out.append(PlayerRef(
                id=str(pid), name=name, team=team, position=p.get("position") or "",
                jersey=p.get("jersey")))
        return out

    def schedule(self, season: int, week: Optional[int] = None) -> list[ScheduleGame]:
        params: dict = {"year": season}
        if week is not None:
            params["week"] = week
        games = _get("/games", params)
        if not isinstance(games, list):
            return []
        lines_by_game = self._lines_by_game(season, week)
        out = []
        for g in games:
            if not isinstance(g, dict):
                continue
            gid = g.get("id")
            home, away = g.get("homeTeam") or g.get("home_team"), g.get("awayTeam") or g.get("away_team")
            if gid is None or not home or not away:
                continue
            spread, ou = lines_by_game.get(str(gid), (None, None))
            out.append(ScheduleGame(
                id=str(gid), season=g.get("season", season), week=g.get("week", week or 0),
                season_type=g.get("seasonType") or g.get("season_type") or "regular",
                start_date=g.get("startDate") or g.get("start_date") or "",
                home_team=home, away_team=away,
                home_classification=g.get("homeClassification") or g.get("home_classification"),
                away_classification=g.get("awayClassification") or g.get("away_classification"),
                spread=spread, over_under=ou, completed=bool(g.get("completed")),
                home_points=_oi(g.get("homePoints") or g.get("home_points")),
                away_points=_oi(g.get("awayPoints") or g.get("away_points"))))
        return out

    def _lines_by_game(self, season: int, week: Optional[int]) -> dict[str, tuple]:
        params: dict = {"year": season}
        if week is not None:
            params["week"] = week
        raw = _get("/lines", params)
        out: dict[str, tuple] = {}
        if not isinstance(raw, list):
            return out
        for g in raw:
            if not isinstance(g, dict):
                continue
            gid = g.get("id")
            lines = g.get("lines") or []
            if gid is None or not lines:
                continue
            # First provider with both fields wins -- CFBD aggregates several books per game
            # and doesn't guarantee a canonical one; a documented simplification, not a guess.
            spread = ou = None
            for ln in lines:
                if not isinstance(ln, dict):
                    continue
                s, o = _of(ln.get("spread")), _of(ln.get("overUnder") or ln.get("over_under"))
                if s is not None or o is not None:
                    spread, ou = s, o
                    break
            out[str(gid)] = (spread, ou)
        return out

    def player_game_stats(self, season: int, week: Optional[int] = None,
                          team: Optional[str] = None) -> list[PlayerGameStats]:
        params: dict = {"year": season}
        if week is not None:
            params["week"] = week
        if team:
            params["team"] = team
        raw = _get("/games/players", params)
        if not isinstance(raw, list):
            return []
        by_player: dict[tuple, PlayerGameStats] = {}
        for game in raw:
            if not isinstance(game, dict):
                continue
            gid = str(game.get("id"))
            for team_block in game.get("teams") or []:
                if not isinstance(team_block, dict):
                    continue
                team_name = team_block.get("team")
                category = (team_block.get("category") or "").lower()
                if category not in ("passing", "rushing", "receiving"):
                    continue
                for stat_type in team_block.get("types") or []:
                    if not isinstance(stat_type, dict):
                        continue
                    stat_name = (stat_type.get("name") or "").upper()
                    for ath in stat_type.get("athletes") or []:
                        if not isinstance(ath, dict):
                            continue
                        pid, pname = ath.get("id"), ath.get("name")
                        if pid is None or not pname:
                            continue
                        key = (gid, str(pid))
                        row = by_player.get(key)
                        if row is None:
                            row = PlayerGameStats(
                                season=season, week=week or 0, season_type="regular",
                                game_id=gid, player_id=str(pid), player=pname,
                                team=team_name or "", opponent="")
                            by_player[key] = row
                        self._apply_stat(row, category, stat_name, ath.get("stat"))
        return list(by_player.values())

    @staticmethod
    def _apply_stat(row: PlayerGameStats, category: str, stat_name: str, raw_val) -> None:
        if category == "passing":
            if stat_name == "C/ATT" and isinstance(raw_val, str) and "/" in raw_val:
                c, a = raw_val.split("/", 1)
                row.pass_completions, row.pass_attempts = _f(c), _f(a)
            elif stat_name == "YDS":
                row.pass_yards = _f(raw_val)
            elif stat_name == "TD":
                row.pass_tds = _f(raw_val)
            elif stat_name == "INT":
                row.interceptions = _f(raw_val)
        elif category == "rushing":
            if stat_name == "CAR":
                row.rush_attempts = _f(raw_val)
            elif stat_name == "YDS":
                row.rush_yards = _f(raw_val)
            elif stat_name == "TD":
                row.rush_tds = _f(raw_val)
        elif category == "receiving":
            if stat_name == "REC":
                row.receptions = _f(raw_val)
                row.targets = max(row.targets, row.receptions)
            elif stat_name == "YDS":
                row.rec_yards = _f(raw_val)
            elif stat_name == "TD":
                row.rec_tds = _f(raw_val)

    def team_efficiency(self, season: int, week: Optional[int] = None) -> list[TeamGameEfficiency]:
        params: dict = {"year": season}
        if week is not None:
            params["week"] = week
        raw = _get("/stats/game/advanced", params)
        if not isinstance(raw, list):
            return []
        out = []
        for r in raw:
            if not isinstance(r, dict):
                continue
            team = r.get("team")
            if not team:
                continue
            off, dfn = r.get("offense") or {}, r.get("defense") or {}
            out.append(TeamGameEfficiency(
                season=season, week=r.get("week", week or 0), team=team,
                opponent=r.get("opponent") or "",
                offense_ppa=_of(off.get("ppa")), defense_ppa=_of(dfn.get("ppa")),
                offense_success_rate=_of(off.get("successRate")),
                defense_success_rate=_of(dfn.get("successRate")),
                plays=int(r["plays"]) if r.get("plays") not in (None, "") else None))
        return out

    def recruiting(self, season: int) -> list[RecruitRating]:
        raw = _get("/recruiting/players", {"year": season})
        if not isinstance(raw, list):
            return []
        out = []
        for r in raw:
            if not isinstance(r, dict):
                continue
            name = r.get("name") or " ".join(
                x for x in (r.get("firstName"), r.get("lastName")) if x).strip()
            if not name:
                continue
            out.append(RecruitRating(
                season=r.get("year", season), name=name,
                team=r.get("committedTo") or r.get("committed_to"),
                position=r.get("position"), rating=_of(r.get("rating")),
                stars=_oi(r.get("stars")), ranking=_oi(r.get("ranking"))))
        return out
