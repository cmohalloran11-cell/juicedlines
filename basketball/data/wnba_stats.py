"""
Live stats.wnba.com + cdn.wnba.com adapter — replaces ESPN as WNBA's roster/game-log/pace
source.

Why: ESPN started returning HTTP 403 on every basketball/wnba endpoint (2026-08 — see
basketball/data/espn.py's own diagnosis comment). Two mitigations against ESPN itself (a
real-browser header set, then falling back to ESPN's v3 host) were tried and neither
recovered live data (confirmed against two full refresh.yml cycles). This is a different
provider, not a third ESPN mitigation.

stats.wnba.com is the WNBA's own official stats API. It runs the same platform as
stats.nba.com — same JSON envelope (`{"resultSets": [{"headers": [...], "rowSet":
[[...]]}]}`), same endpoint family (commonteamroster, playergamelog, teamgamelog,
scoreboardv2), just LeagueID "10" instead of "00". That envelope shape and those endpoint
names are HIGH confidence — stats.nba.com is one of the most heavily scraped public sports
APIs there is, and stats.wnba.com is well documented as mirroring it exactly.

What is NOT independently verified: this sandbox's egress proxy blocks stats.wnba.com and
cdn.wnba.com exactly as it blocked ESPN, so nothing here has been confirmed against a real
live response. Two things carry real uncertainty: (1) whether stats.wnba.com enforces the
same bot-detection stats.nba.com is known to (the x-nba-stats-* headers below are the
documented workaround for THAT host — unconfirmed here), and (2) the exact cdn.wnba.com
logo asset path (headshots and the live scoreboard feed are better-known patterns). Every
parse failure prints a `[basketball.wnbastats]`-prefixed diagnostic (flush=True, matching
the fix already shipped for nfl/data/espn.py's silent-failure bug) so a wrong guess fails
loud in the next refresh.yml run's logs instead of silently going to zero again.

A real, incidental improvement over the ESPN adapter: playergamelog carries OREB/DREB
per game directly, so the orb/drb split no longer needs ESPN's separate, more expensive
box-score-index fallback (see basketball/projections.py's `split_games` — it now gets
what it needs straight from `gamelog()`). And requesting SeasonType="Regular Season"
excludes the All-Star Game / playoffs at the source, so the exhibition-game filter
basketball/data/espn.py needed (see its own module + tests) isn't needed here either.
"""

from __future__ import annotations

import os
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import requests

from .base import GameLogSource, PlayerRef, PlayerGame, TeamPace
from ..config import league_cfg, cfg

_STATS = "https://stats.wnba.com/stats"
_CDN = "https://cdn.wnba.com"

_LEAGUE_ID = {"WNBA": "10"}

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Connection": "keep-alive",
    "Referer": "https://www.wnba.com/",
    "Origin": "https://www.wnba.com",
}

# Real, stable public facts (franchise identities), not measured/fabricated data — generously
# aliased because the exact abbreviation stats.wnba.com's MATCHUP field uses, and the exact
# abbreviation a book sends on a line, are both unconfirmed from this sandbox. `players()`
# sets PlayerRef.team from this table's own canonical `name`, so the primary lookup path
# (analytics.enrich_lines resolving a player then reading THIS SAME table) never actually
# depends on the aliasing below — it only protects the book-abbreviation fallback path.
_TEAMS = [
    {"name": "Atlanta Dream", "aliases": ("atl", "atlanta")},
    {"name": "Chicago Sky", "aliases": ("chi", "chicago")},
    {"name": "Connecticut Sun", "aliases": ("conn", "connecticut", "ct")},
    {"name": "Dallas Wings", "aliases": ("dal", "dallas")},
    {"name": "Golden State Valkyries", "aliases": ("gs", "gsv", "golden state", "valkyries")},
    {"name": "Indiana Fever", "aliases": ("ind", "indiana")},
    {"name": "Las Vegas Aces", "aliases": ("lv", "lva", "las vegas")},
    {"name": "Los Angeles Sparks", "aliases": ("la", "las", "los angeles")},
    {"name": "Minnesota Lynx", "aliases": ("min", "minnesota")},
    {"name": "New York Liberty", "aliases": ("ny", "nyl", "new york")},
    {"name": "Phoenix Mercury", "aliases": ("phx", "pho", "phoenix")},
    {"name": "Seattle Storm", "aliases": ("sea", "seattle")},
    {"name": "Washington Mystics", "aliases": ("wsh", "was", "washington")},
]

_POS = {"G": "G", "F": "F", "C": "C"}

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "wnba_cache"
_memory: dict = {}


def _cache_dir() -> Path:
    return Path(os.getenv("WNBA_CACHE_DIR", str(_DEFAULT_CACHE_DIR)))


def _norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace(".", "").replace("'", "").split())


def _season(league: str) -> str:
    override = league_cfg(league).get("stats_season")
    return str(override) if override else str(date.today().year)


def _get(url: str, ttl: float = 900) -> dict | None:
    """Memory + disk cached (survives a fresh `python build_static.py` process — every
    refresh.yml cycle is one — matching nfl/data/cache.py's fetch_text pattern and the
    exact lesson already learned from nfl/data/espn.py's own in-process-only-cache bug)."""
    now = time.time()
    hit = _memory.get(url)
    if hit and now - hit[0] < ttl:
        return hit[1]
    d = _cache_dir()
    path = d / (url.replace("://", "_").replace("/", "_").replace("?", "_")[:200] + ".json")
    if path.exists() and now - path.stat().st_mtime < ttl:
        try:
            import json
            d_val = json.loads(path.read_text(encoding="utf-8"))
            _memory[url] = (now, d_val)
            return d_val
        except Exception:
            pass
    val = None
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        if resp.status_code != 200:
            print(f"[basketball.wnbastats] {url} -> HTTP {resp.status_code}", flush=True)
        else:
            val = resp.json()
    except Exception as exc:
        print(f"[basketball.wnbastats] {url} -> {type(exc).__name__}: {exc}", flush=True)
    _memory[url] = (now, val)
    if val is not None:
        try:
            import json
            d.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(val), encoding="utf-8")
        except OSError:
            pass
    return val


def _rows(d: dict | None, index: int = 0) -> tuple[list[str], list[list]]:
    """One resultSet's (headers, rowSet) from the standard stats.{nba,wnba}.com envelope."""
    try:
        rs = d["resultSets"][index]
        return rs["headers"], rs["rowSet"]
    except Exception as exc:
        print(f"[basketball.wnbastats] resultSets[{index}] parse failed on "
              f"{type(d)}: {type(exc).__name__}: {exc}", flush=True)
        return [], []


def _dictrows(d: dict | None, index: int = 0) -> list[dict]:
    headers, rows = _rows(d, index)
    return [dict(zip(headers, r)) for r in rows]


class WnbaStats(GameLogSource):
    # ── teams ─────────────────────────────────────────────────────────────────
    def _team_table(self, league: str) -> dict:
        """team_id (str) -> {"id", "abbr", "name", "logo"}. TEAM_ID comes straight off
        leaguedashteamstats — the id used consistently by every other call in this module
        (roster, gamelog, pace, scoreboard), so there is no cross-source id mismatch."""
        key = f"teamtable::{league}"
        hit = _memory.get(key)
        if hit and time.time() - hit[0] < 12 * 3600:
            return hit[1]
        lid = _LEAGUE_ID.get(league, "10")
        d = _get(f"{_STATS}/leaguedashteamstats?LeagueID={lid}&Season={_season(league)}"
                 f"&SeasonType=Regular+Season&PerMode=Totals&MeasureType=Base", 12 * 3600)
        out: dict = {}
        for row in _dictrows(d):
            tid, tname = str(row.get("TEAM_ID") or ""), row.get("TEAM_NAME") or ""
            if not tid or not tname:
                continue
            static = next((t for t in _TEAMS if t["name"].lower() in tname.lower()
                          or tname.lower() in t["name"].lower()), None)
            abbr = None
            if static:
                # shortest alias that looks like a real code (2-3 letters) reads best on a chip
                abbr = next((a.upper() for a in sorted(static["aliases"], key=len)
                            if 2 <= len(a) <= 3), None)
            out[tid] = {"id": tid, "abbr": abbr, "name": tname,
                       # Best-effort constructed CDN path — unverified (see module docstring).
                       # Degrades to a broken-image icon exactly like a missing logo already
                       # does (static/dashboard.html's teamBadge has onerror handling), never
                       # worse than not trying.
                       "logo": f"{_CDN}/logos/wnba/{tid}/global/L/logo.svg"}
        if not out:
            print(f"[basketball.wnbastats] leaguedashteamstats returned no usable rows for "
                  f"{league!r}", flush=True)
        _memory[key] = (time.time(), out)
        return out

    def teams(self, league: str) -> dict:
        return {t["name"]: tid for tid, t in self._team_table(league).items()}

    def team_assets(self, league: str) -> dict:
        out: dict = {}
        for tid, t in self._team_table(league).items():
            rec = {"id": tid, "abbr": t["abbr"], "logo": t["logo"], "name": t["name"]}
            out[_norm_name(t["name"])] = rec
            if t["abbr"]:
                out[_norm_name(t["abbr"])] = rec
        for static in _TEAMS:
            match = next((r for r in out.values() if r["name"] == static["name"]), None)
            if match:
                for alias in static["aliases"]:
                    out.setdefault(_norm_name(alias), match)
        return out

    # ── rosters ───────────────────────────────────────────────────────────────
    def _roster(self, league: str, team_id: str, team_name: str) -> list[PlayerRef]:
        lid = _LEAGUE_ID.get(league, "10")
        d = _get(f"{_STATS}/commonteamroster?TeamID={team_id}&Season={_season(league)}"
                 f"&LeagueID={lid}", 6 * 3600)
        out = []
        for row in _dictrows(d):
            pid, nm = row.get("PLAYER_ID"), row.get("PLAYER")
            if not pid or not nm:
                continue
            pos = _POS.get((row.get("POSITION") or "")[:1].upper(), "")
            out.append(PlayerRef(str(pid), nm, team_id, team_name, pos))
        return out

    def players(self, league: str) -> dict:
        key = f"players::{league}"
        hit = _memory.get(key)
        if hit and time.time() - hit[0] < 6 * 3600:
            return hit[1]
        teams = self._team_table(league)
        out: dict[str, PlayerRef] = {}
        with ThreadPoolExecutor(max_workers=10) as ex:
            rosters = ex.map(lambda kv: self._roster(league, kv[0], kv[1]["name"]),
                             list(teams.items()))
            for refs in rosters:
                for r in refs:
                    out[_norm_name(r.name)] = r
        _memory[key] = (time.time(), out)
        return out

    def all_headshots(self, league: str) -> dict:
        """normalized player name -> cdn.wnba.com headshot url. Same 'best-effort constructed,
        degrades to broken-image, never blocks a projection' contract as team logos above."""
        return {name: f"{_CDN}/headshots/wnba/latest/1040x760/{ref.id}.png"
               for name, ref in self.players(league).items()}

    # ── per-game logs ───────────────────────────────────────────────────────────
    def gamelog(self, league: str, player_id: str) -> list[PlayerGame]:
        lid = _LEAGUE_ID.get(league, "10")
        d = _get(f"{_STATS}/playergamelog?PlayerID={player_id}&Season={_season(league)}"
                 f"&SeasonType=Regular+Season&LeagueID={lid}", 1800)
        games = []
        for row in _dictrows(d):
            mins = float(row.get("MIN") or 0)
            if mins <= 0:
                continue
            matchup = str(row.get("MATCHUP") or "")
            opp = matchup.split()[-1] if matchup else ""
            games.append(PlayerGame(
                date=str(row.get("GAME_DATE") or "")[:10], league=league,
                player_id=str(player_id), player="", team_id="", team="",
                opp_id="", opp=opp, minutes=mins,
                pts=float(row.get("PTS") or 0), reb=float(row.get("REB") or 0),
                orb=float(row.get("OREB") or 0), drb=float(row.get("DREB") or 0),
                ast=float(row.get("AST") or 0), stl=float(row.get("STL") or 0),
                blk=float(row.get("BLK") or 0), to=float(row.get("TOV") or 0),
                fgm=float(row.get("FGM") or 0), fga=float(row.get("FGA") or 0),
                tpm=float(row.get("FG3M") or 0), tpa=float(row.get("FG3A") or 0),
                ftm=float(row.get("FTM") or 0), fta=float(row.get("FTA") or 0),
                pf=float(row.get("PF") or 0)))
        games.sort(key=lambda g: g.date, reverse=True)
        return games

    # ── pace ────────────────────────────────────────────────────────────────────
    def league_pace(self, league: str) -> float:
        return float(league_cfg(league).get("league_pace", 96.0))

    def team_pace(self, league: str, team_id: str) -> TeamPace | None:
        """Team pace + off/def rating from the last 6 completed games. teamgamelog's own
        PLUS_MINUS gives opponent points for free (PTS - PLUS_MINUS = points allowed) —
        no separate boxscore-per-game fetch needed, unlike the ESPN adapter this replaces."""
        key = f"pace::{league}::{team_id}"
        hit = _memory.get(key)
        if hit and time.time() - hit[0] < 12 * 3600:
            return hit[1]
        lid = _LEAGUE_ID.get(league, "10")
        d = _get(f"{_STATS}/teamgamelog?TeamID={team_id}&Season={_season(league)}"
                 f"&SeasonType=Regular+Season&LeagueID={lid}", 6 * 3600)
        rows = _dictrows(d)
        rows.sort(key=lambda r: str(r.get("GAME_DATE") or ""), reverse=True)
        poss_sum = pf_sum = pa_sum = n = 0.0
        for row in rows[:6]:
            fga, fta = float(row.get("FGA") or 0), float(row.get("FTA") or 0)
            orb, to = float(row.get("OREB") or 0), float(row.get("TOV") or 0)
            p = 0.96 * (fga + 0.44 * fta - orb + to)
            if p <= 0:
                continue
            pts = float(row.get("PTS") or 0)
            plus_minus = float(row.get("PLUS_MINUS") or 0)
            poss_sum += p
            pf_sum += pts
            pa_sum += pts - plus_minus
            n += 1
        if n == 0 or poss_sum <= 0:
            return None
        tp = TeamPace(str(team_id), "", pace=round(poss_sum / n, 1),
                     off_rtg=round(100.0 * pf_sum / poss_sum, 1),
                     def_rtg=round(100.0 * pa_sum / poss_sum, 1), games=int(n))
        _memory[key] = (time.time(), tp)
        return tp

    def injuries(self, league: str) -> dict:
        """No free WNBA injury JSON feed identified for stats.wnba.com/cdn.wnba.com (unlike
        ESPN's /injuries, which existed but is blocked). Honest empty result rather than a
        guessed URL with zero confidence behind it — board.py already treats a missing/failed
        injuries() call as 'no signal', which degrades to projecting through it instead of
        suppressing, never to fabricating a status."""
        return {}

    def upcoming_opponents(self, league: str) -> dict:
        key = f"oppmap::{league}"
        hit = _memory.get(key)
        if hit and time.time() - hit[0] < 1800:
            return hit[1]
        lid = _LEAGUE_ID.get(league, "10")
        today = date.today().strftime("%m/%d/%Y")
        d = _get(f"{_STATS}/scoreboardv2?GameDate={today}&LeagueID={lid}&DayOffset=0", 900)
        out: dict = {}
        for row in _dictrows(d, 0):    # GameHeader is resultSets[0]
            home, away = row.get("HOME_TEAM_ID"), row.get("VISITOR_TEAM_ID")
            if home and away:
                out.setdefault(str(home), str(away))
                out.setdefault(str(away), str(home))
        _memory[key] = (time.time(), out)
        return out

    def league_pace_avg(self, league: str) -> float | None:
        key = f"paceavg::{league}"
        hit = _memory.get(key)
        if hit and time.time() - hit[0] < 12 * 3600:
            return hit[1]
        vals = [tp.pace for tid in self.upcoming_opponents(league)
               if (tp := self.team_pace(league, tid)) and tp.pace]
        avg = round(sum(vals) / len(vals), 1) if len(vals) >= 4 else None
        _memory[key] = (time.time(), avg)
        return avg

    def league_def_avg(self, league: str) -> float | None:
        key = f"defavg::{league}"
        hit = _memory.get(key)
        if hit and time.time() - hit[0] < 12 * 3600:
            return hit[1]
        vals = [tp.def_rtg for tid in self.upcoming_opponents(league)
               if (tp := self.team_pace(league, tid)) and tp.def_rtg]
        avg = round(sum(vals) / len(vals), 1) if len(vals) >= 4 else None
        _memory[key] = (time.time(), avg)
        return avg
