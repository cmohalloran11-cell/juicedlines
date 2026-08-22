"""
balldontlie.io adapter — WNBA's third data-source attempt, after two free/keyless sources
both failed in production with real evidence (not just theory):

  1. ESPN (basketball/data/espn.py) — HTTP 403 on every basketball/wnba endpoint, 2026-08.
     Two mitigations tried against ESPN itself (real-browser headers, then its v3 host)
     confirmed dead across two full refresh.yml cycles.
  2. stats.wnba.com (basketball/data/wnba_stats.py) — every single call from the GitHub
     Actions runner timed out (ReadTimeout after 20s, not even a 403 — the host never
     responded) across an unbroken hour of retry cycles. Confirmed dead, not unverified.

Both of those were unofficial/scraped APIs. balldontlie.io is a real documented product
(https://docs.balldontlie.io) with a stable REST contract and a free tier — chosen
specifically because "actively deprioritizes non-browser traffic" is a much less likely
failure mode for a paid API product than for a scraped internal endpoint.

Requires an API key: set the BALLDONTLIE_API_KEY env var (GitHub Actions secret for the
refresh.yml/static path, host env var for the live-server path — both read it the same
way, no separate wiring needed). Absent, every method below returns an honestly-empty
result (never a crash, never fabricated data) — see `_key()`.

Confidence note: the general v1 shape here (cursor-paginated `{"data": [...], "meta":
{"next_cursor": ...}}`, `/teams`, `/players`, `/stats`, `/games`) matches balldontlie's
long-documented NBA API, applied to the "wnba" path per their multi-sport rollout. This
has NOT been confirmed against a live response — this sandbox's egress proxy blocks
api.balldontlie.io exactly like it blocked ESPN and stats.wnba.com. If the shape is off,
every parse failure prints a `[basketball.balldontlie]` diagnostic (flush=True) so it
fails loud in the next refresh.yml run's logs, matching the pattern already proven twice
this session. Cross-check https://docs.balldontlie.io/wnba first if these guesses are wrong.
"""

from __future__ import annotations

import os
import time
import unicodedata
from datetime import date, timedelta

import requests

from .base import GameLogSource, PlayerRef, PlayerGame, TeamPace
from ..config import league_cfg

_BASE = "https://api.balldontlie.io/wnba/v1"

_POS = {"G": "G", "F": "F", "C": "C"}

_memory: dict = {}


def _key() -> str | None:
    return os.getenv("BALLDONTLIE_API_KEY") or None


def _norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace(".", "").replace("'", "").split())


_warned_no_key = False


def _get(path: str, params: dict | None = None, ttl: float = 900) -> dict | None:
    """None (never a crash) on: no API key configured, any HTTP/network failure, or a
    response that doesn't parse as JSON. Every failure prints a diagnostic. 'No key
    configured' is logged once per process (not per-call — every WNBA request hits this
    identically until a key is set, and refresh.yml's cycle can make hundreds of calls) so
    a genuine data-pipeline investigation ("why are WNBA recent games empty") doesn't have
    to guess between this and every other possible failure mode — see the 2026-08 live
    audit that found this specific silent case was masking the single actual cause."""
    global _warned_no_key
    key = _key()
    if not key:
        if not _warned_no_key:
            print("[basketball.balldontlie] BALLDONTLIE_API_KEY is not set -- every WNBA "
                 "request (rosters, game logs, pace, recent games) returns empty until it "
                 "is. See DEPLOY.md's env var table.", flush=True)
            _warned_no_key = True
        return None
    ck = (path, tuple(sorted((params or {}).items())))
    now = time.time()
    hit = _memory.get(ck)
    if hit and now - hit[0] < ttl:
        return hit[1]
    url = f"{_BASE}{path}"
    val = None
    try:
        resp = requests.get(url, params=params, headers={"Authorization": key}, timeout=20)
        if resp.status_code != 200:
            print(f"[basketball.balldontlie] {path} {params} -> HTTP {resp.status_code}: "
                 f"{resp.text[:200]}", flush=True)
        else:
            val = resp.json()
    except Exception as exc:
        print(f"[basketball.balldontlie] {path} {params} -> {type(exc).__name__}: {exc}",
             flush=True)
    _memory[ck] = (now, val)
    return val


def _paged(path: str, params: dict, ttl: float, max_pages: int = 20) -> list[dict]:
    """Walk balldontlie's cursor pagination (meta.next_cursor) to completion."""
    out: list[dict] = []
    cursor = None
    for _ in range(max_pages):
        p = dict(params, per_page=100)
        if cursor is not None:
            p["cursor"] = cursor
        d = _get(path, p, ttl)
        if not d or "data" not in d:
            if d is not None:
                print(f"[basketball.balldontlie] {path} response missing 'data': "
                     f"keys={list(d.keys())}", flush=True)
            break
        out.extend(d["data"])
        cursor = (d.get("meta") or {}).get("next_cursor")
        if not cursor:
            break
    return out


class BallDontLie(GameLogSource):
    # ── teams ─────────────────────────────────────────────────────────────────
    def _team_table(self, league: str) -> dict:
        key = f"teamtable::{league}"
        hit = _memory.get(key)
        if hit and time.time() - hit[0] < 12 * 3600:
            return hit[1]
        rows = _paged("/teams", {}, 12 * 3600, max_pages=3)
        out: dict = {}
        for t in rows:
            tid = t.get("id")
            name = t.get("full_name") or t.get("name")
            if tid is None or not name:
                continue
            out[str(tid)] = {"id": str(tid), "abbr": t.get("abbreviation"), "name": name,
                             "city": t.get("city")}
        if not rows:
            print(f"[basketball.balldontlie] /teams returned nothing for {league!r} "
                 f"(no API key configured, or the request failed — see the diagnostic "
                 f"above if BALLDONTLIE_API_KEY is set)", flush=True)
        _memory[key] = (time.time(), out)
        return out

    def teams(self, league: str) -> dict:
        return {t["name"]: tid for tid, t in self._team_table(league).items()}

    def team_assets(self, league: str) -> dict:
        """No logo field in balldontlie's /teams payload (confirmed absent from their docs,
        not just unchecked) — team crests stay unavailable until a real image source is
        wired for them specifically. Never guess a CDN path with zero basis, unlike the two
        best-effort constructions in the now-dead wnba_stats.py adapter."""
        out: dict = {}
        for tid, t in self._team_table(league).items():
            rec = {"id": tid, "abbr": t["abbr"], "logo": None, "name": t["name"]}
            out[_norm_name(t["name"])] = rec
            if t["abbr"]:
                out[_norm_name(t["abbr"])] = rec
            if t.get("city"):
                out.setdefault(_norm_name(t["city"]), rec)
        return out

    # ── rosters ───────────────────────────────────────────────────────────────
    def players(self, league: str) -> dict:
        key = f"players::{league}"
        hit = _memory.get(key)
        if hit and time.time() - hit[0] < 6 * 3600:
            return hit[1]
        rows = _paged("/players", {}, 6 * 3600, max_pages=10)
        out: dict[str, PlayerRef] = {}
        for p in rows:
            pid = p.get("id")
            nm = " ".join(x for x in (p.get("first_name"), p.get("last_name")) if x).strip()
            if pid is None or not nm:
                continue
            team = p.get("team") or {}
            pos = _POS.get((p.get("position") or "")[:1].upper(), "")
            out[_norm_name(nm)] = PlayerRef(str(pid), nm, str(team.get("id") or ""),
                                            team.get("full_name") or team.get("name") or "", pos)
        _memory[key] = (time.time(), out)
        return out

    def all_headshots(self, league: str) -> dict:
        """No headshot field in balldontlie's /players payload either — same honest-gap
        reasoning as team_assets' logo above. Callers already treat a missing headshot as
        'keep the book's own image', never as a blocker."""
        return {}

    # ── per-game logs ───────────────────────────────────────────────────────────
    def gamelog(self, league: str, player_id: str) -> list[PlayerGame]:
        season = int(league_cfg(league).get("stats_season") or date.today().year)
        rows = _paged("/stats", {"player_ids[]": player_id, "seasons[]": season}, 1800)
        games = []
        for row in rows:
            mins = _minutes(row.get("min"))
            if mins <= 0:
                continue
            g = row.get("game") or {}
            team = row.get("team") or {}
            games.append(PlayerGame(
                date=str(g.get("date") or "")[:10], league=league,
                player_id=str(player_id), player="", team_id=str(team.get("id") or ""),
                team="", opp_id="", opp="", minutes=mins,
                pts=float(row.get("pts") or 0), reb=float(row.get("reb") or 0),
                orb=float(row.get("oreb") or 0), drb=float(row.get("dreb") or 0),
                ast=float(row.get("ast") or 0), stl=float(row.get("stl") or 0),
                blk=float(row.get("blk") or 0), to=float(row.get("turnover") or 0),
                fgm=float(row.get("fgm") or 0), fga=float(row.get("fga") or 0),
                tpm=float(row.get("fg3m") or 0), tpa=float(row.get("fg3a") or 0),
                ftm=float(row.get("ftm") or 0), fta=float(row.get("fta") or 0),
                pf=float(row.get("pf") or 0)))
        games.sort(key=lambda g: g.date, reverse=True)
        return games

    # ── pace ────────────────────────────────────────────────────────────────────
    def league_pace(self, league: str) -> float:
        return float(league_cfg(league).get("league_pace", 96.0))

    def team_pace(self, league: str, team_id: str) -> TeamPace | None:
        """No team-level box-score endpoint in balldontlie's WNBA API (confirmed absent from
        their docs) — team totals are the SUM of that team's players' /stats rows for each
        game, same aggregation basketball/data/espn.py's team_pace did off ESPN box scores.
        Opponent points come from /games' own home/visitor score, matched by game id."""
        key = f"pace::{league}::{team_id}"
        hit = _memory.get(key)
        if hit and time.time() - hit[0] < 12 * 3600:
            return hit[1]
        season = int(league_cfg(league).get("stats_season") or date.today().year)
        rows = _paged("/stats", {"team_ids[]": team_id, "seasons[]": season}, 6 * 3600)
        by_game: dict[str, dict] = {}
        for row in rows:
            g = row.get("game") or {}
            gid = g.get("id")
            if gid is None:
                continue
            acc = by_game.setdefault(str(gid), {"date": str(g.get("date") or ""),
                                                "fga": 0.0, "fta": 0.0, "oreb": 0.0,
                                                "tov": 0.0, "pts": 0.0,
                                                "home_id": g.get("home_team_id"),
                                                "visitor_id": g.get("visitor_team_id"),
                                                "home_score": g.get("home_team_score"),
                                                "visitor_score": g.get("visitor_team_score")})
            acc["fga"] += float(row.get("fga") or 0)
            acc["fta"] += float(row.get("fta") or 0)
            acc["oreb"] += float(row.get("oreb") or 0)
            acc["tov"] += float(row.get("turnover") or 0)
            acc["pts"] += float(row.get("pts") or 0)
        ordered = sorted(by_game.values(), key=lambda a: a["date"], reverse=True)
        poss_sum = pf_sum = pa_sum = n = 0.0
        for acc in ordered[:6]:
            p = 0.96 * (acc["fga"] + 0.44 * acc["fta"] - acc["oreb"] + acc["tov"])
            if p <= 0:
                continue
            is_home = str(acc["home_id"]) == str(team_id)
            allowed = acc["visitor_score"] if is_home else acc["home_score"]
            if allowed is None:
                continue
            poss_sum += p
            pf_sum += acc["pts"]
            pa_sum += float(allowed)
            n += 1
        if n == 0 or poss_sum <= 0:
            return None
        tp = TeamPace(str(team_id), "", pace=round(poss_sum / n, 1),
                     off_rtg=round(100.0 * pf_sum / poss_sum, 1),
                     def_rtg=round(100.0 * pa_sum / poss_sum, 1), games=int(n))
        _memory[key] = (time.time(), tp)
        return tp

    def injuries(self, league: str) -> dict:
        """No injury endpoint in balldontlie's free-tier WNBA API — honest empty result,
        same reasoning as wnba_stats.py's injuries(). board.py already treats this as 'no
        signal' (projects through it) rather than suppressing on a fabricated status."""
        return {}

    def upcoming_opponents(self, league: str) -> dict:
        key = f"oppmap::{league}"
        hit = _memory.get(key)
        if hit and time.time() - hit[0] < 1800:
            return hit[1]
        today = date.today()
        rows = _paged("/games", {"dates[]": today.isoformat()}, 900)
        if not rows:   # a slate can genuinely be empty; try tomorrow before giving up
            rows = _paged("/games", {"dates[]": (today + timedelta(days=1)).isoformat()}, 900)
        out: dict = {}
        for g in rows:
            home, away = g.get("home_team_id"), g.get("visitor_team_id")
            if home is not None and away is not None:
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


def _minutes(raw) -> float:
    """balldontlie's 'min' field has historically been a 'MM' or 'MM:SS' string (unlike
    stats.wnba.com's plain numeric minutes) — handle both rather than assume one."""
    if raw is None:
        return 0.0
    s = str(raw).strip()
    if not s:
        return 0.0
    if ":" in s:
        try:
            m, sec = s.split(":")
            return float(m) + float(sec) / 60.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0
