"""
Live ESPN adapter — free, no key. Real rosters + per-game logs + team pace for
WNBA and NBA Summer League (Las Vegas), the same public API the soccer/tennis
adapters already use. Select paths come from config (`basketball/wnba`,
`basketball/nba-summer-las-vegas`).

Honest limits: ESPN box scores give counts + minutes but not possessions, so pace
is derived from team box scores (`≈0.96·(FGA+0.44·FTA−ORB+TO)`). On the board the
matchup pace falls back to the league baseline (fast, game-level not player-level);
per-team pace is used by the backtest and available as a refinement.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import requests

from .base import GameLogSource, PlayerRef, PlayerGame, TeamPace
from ..config import league_cfg, cfg

_SITE = "https://site.api.espn.com/apis/site/v2/sports/{path}"
_WEB = "https://site.web.api.espn.com/apis/common/v3/sports/{path}"

_S = requests.Session()
_S.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json"})

_POS = {"G": "G", "F": "F", "C": "C", "PG": "G", "SG": "G", "SF": "F", "PF": "F"}

_cache: dict = {}


def _norm_name(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace(".", "").replace("'", "").split())


def _get(url: str, ttl: float = 1800) -> dict | None:
    now = time.time()
    hit = _cache.get(url)
    if hit and now - hit[0] < ttl:
        return hit[1]
    try:
        d = _S.get(url, timeout=20).json()
    except Exception:
        d = None
    _cache[url] = (now, d)
    return d


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _made_att(s: str) -> tuple[float, float]:
    """'8-17' → (8, 17)."""
    try:
        a, b = str(s).split("-")
        return _num(a), _num(b)
    except (ValueError, AttributeError):
        return 0.0, 0.0


def _possessions(fga, fta, orb, to) -> float:
    return 0.96 * (fga + 0.44 * fta - orb + to)


class EspnBasketball(GameLogSource):
    def _path(self, league: str) -> str:
        return league_cfg(league).get("espn_path", "basketball/wnba")

    # ── rosters ───────────────────────────────────────────────────────────────
    def teams(self, league: str) -> dict:
        d = _get(_SITE.format(path=self._path(league)) + "/teams", 12 * 3600)
        out: dict[str, str] = {}
        try:
            for t in d["sports"][0]["leagues"][0]["teams"]:
                out[t["team"]["displayName"]] = str(t["team"]["id"])
        except Exception:
            pass
        return out

    def team_assets(self, league: str) -> dict:
        """normalized team name/abbr → {id, abbr, logo, name} for board logos + drawer."""
        key = f"assets::{league}"
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < 12 * 3600:
            return hit[1]
        d = _get(_SITE.format(path=self._path(league)) + "/teams", 12 * 3600)
        out: dict = {}
        try:
            for t in d["sports"][0]["leagues"][0]["teams"]:
                tm = t["team"]
                logos = tm.get("logos") or []
                rec = {"id": str(tm.get("id")), "abbr": tm.get("abbreviation"),
                       "logo": (logos[0]["href"] if logos else None),
                       "name": tm.get("displayName")}
                out[_norm_name(tm.get("displayName", ""))] = rec
                if tm.get("abbreviation"):
                    out[_norm_name(tm["abbreviation"])] = rec
                if tm.get("shortDisplayName"):
                    out[_norm_name(tm["shortDisplayName"])] = rec
        except Exception:
            pass
        _cache[key] = (time.time(), out)
        return out

    def _roster(self, league: str, team_id: str, team_name: str) -> list[PlayerRef]:
        d = _get(_SITE.format(path=self._path(league)) + f"/teams/{team_id}/roster", 6 * 3600)
        out = []
        for a in (d or {}).get("athletes", []):
            for p in (a.get("items", [a]) if "items" in a else [a]):
                pid, nm = p.get("id"), p.get("displayName")
                if not pid or not nm:
                    continue
                pos = _POS.get((p.get("position") or {}).get("abbreviation", ""), "")
                out.append(PlayerRef(str(pid), nm, team_id, team_name, pos))
        return out

    def players(self, league: str) -> dict:
        key = f"players::{league}"
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < 6 * 3600:
            return hit[1]
        teams = self.teams(league)
        out: dict[str, PlayerRef] = {}
        with ThreadPoolExecutor(max_workers=10) as ex:
            rosters = ex.map(lambda kv: self._roster(league, kv[1], kv[0]), list(teams.items()))
            for refs in rosters:
                for r in refs:
                    out[_norm_name(r.name)] = r
        _cache[key] = (time.time(), out)
        return out

    # ── per-game logs ───────────────────────────────────────────────────────────
    def gamelog(self, league: str, player_id: str) -> list[PlayerGame]:
        if league_cfg(league).get("gamelog_mode") == "boxscore":
            return self._boxscore_index(league).get(str(player_id), [])
        return self._athlete_gamelog(league, player_id)

    def _completed_games(self, league: str) -> list[str]:
        from datetime import date, timedelta
        today = date.today()
        span = f"{today - timedelta(days=25):%Y%m%d}-{today + timedelta(days=2):%Y%m%d}"
        d = _get(_SITE.format(path=self._path(league)) + f"/scoreboard?dates={span}", 1800)
        out = []
        for ev in (d or {}).get("events", []):
            comp = ev.get("competitions", [{}])[0]
            if comp.get("status", {}).get("type", {}).get("state") == "post":
                out.append(ev.get("id"))
        return out

    def _parse_box(self, league: str, summ: dict) -> list[tuple]:
        """Yield (player_id, PlayerGame) for every player line in a game summary."""
        box = summ.get("boxscore", {}) or {}
        groups = box.get("players", []) or []
        date = ""
        comp = (summ.get("header", {}) or {}).get("competitions", [{}]) or [{}]
        if comp:
            date = (comp[0].get("date") or "")[:10]
        team_ref = [(str(g.get("team", {}).get("id", "")), g.get("team", {}).get("displayName", ""))
                    for g in groups]
        out = []
        for gi, grp in enumerate(groups):
            tid, tname = team_ref[gi] if gi < len(team_ref) else ("", "")
            oid, oname = team_ref[1 - gi] if len(team_ref) == 2 else ("", "")
            for sb in grp.get("statistics", []) or []:
                labels = sb.get("names") or sb.get("labels") or []
                li = {lab: i for i, lab in enumerate(labels)}
                for a in sb.get("athletes", []) or []:
                    ath = a.get("athlete", {}) or {}
                    pid, nm = str(ath.get("id", "")), ath.get("displayName", "")
                    stats = a.get("stats") or []
                    if not pid or len(stats) < len(labels):
                        continue                     # DNP / no line
                    def v(lab):
                        i = li.get(lab)
                        return stats[i] if i is not None and i < len(stats) else ""
                    mins = _num(v("MIN"))
                    if mins <= 0:
                        continue
                    fgm, fga = _made_att(v("FG"))
                    tpm, tpa = _made_att(v("3PT"))
                    ftm, fta = _made_att(v("FT"))
                    # OREB/DREB feed the offensive-SHARE estimate, so a missing column must
                    # stay 0/0 (→ the game is dropped from the split sample → the player falls
                    # back to the positional prior). Do NOT synthesise a league-average split
                    # here: it would look like real evidence and pin every player's share to
                    # the average with the full weight of their game count.
                    reb = _num(v("REB"))
                    orb = _num(v("OREB")) if "OREB" in li else 0.0
                    drb = _num(v("DREB")) if "DREB" in li else 0.0
                    out.append((pid, PlayerGame(
                        date=date, league=league, player_id=pid, player=nm,
                        team_id=tid, team=tname, opp_id=oid, opp=oname, minutes=mins,
                        pts=_num(v("PTS")), reb=reb, orb=orb, drb=drb, ast=_num(v("AST")),
                        stl=_num(v("STL")), blk=_num(v("BLK")), to=_num(v("TO")),
                        fgm=fgm, fga=fga, tpm=tpm, tpa=tpa, ftm=ftm, fta=fta,
                        pf=_num(v("PF")))))
        return out

    def _boxscore_index(self, league: str) -> dict:
        key = f"boxidx::{league}"
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < 1800:
            return hit[1]
        path = self._path(league)
        idx: dict[str, list] = {}
        for gid in self._completed_games(league):
            summ = _get(_SITE.format(path=path) + f"/summary?event={gid}", 6 * 3600)
            if not summ:
                continue
            for pid, pg in self._parse_box(league, summ):
                idx.setdefault(pid, []).append(pg)
        for pid in idx:
            idx[pid].sort(key=lambda g: g.date, reverse=True)   # most-recent-first
        _cache[key] = (time.time(), idx)
        return idx

    def _athlete_gamelog(self, league: str, player_id: str) -> list[PlayerGame]:
        d = _get(_WEB.format(path=self._path(league)) + f"/athletes/{player_id}/gamelog", 3 * 3600)
        names = (d or {}).get("names") or []
        if not names:
            return []
        ix = {n: i for i, n in enumerate(names)}
        ev_meta = (d or {}).get("events") or {}          # {eventId: {gameDate, opponent, ...}}
        games: list[PlayerGame] = []
        for stp in d.get("seasonTypes", []) or []:
            for cat in stp.get("categories", []) or []:
                for ev in cat.get("events", []) or []:
                    stats = ev.get("stats") or []
                    if len(stats) != len(names):
                        continue
                    def g(name):
                        i = ix.get(name)
                        return stats[i] if i is not None else None
                    mins = _num(g("minutes"))
                    if mins <= 0:
                        continue
                    fgm, fga = _made_att(g("fieldGoalsMade-fieldGoalsAttempted"))
                    tpm, tpa = _made_att(g("threePointFieldGoalsMade-threePointFieldGoalsAttempted"))
                    ftm, fta = _made_att(g("freeThrowsMade-freeThrowsAttempted"))
                    meta = ev_meta.get(str(ev.get("eventId")), {}) if ev_meta else {}
                    opp = ""
                    o = meta.get("opponent") or {}
                    if isinstance(o, dict):
                        opp = o.get("displayName") or o.get("abbreviation") or ""
                    games.append(PlayerGame(
                        date=(meta.get("gameDate") or "")[:10], league=league,
                        player_id=str(player_id), player="",
                        team_id="", team="", opp_id="", opp=opp,
                        minutes=mins,
                        pts=_num(g("points")), reb=_num(g("totalRebounds")),
                        ast=_num(g("assists")), stl=_num(g("steals")),
                        blk=_num(g("blocks")), to=_num(g("turnovers")),
                        fgm=fgm, fga=fga, tpm=tpm, tpa=tpa, ftm=ftm, fta=fta,
                        pf=_num(g("fouls"))))
        return games

    # ── pace ────────────────────────────────────────────────────────────────────
    def league_pace(self, league: str) -> float:
        return float(league_cfg(league).get("league_pace", 98.0))

    def team_pace(self, league: str, team_id: str) -> TeamPace | None:
        """Team pace + off/def rating from the last ~6 completed games.

        def_rtg (points allowed / 100 poss) is what the board's opponent adjustment scales
        against. Possessions come from the team's own box line; points for/against from the
        final score. (Both teams in a game see ~the same possessions, so using this team's
        possessions for the 'allowed' rate is the standard approximation.)
        """
        key = f"pace::{league}::{team_id}"
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < 12 * 3600:
            return hit[1]
        path = self._path(league)
        sched = _get(_SITE.format(path=path) + f"/teams/{team_id}/schedule", 6 * 3600)
        done = [ev for ev in (sched or {}).get("events", [])
                if ev.get("competitions", [{}])[0].get("status", {}).get("type", {}).get("state") == "post"]
        poss_sum = pf_sum = pa_sum = n = 0.0
        for ev in done[-6:]:
            gid = ev.get("id")
            summ = _get(_SITE.format(path=path) + f"/summary?event={gid}", 24 * 3600)
            comp = (((summ or {}).get("header", {}) or {}).get("competitions") or [{}])[0]
            scores = {str((c.get("team") or {}).get("id")): _num(c.get("score"))
                      for c in (comp.get("competitors") or [])}
            if len(scores) != 2 or str(team_id) not in scores:
                continue                       # can't attribute points → skip this game
            for tb in (summ or {}).get("boxscore", {}).get("teams", []):
                st = {s.get("name"): s.get("displayValue") for s in tb.get("statistics", [])}
                if str(tb.get("team", {}).get("id")) != str(team_id):
                    continue
                _, fga = _made_att(st.get("fieldGoalsMade-fieldGoalsAttempted"))
                _, fta = _made_att(st.get("freeThrowsMade-freeThrowsAttempted"))
                orb = _num(st.get("offensiveRebounds"))
                to = _num(st.get("totalTurnovers") or st.get("turnovers"))
                p = _possessions(fga, fta, orb, to)
                if p <= 0:
                    continue
                poss_sum += p
                pf_sum += scores[str(team_id)]
                pa_sum += sum(v for k, v in scores.items() if k != str(team_id))
                n += 1
        if n == 0 or poss_sum <= 0:
            return None
        tp = TeamPace(str(team_id), "", pace=round(poss_sum / n, 1),
                      off_rtg=round(100.0 * pf_sum / poss_sum, 1),
                      def_rtg=round(100.0 * pa_sum / poss_sum, 1),
                      games=int(n))
        _cache[key] = (time.time(), tp)
        return tp

    def injuries(self, league: str) -> dict:
        """{normalized_player_name: 'out' | 'questionable'} from ESPN's league injury report.

        The only reliable role signal on the free feed — ESPN does NOT publish pregame
        starting fives for the WNBA (the summary `rosters`/starter flags come back empty for
        upcoming AND past games), but the injury list is complete and current (matched by name;
        the entries carry no athlete id). 'Out' → a confirmed DNP (suppress like an MLB scratch);
        everything else present (Day-To-Day, Questionable, Doubtful, GTD) → 'questionable', which
        we flag but still project, since the player may well play."""
        d = _get(_SITE.format(path=self._path(league)) + "/injuries", 900)
        out: dict = {}
        for team in (d or {}).get("injuries", []):
            for it in team.get("injuries", []) or []:
                nm = _norm_name(((it.get("athlete") or {}).get("displayName")) or "")
                st = str(it.get("status") or "").strip().lower()
                if not nm or not st:
                    continue
                out[nm] = "out" if "out" in st else "questionable"
        return out

    def upcoming_opponents(self, league: str) -> dict:
        """{team_id: opp_team_id} for the current slate — lets the board use the REAL matchup
        pace + opponent defense instead of the league baseline. Games already final are
        included too (props are posted pre-game, but a late build shouldn't lose the map)."""
        key = f"oppmap::{league}"
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < 1800:
            return hit[1]
        from datetime import date, timedelta
        today = date.today()
        span = f"{today:%Y%m%d}-{today + timedelta(days=1):%Y%m%d}"
        d = _get(_SITE.format(path=self._path(league)) + f"/scoreboard?dates={span}", 900)
        out: dict = {}
        for ev in (d or {}).get("events", []):
            cs = (ev.get("competitions") or [{}])[0].get("competitors") or []
            if len(cs) != 2:
                continue
            a = str((cs[0].get("team") or {}).get("id") or "")
            b = str((cs[1].get("team") or {}).get("id") or "")
            if a and b:
                out.setdefault(a, b)
                out.setdefault(b, a)
        _cache[key] = (time.time(), out)
        return out

    def league_pace_avg(self, league: str) -> float | None:
        """Mean COMPUTED team pace across today's slate.

        Load-bearing: `_possessions()` yields ~80-85 for the WNBA while config's
        `league_pace` is 96 — different scales. Rates are FIT at config's league_pace, so
        feeding a raw computed pace into the sim would shrink every counting stat ~12%
        (measured: WNBA calibration +0.00 → −2.76). The board therefore applies pace as a
        RATIO against this average, which keeps the fitted scale intact.
        """
        key = f"paceavg::{league}"
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < 12 * 3600:
            return hit[1]
        vals = []
        for tid in self.upcoming_opponents(league):
            tp = self.team_pace(league, tid)
            if tp and tp.pace:
                vals.append(tp.pace)
        avg = round(sum(vals) / len(vals), 1) if len(vals) >= 4 else None
        _cache[key] = (time.time(), avg)
        return avg

    def league_def_avg(self, league: str) -> float | None:
        """Mean def rating across the teams on today's slate — the baseline the opponent
        adjustment scales against. Uses only teams we already fetch paces for, so it costs
        nothing extra; returns None (→ neutral adjustment) if the slate is too thin."""
        key = f"defavg::{league}"
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < 12 * 3600:
            return hit[1]
        vals = []
        for tid in self.upcoming_opponents(league):
            tp = self.team_pace(league, tid)
            if tp and tp.def_rtg:
                vals.append(tp.def_rtg)
        avg = round(sum(vals) / len(vals), 1) if len(vals) >= 4 else None
        _cache[key] = (time.time(), avg)
        return avg
