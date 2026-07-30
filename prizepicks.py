"""
prizepicks.py — a lightweight read-only client for PrizePicks projections.

Reads ALL available props (projections) and their lines from PrizePicks'
public JSON:API endpoint, flattens the data, and returns clean records.

There is no official PrizePicks API. This calls the same public endpoint the
prizepicks.com site uses to render its board. Intended for personal/research
use — please respect PrizePicks' Terms of Service and rate limits.

Usage (as a library):
    from prizepicks import PrizePicks
    pp = PrizePicks()
    for proj in pp.get_all_projections():
        print(proj.player_name, proj.stat_type, proj.line_score)

Usage (CLI):
    python prizepicks.py                 # all leagues -> stdout (table)
    python prizepicks.py --league NBA    # single league
    python prizepicks.py --json out.json # dump everything to JSON
    python prizepicks.py --csv out.csv   # dump everything to CSV
    python prizepicks.py --list-leagues  # show available leagues + ids
    python prizepicks.py --odds-type demon,goblin     # only special props

    # Auto-update: poll every 60s, print line movements, keep the file fresh
    python prizepicks.py --league NBA --watch --interval 60 --csv board.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://api.prizepicks.com"

# A browser-like UA tends to be required; the endpoint sits behind Cloudflare.
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Origin": "https://app.prizepicks.com",
    "Referer": "https://app.prizepicks.com/",
}


@dataclass
class League:
    id: str
    name: str
    projections_count: Optional[int] = None


@dataclass
class Projection:
    """One prop line: a player + a stat type + the over/under number."""

    id: str
    player_id: Optional[str]
    player_name: Optional[str]
    team: Optional[str]
    position: Optional[str]
    league: Optional[str]
    league_id: Optional[str]
    stat_type: Optional[str]          # e.g. "Points", "Pass Yards"
    line_score: Optional[float]       # the over/under number
    description: Optional[str]        # usually the opponent / matchup
    start_time: Optional[str]         # ISO8601 game/event start
    status: Optional[str]             # pre_game / in_progress / etc.
    odds_type: Optional[str]          # standard / demon / goblin
    is_promo: bool = False
    rank: Optional[int] = None
    board_time: Optional[str] = None
    image_url: Optional[str] = None        # player headshot (PrizePicks CDN)
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)


class PrizePicksError(RuntimeError):
    pass


class PrizePicks:
    def __init__(
        self,
        base_url: str = BASE_URL,
        headers: Optional[Dict[str, str]] = None,
        request_delay: float = 1.0,
        timeout: float = 20.0,
    ) -> None:
        """
        request_delay: seconds to sleep between requests (be polite / avoid 429s).
        """
        self.base_url = base_url.rstrip("/")
        self.request_delay = request_delay
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers.update(headers or DEFAULT_HEADERS)
        retry = Retry(
            total=4,
            backoff_factor=1.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    # ----------------------------------------------------------------- requests
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = self.session.get(url, params=params, timeout=self.timeout)
        if resp.status_code == 403:
            raise PrizePicksError(
                "403 Forbidden — PrizePicks /projections is behind PerimeterX/HUMAN "
                "bot protection. It needs a real browser's _px* cookies + fingerprint; "
                "a plain server request can't pass the JS/captcha challenge. Capture a "
                "browser session (DevTools) and pass headers/cookies, or use Underdog."
            )
        if not resp.ok:
            raise PrizePicksError(f"GET {url} failed: {resp.status_code} {resp.text[:200]}")
        try:
            return resp.json()
        except ValueError as exc:
            raise PrizePicksError(f"Non-JSON response from {url}") from exc

    # ------------------------------------------------------------------ leagues
    def get_leagues(self) -> List[League]:
        payload = self._get("leagues", params={"per_page": 250})
        leagues: List[League] = []
        for item in payload.get("data", []):
            attrs = item.get("attributes", {})
            leagues.append(
                League(
                    id=str(item.get("id")),
                    name=attrs.get("name") or attrs.get("display_name") or "",
                    projections_count=attrs.get("projections_count"),
                )
            )
        return leagues

    def resolve_league_id(self, league: str) -> Optional[str]:
        """Accept a league id ('7') or a name/abbrev ('NBA') and return the id."""
        if league.isdigit():
            return league
        wanted = league.strip().lower()
        for lg in self.get_leagues():
            if lg.name.lower() == wanted:
                return lg.id
        return None

    # -------------------------------------------------------------- projections
    def get_projections(
        self,
        league_id: Optional[str] = None,
        per_page: int = 250,
        max_pages: int = 50,
    ) -> List[Projection]:
        """
        Fetch projections for one league (or the whole board if league_id is None),
        following pagination until exhausted.
        """
        results: List[Projection] = []
        page = 1
        while page <= max_pages:
            params: Dict[str, Any] = {
                "per_page": per_page,
                "page": page,
                "single_stat": "true",
            }
            if league_id is not None:
                params["league_id"] = league_id

            payload = self._get("projections", params=params)
            included_index = self._index_included(payload.get("included", []))
            data = payload.get("data", [])
            if not data:
                break

            for item in data:
                results.append(self._build_projection(item, included_index))

            # JSON:API pagination — stop when we've read every page.
            meta = payload.get("meta", {})
            total_pages = meta.get("total_pages")
            if total_pages is not None:
                if page >= total_pages:
                    break
            elif len(data) < per_page:
                break

            page += 1
            if self.request_delay:
                time.sleep(self.request_delay)

        return results

    def get_all_projections(self) -> Iterator[Projection]:
        """Iterate every projection across every league with projections."""
        for lg in self.get_leagues():
            if lg.projections_count == 0:
                continue
            try:
                yield from self.get_projections(league_id=lg.id)
            except PrizePicksError as exc:
                print(f"[warn] skipping league {lg.name} ({lg.id}): {exc}", file=sys.stderr)
            if self.request_delay:
                time.sleep(self.request_delay)

    # -------------------------------------------------------------- parsing util
    @staticmethod
    def _index_included(included: Iterable[Dict[str, Any]]) -> Dict[tuple, Dict[str, Any]]:
        """Build a {(type, id): attributes} lookup from the JSON:API 'included' array."""
        idx: Dict[tuple, Dict[str, Any]] = {}
        for obj in included:
            key = (obj.get("type"), str(obj.get("id")))
            idx[key] = obj.get("attributes", {})
        return idx

    @staticmethod
    def _rel_ref(item: Dict[str, Any], *rel_names: str) -> Optional[tuple]:
        """Return the (type, id) reference for the first matching relationship."""
        rels = item.get("relationships", {})
        for name in rel_names:
            data = rels.get(name, {}).get("data")
            if data:
                return (data.get("type"), str(data.get("id")))
        return None

    def _build_projection(
        self, item: Dict[str, Any], included_index: Dict[tuple, Dict[str, Any]]
    ) -> Projection:
        attrs = item.get("attributes", {})

        # PrizePicks has used both "new_player" and "player" over time.
        player_ref = self._rel_ref(item, "new_player", "player")
        player_attrs = included_index.get(player_ref, {}) if player_ref else {}

        league_ref = self._rel_ref(item, "league")
        stat_ref = self._rel_ref(item, "stat_type")
        stat_attrs = included_index.get(stat_ref, {}) if stat_ref else {}

        line = attrs.get("line_score")
        try:
            line = float(line) if line is not None else None
        except (TypeError, ValueError):
            line = None

        return Projection(
            id=str(item.get("id")),
            player_id=player_ref[1] if player_ref else None,
            player_name=player_attrs.get("name") or player_attrs.get("display_name"),
            team=player_attrs.get("team") or player_attrs.get("team_name"),
            position=player_attrs.get("position"),
            league=player_attrs.get("league") or attrs.get("league"),
            league_id=league_ref[1] if league_ref else None,
            stat_type=attrs.get("stat_type") or stat_attrs.get("name"),
            line_score=line,
            description=attrs.get("description"),
            start_time=attrs.get("start_time"),
            status=attrs.get("status"),
            odds_type=attrs.get("odds_type"),
            is_promo=bool(attrs.get("is_promo", False)),
            rank=attrs.get("rank"),
            board_time=attrs.get("board_time"),
            image_url=player_attrs.get("image_url"),
            raw=item,
        )


# ----------------------------------------------------------------------- export
def to_dicts(projections: Iterable[Projection]) -> List[Dict[str, Any]]:
    out = []
    for p in projections:
        d = asdict(p)
        d.pop("raw", None)  # drop the bulky original payload
        out.append(d)
    return out


def write_json(projections: Iterable[Projection], path: str) -> int:
    rows = to_dicts(projections)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    return len(rows)


def write_csv(projections: Iterable[Projection], path: str) -> int:
    rows = to_dicts(projections)
    if not rows:
        open(path, "w").close()
        return 0
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


# ------------------------------------------------------------------- filtering
def filter_by_odds_type(
    projections: Iterable[Projection], odds_types: Iterable[str]
) -> List[Projection]:
    """Keep only projections whose odds_type is in the given set.

    PrizePicks uses: 'standard' (normal), 'demon' (harder, higher payout),
    'goblin' (easier, lower payout).
    """
    wanted = {t.strip().lower() for t in odds_types if t.strip()}
    if not wanted:
        return list(projections)
    return [p for p in projections if (p.odds_type or "standard").lower() in wanted]


# ----------------------------------------------------------- snapshot / diffing
def snapshot(projections: Iterable[Projection]) -> Dict[str, Projection]:
    """Index projections by id so two pulls can be compared."""
    return {p.id: p for p in projections}


def diff_snapshots(
    old: Dict[str, Projection], new: Dict[str, Projection]
) -> Dict[str, list]:
    """Compare two snapshots and return what changed.

    Returns dict with keys: 'moved' (line changed), 'added', 'removed'.
    Each 'moved' entry is (projection, old_line, new_line).
    """
    moved, added, removed = [], [], []
    for pid, p in new.items():
        if pid not in old:
            added.append(p)
        elif old[pid].line_score != p.line_score:
            moved.append((p, old[pid].line_score, p.line_score))
    for pid, p in old.items():
        if pid not in new:
            removed.append(p)
    return {"moved": moved, "added": added, "removed": removed}


def _fmt_proj(p: Projection) -> str:
    return (
        f"{p.player_name or '?'} — {p.stat_type or '?'} {p.line_score} "
        f"({p.league or '?'}{', ' + p.odds_type if p.odds_type and p.odds_type != 'standard' else ''})"
    )


def print_diff(changes: Dict[str, list]) -> bool:
    """Print a human-readable changelog. Returns True if anything changed."""
    moved, added, removed = changes["moved"], changes["added"], changes["removed"]
    if not (moved or added or removed):
        return False
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{stamp}] {len(moved)} moved · {len(added)} new · {len(removed)} gone")
    for p, old_line, new_line in moved:
        arrow = "↑" if (new_line or 0) > (old_line or 0) else "↓"
        print(f"  {arrow} LINE  {p.player_name} {p.stat_type}: {old_line} → {new_line}")
    for p in added:
        print(f"  + NEW   {_fmt_proj(p)}")
    for p in removed:
        print(f"  - GONE  {_fmt_proj(p)}")
    return True


def watch(
    pp: "PrizePicks",
    fetch,
    interval: float = 60.0,
    odds_types: Optional[List[str]] = None,
    on_update=None,
) -> None:
    """Poll the board forever, reporting line movement each cycle.

    fetch: a zero-arg callable returning a fresh List[Projection].
    on_update: optional callback(projections) run every cycle — e.g. to
               re-write a JSON/CSV file so it stays current.
    """
    prev: Dict[str, Projection] = {}
    cycle = 0
    print(f"Watching PrizePicks every {interval:.0f}s. Ctrl-C to stop.")
    try:
        while True:
            cycle += 1
            try:
                projections = fetch()
                if odds_types:
                    projections = filter_by_odds_type(projections, odds_types)
            except PrizePicksError as exc:
                print(f"[{datetime.now():%H:%M:%S}] fetch failed: {exc}", file=sys.stderr)
                time.sleep(interval)
                continue

            current = snapshot(projections)
            if cycle == 1:
                print(f"[{datetime.now():%H:%M:%S}] baseline: {len(current)} props.")
            else:
                if not print_diff(diff_snapshots(prev, current)):
                    print(f"[{datetime.now():%H:%M:%S}] no changes ({len(current)} props).")
            prev = current

            if on_update:
                on_update(projections)

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")


# -------------------------------------------------------------------------- CLI
def _print_table(projections: List[Projection]) -> None:
    if not projections:
        print("No projections found.")
        return
    header = f"{'PLAYER':<24}{'LG':<6}{'STAT':<18}{'LINE':>7}  {'MATCHUP':<24}{'TYPE':<10}"
    print(header)
    print("-" * len(header))
    for p in projections:
        print(
            f"{(p.player_name or '?')[:23]:<24}"
            f"{(p.league or '')[:5]:<6}"
            f"{(p.stat_type or '')[:17]:<18}"
            f"{('' if p.line_score is None else p.line_score):>7}  "
            f"{(p.description or '')[:23]:<24}"
            f"{(p.odds_type or '')[:9]:<10}"
        )
    print(f"\n{len(projections)} projections.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Read all PrizePicks props and lines.")
    parser.add_argument("--league", help="League id (e.g. 7) or name (e.g. NBA).")
    parser.add_argument("--list-leagues", action="store_true", help="List leagues and exit.")
    parser.add_argument("--json", metavar="PATH", help="Write all projections to a JSON file.")
    parser.add_argument("--csv", metavar="PATH", help="Write all projections to a CSV file.")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between requests.")
    parser.add_argument(
        "--odds-type",
        help="Comma-separated filter: standard,demon,goblin (e.g. --odds-type demon,goblin).",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep polling and report line movement until stopped (auto-update).",
    )
    parser.add_argument(
        "--interval", type=float, default=60.0, help="Seconds between polls in --watch mode."
    )
    args = parser.parse_args(argv)

    pp = PrizePicks(request_delay=args.delay)
    odds_types = [t for t in args.odds_type.split(",")] if args.odds_type else None

    if args.list_leagues:
        for lg in pp.get_leagues():
            count = "" if lg.projections_count is None else f"  ({lg.projections_count} props)"
            print(f"{lg.id:>5}  {lg.name}{count}")
        return 0

    # Resolve a fetch() closure used by both one-shot and watch modes.
    league_id = None
    if args.league:
        league_id = pp.resolve_league_id(args.league)
        if league_id is None:
            print(f"Unknown league: {args.league!r}. Try --list-leagues.", file=sys.stderr)
            return 2

    def fetch() -> List[Projection]:
        if league_id is not None:
            return pp.get_projections(league_id=league_id)
        return list(pp.get_all_projections())

    # Callback that keeps output files fresh on every refresh.
    def on_update(projections: List[Projection]) -> None:
        if args.json:
            write_json(projections, args.json)
        if args.csv:
            write_csv(projections, args.csv)

    if args.watch:
        watch(pp, fetch, interval=args.interval, odds_types=odds_types, on_update=on_update)
        return 0

    projections = fetch()
    if odds_types:
        projections = filter_by_odds_type(projections, odds_types)

    if args.json:
        n = write_json(projections, args.json)
        print(f"Wrote {n} projections to {args.json}")
    if args.csv:
        n = write_csv(projections, args.csv)
        print(f"Wrote {n} projections to {args.csv}")
    if not args.json and not args.csv:
        _print_table(projections)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
