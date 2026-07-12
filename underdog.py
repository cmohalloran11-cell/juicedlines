"""
underdog.py — a read-only client for Underdog Fantasy pick'em props.

Like the PrizePicks client, this talks directly to Underdog's own public board
endpoint (the same one their site calls) and flattens it into clean records.
No API key. Personal/research use — respect Underdog's Terms and rate limits.

Underdog returns the board as separate `players`, `appearances`, and
`over_under_lines` arrays that must be joined:
    over_under_line → over_under.appearance_stat.appearance_id
                    → appearance.player_id → player
Each line has two `options` (higher / lower), each with a payout multiplier.
A multiplier other than 1.0 means a boosted/insured pick (Underdog's analog to
PrizePicks demons/goblins).

Usage (library):
    from underdog import Underdog
    ud = Underdog()
    for p in ud.get_props():
        print(p.player_name, p.stat, p.line, p.choice, p.payout_multiplier)

Usage (CLI):
    python underdog.py                       # all props -> table
    python underdog.py --sport NBA           # filter by sport
    python underdog.py --boosted             # only payout multipliers != 1.0
    python underdog.py --csv ud.csv
    python underdog.py --watch --interval 60 --csv ud.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PICKEM_URL = "https://api.underdogfantasy.com/beta/v5/over_under_lines"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://underdogfantasy.com/",
}

_CHOICE_MAP = {"higher": "over", "lower": "under"}


@dataclass
class UnderdogProp:
    line_id: str
    option_id: Optional[str]
    player_name: Optional[str]
    team_id: Optional[str]
    sport: Optional[str]
    stat: Optional[str]
    line: Optional[float]
    choice: Optional[str]              # over / under
    payout_multiplier: Optional[float]
    american_price: Optional[Any]
    match_id: Optional[str]
    status: Optional[str]
    image_url: Optional[str] = None     # player headshot (Underdog CDN)
    country: Optional[str] = None       # ISO-3 country code (soccer)
    position: Optional[str] = None      # position display name
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def key(self) -> str:
        return f"{self.line_id}|{self.choice}"

    @property
    def is_boosted(self) -> bool:
        return self.payout_multiplier is not None and abs(self.payout_multiplier - 1.0) > 1e-9


class UnderdogError(RuntimeError):
    pass


class Underdog:
    def __init__(
        self,
        url: str = PICKEM_URL,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 20.0,
    ) -> None:
        self.url = url
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

    def _get(self) -> Dict[str, Any]:
        resp = self.session.get(self.url, timeout=self.timeout)
        if resp.status_code in (401, 403):
            raise UnderdogError(
                f"{resp.status_code} — Underdog rejected the request. The public board "
                "endpoint sometimes needs headers/cookies from a real browser session; "
                "capture them from DevTools and pass headers=... if this persists."
            )
        if not resp.ok:
            raise UnderdogError(f"GET {self.url} failed: {resp.status_code} {resp.text[:200]}")
        try:
            return resp.json()
        except ValueError as exc:
            raise UnderdogError("Non-JSON response from Underdog.") from exc

    def get_props(self) -> List[UnderdogProp]:
        return self._flatten(self._get())

    # -------------------------------------------------------------- flattening
    @staticmethod
    def _flatten(data: Dict[str, Any]) -> List[UnderdogProp]:
        players = {p.get("id"): p for p in data.get("players", [])}
        appearances = {a.get("id"): a for a in data.get("appearances", [])}

        out: List[UnderdogProp] = []
        for line in data.get("over_under_lines", []):
            ou = line.get("over_under", {}) or {}
            appearance_stat = ou.get("appearance_stat", {}) or {}
            appearance_id = appearance_stat.get("appearance_id")
            stat = appearance_stat.get("stat") or appearance_stat.get("display_stat")

            appearance = appearances.get(appearance_id, {}) or {}
            player = players.get(appearance.get("player_id"), {}) or {}
            name = ou.get("title") or " ".join(
                x for x in [player.get("first_name"), player.get("last_name")] if x
            ) or None

            line_val = line.get("stat_value", line.get("line"))
            try:
                line_val = float(line_val) if line_val is not None else None
            except (TypeError, ValueError):
                line_val = None

            sport = player.get("sport_id") or appearance.get("sport_id")
            image_url = player.get("image_url") or player.get("light_image_url")
            country = player.get("country")
            position = player.get("position_name") or player.get("position_display_name")
            options = line.get("options") or [{}]
            for opt in options:
                choice_raw = opt.get("choice")
                mult = opt.get("payout_multiplier")
                try:
                    mult = float(mult) if mult is not None else None
                except (TypeError, ValueError):
                    mult = None
                out.append(
                    UnderdogProp(
                        line_id=str(line.get("id", "")),
                        option_id=opt.get("id"),
                        player_name=name,
                        team_id=appearance.get("team_id") or player.get("team_id"),
                        sport=sport,
                        stat=stat,
                        line=line_val,
                        choice=_CHOICE_MAP.get(choice_raw, choice_raw),
                        payout_multiplier=mult,
                        american_price=opt.get("american_price"),
                        match_id=appearance.get("match_id"),
                        status=line.get("status"),
                        image_url=image_url,
                        country=country,
                        position=position,
                        raw=line,
                    )
                )
        return out


# ------------------------------------------------------------------- filtering
def filter_by_sport(props: Iterable[UnderdogProp], sports: Iterable[str]) -> List[UnderdogProp]:
    wanted = {s.strip().upper() for s in sports if s.strip()}
    if not wanted:
        return list(props)
    return [p for p in props if (p.sport or "").upper() in wanted]


def boosted_only(props: Iterable[UnderdogProp]) -> List[UnderdogProp]:
    return [p for p in props if p.is_boosted]


# ----------------------------------------------------------------------- export
def to_dicts(props: Iterable[UnderdogProp]) -> List[Dict[str, Any]]:
    rows = []
    for p in props:
        d = asdict(p)
        d.pop("raw", None)
        rows.append(d)
    return rows


def write_json(props: Iterable[UnderdogProp], path: str) -> int:
    rows = to_dicts(props)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    return len(rows)


def write_csv(props: Iterable[UnderdogProp], path: str) -> int:
    rows = to_dicts(props)
    if not rows:
        open(path, "w").close()
        return 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


# ----------------------------------------------------------- snapshot / diffing
def snapshot(props: Iterable[UnderdogProp]) -> Dict[str, UnderdogProp]:
    return {p.key: p for p in props}


def diff_snapshots(old: Dict[str, UnderdogProp], new: Dict[str, UnderdogProp]) -> Dict[str, list]:
    """'moved' = line or payout multiplier changed."""
    moved, added, removed = [], [], []
    for k, p in new.items():
        if k not in old:
            added.append(p)
        elif (old[k].line, old[k].payout_multiplier) != (p.line, p.payout_multiplier):
            moved.append((p, (old[k].line, old[k].payout_multiplier), (p.line, p.payout_multiplier)))
    for k, p in old.items():
        if k not in new:
            removed.append(p)
    return {"moved": moved, "added": added, "removed": removed}


def print_diff(changes: Dict[str, list]) -> bool:
    moved, added, removed = changes["moved"], changes["added"], changes["removed"]
    if not (moved or added or removed):
        return False
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{stamp}] {len(moved)} moved · {len(added)} new · {len(removed)} gone")
    for p, ov, nv in moved:
        print(f"  ~ {p.player_name} {p.stat} ({p.choice}): line {ov[0]}→{nv[0]} mult {ov[1]}→{nv[1]}")
    for p in added:
        b = " [boosted]" if p.is_boosted else ""
        print(f"  + NEW  {p.player_name} {p.stat} {p.choice} {p.line}{b}")
    for p in removed:
        print(f"  - GONE {p.player_name} {p.stat} {p.choice}")
    return True


def watch(fetch, interval: float = 60.0, sports: Optional[List[str]] = None,
          boosted: bool = False, on_update=None) -> None:
    prev: Dict[str, UnderdogProp] = {}
    cycle = 0
    print(f"Watching Underdog every {interval:.0f}s. Ctrl-C to stop.")
    try:
        while True:
            cycle += 1
            try:
                props = fetch()
                if sports:
                    props = filter_by_sport(props, sports)
                if boosted:
                    props = boosted_only(props)
            except UnderdogError as exc:
                print(f"[{datetime.now():%H:%M:%S}] fetch failed: {exc}", file=sys.stderr)
                time.sleep(interval)
                continue
            cur = snapshot(props)
            if cycle == 1:
                print(f"[{datetime.now():%H:%M:%S}] baseline: {len(cur)} options.")
            elif not print_diff(diff_snapshots(prev, cur)):
                print(f"[{datetime.now():%H:%M:%S}] no changes ({len(cur)} options).")
            prev = cur
            if on_update:
                on_update(props)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")


# -------------------------------------------------------------------------- CLI
def _print_table(props: List[UnderdogProp]) -> None:
    if not props:
        print("No props found.")
        return
    header = f"{'PLAYER':<24}{'SPORT':<7}{'STAT':<20}{'LINE':>7} {'CHOICE':<7}{'MULT':>6}"
    print(header)
    print("-" * len(header))
    for p in props:
        print(
            f"{(p.player_name or '?')[:23]:<24}{(p.sport or '')[:6]:<7}"
            f"{(p.stat or '')[:19]:<20}{('' if p.line is None else p.line):>7} "
            f"{(p.choice or '')[:6]:<7}{('' if p.payout_multiplier is None else p.payout_multiplier):>6}"
        )
    print(f"\n{len(props)} options.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Read Underdog Fantasy pick'em props.")
    parser.add_argument("--sport", help="Filter by sport id, e.g. NBA (comma-separated ok).")
    parser.add_argument("--boosted", action="store_true",
                        help="Only props with a payout multiplier != 1.0.")
    parser.add_argument("--json", metavar="PATH", help="Write props to JSON.")
    parser.add_argument("--csv", metavar="PATH", help="Write props to CSV.")
    parser.add_argument("--watch", action="store_true", help="Poll and report movement.")
    parser.add_argument("--interval", type=float, default=60.0, help="Seconds between polls.")
    args = parser.parse_args(argv)

    ud = Underdog()
    sports = [s for s in args.sport.split(",")] if args.sport else None

    def fetch() -> List[UnderdogProp]:
        return ud.get_props()

    def on_update(props: List[UnderdogProp]) -> None:
        if args.json:
            write_json(props, args.json)
        if args.csv:
            write_csv(props, args.csv)

    if args.watch:
        watch(fetch, interval=args.interval, sports=sports, boosted=args.boosted, on_update=on_update)
        return 0

    props = fetch()
    if sports:
        props = filter_by_sport(props, sports)
    if args.boosted:
        props = boosted_only(props)

    if args.json:
        print(f"Wrote {write_json(props, args.json)} props to {args.json}")
    if args.csv:
        print(f"Wrote {write_csv(props, args.csv)} props to {args.csv}")
    if not args.json and not args.csv:
        _print_table(props)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
