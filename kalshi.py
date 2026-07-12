"""
kalshi.py — a read-only client for Kalshi market prices.

Unlike the sportsbook/DFS clients, this uses Kalshi's OFFICIAL, documented public
API. Reading market data needs NO API key and NO scraping — it's a supported REST
endpoint. (Only trading/portfolio endpoints require RSA-signed auth, which this
client does not touch.)

Base URL: https://external-api.kalshi.com/trade-api/v2
(Alternate hosts that also serve public data: api.elections.kalshi.com,
 api.kalshi.com, trading-api.kalshi.com — override with --base-url if one fails.)

Kalshi markets are binary event contracts. Prices are in CENTS (1–99) and read as
implied probability: a yes_bid of 61 means the market prices "yes" at ~61%.

Usage (library):
    from kalshi import Kalshi
    k = Kalshi()
    for m in k.get_markets(status="open", limit=200):
        print(m.ticker, m.title, m.yes_bid, m.implied_prob)

Usage (CLI):
    python kalshi.py --search "fed rate"          # markets whose title matches
    python kalshi.py --series KXHIGHNY            # all markets in a series
    python kalshi.py --event KXFED-26MAR19        # all markets in an event
    python kalshi.py --status open --csv k.csv
    python kalshi.py --search bitcoin --watch --interval 30 --csv btc.csv
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

DEFAULT_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
DEFAULT_HEADERS = {"Accept": "application/json", "User-Agent": "kalshi-reader/1.0"}


def _cents(value: Any, dollars: Any = None) -> Optional[int]:
    """Normalise a price to integer cents. Prefer the cent field; fall back to a
    '*_dollars' string field (e.g. '0.61' -> 61)."""
    if value is not None:
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            pass
    if dollars is not None:
        try:
            return int(round(float(dollars) * 100))
        except (TypeError, ValueError):
            pass
    return None


@dataclass
class KalshiMarket:
    ticker: str
    event_ticker: Optional[str]
    title: Optional[str]
    subtitle: Optional[str]
    status: Optional[str]
    yes_bid: Optional[int]      # cents
    yes_ask: Optional[int]
    no_bid: Optional[int]
    no_ask: Optional[int]
    last_price: Optional[int]
    volume: Optional[int]
    volume_24h: Optional[int]
    open_interest: Optional[int]
    close_time: Optional[str]
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def implied_prob(self) -> Optional[float]:
        """Mid-market implied probability (0–1). Falls back to last_price."""
        if self.yes_bid is not None and self.yes_ask is not None:
            return round((self.yes_bid + self.yes_ask) / 200, 4)
        if self.last_price is not None:
            return round(self.last_price / 100, 4)
        return None

    @property
    def key(self) -> str:
        return self.ticker


class KalshiError(RuntimeError):
    pass


class Kalshi:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 20.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        retry = Retry(
            total=4,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = self.session.get(url, params=params, timeout=self.timeout)
        if not resp.ok:
            raise KalshiError(f"GET {url} failed: {resp.status_code} {resp.text[:200]}")
        try:
            return resp.json()
        except ValueError as exc:
            raise KalshiError(f"Non-JSON response from {url}") from exc

    # ------------------------------------------------------------------ markets
    def get_markets(
        self,
        status: Optional[str] = "open",
        series_ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
        limit: int = 1000,
        max_pages: int = 50,
    ) -> List[KalshiMarket]:
        """List markets, following cursor pagination up to `limit` total."""
        out: List[KalshiMarket] = []
        cursor: Optional[str] = None
        pages = 0
        while pages < max_pages and len(out) < limit:
            params: Dict[str, Any] = {"limit": min(1000, limit - len(out))}
            if status and status != "all":
                params["status"] = status
            if series_ticker:
                params["series_ticker"] = series_ticker
            if event_ticker:
                params["event_ticker"] = event_ticker
            if cursor:
                params["cursor"] = cursor

            payload = self._get("markets", params=params)
            batch = payload.get("markets", [])
            out.extend(self._build_market(m) for m in batch)

            cursor = payload.get("cursor")
            pages += 1
            if not cursor or not batch:
                break
        return out

    def get_market(self, ticker: str) -> KalshiMarket:
        payload = self._get(f"markets/{ticker}")
        return self._build_market(payload.get("market", payload))

    def get_orderbook(self, ticker: str, depth: int = 10) -> Dict[str, Any]:
        return self._get(f"markets/{ticker}/orderbook", params={"depth": depth})

    # ------------------------------------------------------------------- events
    def get_events(
        self, status: Optional[str] = "open", series_ticker: Optional[str] = None,
        with_nested_markets: bool = False, limit: int = 200,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"limit": limit}
        if status and status != "all":
            params["status"] = status
        if series_ticker:
            params["series_ticker"] = series_ticker
        if with_nested_markets:
            params["with_nested_markets"] = "true"
        return self._get("events", params=params).get("events", [])

    def get_series(self, series_ticker: str) -> Dict[str, Any]:
        return self._get(f"series/{series_ticker}").get("series", {})

    # --------------------------------------------------------------- build/util
    @staticmethod
    def _build_market(m: Dict[str, Any]) -> KalshiMarket:
        return KalshiMarket(
            ticker=m.get("ticker", ""),
            event_ticker=m.get("event_ticker"),
            title=m.get("title"),
            subtitle=m.get("subtitle") or m.get("yes_sub_title"),
            status=m.get("status"),
            yes_bid=_cents(m.get("yes_bid"), m.get("yes_bid_dollars")),
            yes_ask=_cents(m.get("yes_ask"), m.get("yes_ask_dollars")),
            no_bid=_cents(m.get("no_bid"), m.get("no_bid_dollars")),
            no_ask=_cents(m.get("no_ask"), m.get("no_ask_dollars")),
            last_price=_cents(m.get("last_price"), m.get("last_price_dollars")),
            volume=m.get("volume", m.get("volume_fp")),
            volume_24h=m.get("volume_24h"),
            open_interest=m.get("open_interest"),
            close_time=m.get("close_time") or m.get("expiration_time"),
            raw=m,
        )


# ------------------------------------------------------------------- filtering
def search_markets(markets: Iterable[KalshiMarket], query: str) -> List[KalshiMarket]:
    q = query.lower()
    return [m for m in markets
            if q in (m.title or "").lower() or q in (m.ticker or "").lower()
            or q in (m.subtitle or "").lower()]


# ----------------------------------------------------------------------- export
def to_dicts(markets: Iterable[KalshiMarket]) -> List[Dict[str, Any]]:
    rows = []
    for m in markets:
        d = asdict(m)
        d.pop("raw", None)
        d["implied_prob"] = m.implied_prob
        rows.append(d)
    return rows


def write_json(markets: Iterable[KalshiMarket], path: str) -> int:
    rows = to_dicts(markets)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    return len(rows)


def write_csv(markets: Iterable[KalshiMarket], path: str) -> int:
    rows = to_dicts(markets)
    if not rows:
        open(path, "w").close()
        return 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


# ----------------------------------------------------------- snapshot / diffing
def snapshot(markets: Iterable[KalshiMarket]) -> Dict[str, KalshiMarket]:
    return {m.key: m for m in markets}


def diff_snapshots(old: Dict[str, KalshiMarket], new: Dict[str, KalshiMarket]) -> Dict[str, list]:
    """'moved' = yes_bid / yes_ask / last_price changed."""
    moved, added, removed = [], [], []
    for k, m in new.items():
        if k not in old:
            added.append(m)
        else:
            o = old[k]
            if (o.yes_bid, o.yes_ask, o.last_price) != (m.yes_bid, m.yes_ask, m.last_price):
                moved.append((m, o.implied_prob, m.implied_prob))
    for k, m in old.items():
        if k not in new:
            removed.append(m)
    return {"moved": moved, "added": added, "removed": removed}


def print_diff(changes: Dict[str, list]) -> bool:
    moved, added, removed = changes["moved"], changes["added"], changes["removed"]
    if not (moved or added or removed):
        return False
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{stamp}] {len(moved)} moved · {len(added)} new · {len(removed)} gone")
    for m, op, npb in moved:
        op = "?" if op is None else f"{op:.0%}"
        npb = "?" if npb is None else f"{npb:.0%}"
        print(f"  ~ {m.ticker}: {op} → {npb}  ({m.title})")
    for m in added:
        ip = "?" if m.implied_prob is None else f"{m.implied_prob:.0%}"
        print(f"  + NEW  {m.ticker} @ {ip}  ({m.title})")
    for m in removed:
        print(f"  - GONE {m.ticker}")
    return True


def watch(fetch, interval: float = 30.0, on_update=None) -> None:
    prev: Dict[str, KalshiMarket] = {}
    cycle = 0
    print(f"Watching Kalshi every {interval:.0f}s. Ctrl-C to stop.")
    try:
        while True:
            cycle += 1
            try:
                markets = fetch()
            except KalshiError as exc:
                print(f"[{datetime.now():%H:%M:%S}] fetch failed: {exc}", file=sys.stderr)
                time.sleep(interval)
                continue
            cur = snapshot(markets)
            if cycle == 1:
                print(f"[{datetime.now():%H:%M:%S}] baseline: {len(cur)} markets.")
            elif not print_diff(diff_snapshots(prev, cur)):
                print(f"[{datetime.now():%H:%M:%S}] no changes ({len(cur)} markets).")
            prev = cur
            if on_update:
                on_update(markets)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")


# -------------------------------------------------------------------------- CLI
def _print_table(markets: List[KalshiMarket]) -> None:
    if not markets:
        print("No markets found.")
        return
    header = f"{'TICKER':<26}{'YES%':>6}{'BID':>5}{'ASK':>5}{'VOL':>9}  {'TITLE':<40}"
    print(header)
    print("-" * len(header))
    for m in markets[:200]:
        prob = "" if m.implied_prob is None else f"{m.implied_prob:.0%}"
        print(
            f"{m.ticker[:25]:<26}{prob:>6}"
            f"{('' if m.yes_bid is None else m.yes_bid):>5}"
            f"{('' if m.yes_ask is None else m.yes_ask):>5}"
            f"{('' if m.volume is None else m.volume):>9}  {(m.title or '')[:39]:<40}"
        )
    print(f"\n{len(markets)} markets" + (" (showing first 200)" if len(markets) > 200 else "") + ".")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Read Kalshi market prices (official public API).")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Override API base URL.")
    parser.add_argument("--status", default="open", help="open / closed / settled / all.")
    parser.add_argument("--series", help="Series ticker, e.g. KXHIGHNY.")
    parser.add_argument("--event", help="Event ticker, e.g. KXFED-26MAR19.")
    parser.add_argument("--search", help="Keep only markets whose title/ticker matches.")
    parser.add_argument("--limit", type=int, default=1000, help="Max markets to fetch.")
    parser.add_argument("--json", metavar="PATH", help="Write markets to JSON.")
    parser.add_argument("--csv", metavar="PATH", help="Write markets to CSV.")
    parser.add_argument("--watch", action="store_true", help="Poll and report price movement.")
    parser.add_argument("--interval", type=float, default=30.0, help="Seconds between polls.")
    args = parser.parse_args(argv)

    k = Kalshi(base_url=args.base_url)

    def fetch() -> List[KalshiMarket]:
        markets = k.get_markets(
            status=args.status, series_ticker=args.series,
            event_ticker=args.event, limit=args.limit,
        )
        if args.search:
            markets = search_markets(markets, args.search)
        return markets

    def on_update(markets: List[KalshiMarket]) -> None:
        if args.json:
            write_json(markets, args.json)
        if args.csv:
            write_csv(markets, args.csv)

    if args.watch:
        watch(fetch, interval=args.interval, on_update=on_update)
        return 0

    markets = fetch()
    if args.json:
        print(f"Wrote {write_json(markets, args.json)} markets to {args.json}")
    if args.csv:
        print(f"Wrote {write_csv(markets, args.csv)} markets to {args.csv}")
    if not args.json and not args.csv:
        _print_table(markets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
