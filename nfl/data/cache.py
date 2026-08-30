"""
Two-level cache for the nflverse pulls.

There is no cross-sport cache helper in this repo to reuse — basketball/data/espn.py keeps a
module-level `{url: (ts, value)}` dict and tennis/data/sackmann.py re-reads its mirror each
run. That in-process pattern alone isn't enough here: the nflverse releases are 2-50MB CSVs
and a cold process would re-download tens of megabytes before serving a single line, so the
raw bytes also land on disk with their own TTL, and only the PARSED result is held in memory.

Disk cache location is `NFL_CACHE_DIR` (default `data/nfl_cache/` under the repo). It is a
pure cache: deleting it costs a re-download and nothing else.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import requests

_DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "nfl_cache"

_memory: dict[tuple, tuple[float, object]] = {}


def cache_dir() -> Path:
    return Path(os.getenv("NFL_CACHE_DIR", str(_DEFAULT_DIR)))


def memoize(key: tuple, ttl: float, build):
    """In-process TTL cache. `key` is a tuple — the parsed caches are keyed by
    ("NFL", release, season) so two releases' parsed results never overwrite each other."""
    hit = _memory.get(key)
    now = time.time()
    if hit is not None and now - hit[0] < ttl:
        return hit[1]
    value = build()
    _memory[key] = (now, value)
    return value


def clear() -> None:
    _memory.clear()


def fetch_text(url: str, ttl: float, timeout: float = 180.0) -> str | None:
    """The release CSV as text, from disk if it's fresh enough, else streamed from `url` and
    written through. None when the release genuinely isn't published (404 — a season that
    hasn't started), which callers must treat as "unknown", not as empty data.

    requests leaves `resp.encoding` unset for these (GitHub serves them as
    application/octet-stream), and `iter_lines(decode_unicode=True)` silently yields BYTES
    rather than str in that case — csv.DictReader then dies with "iterator should return
    strings". Hence the explicit encoding assignment before any decoding happens.
    """
    d = cache_dir()
    path = d / (url.rsplit("/", 2)[-2] + "__" + url.rsplit("/", 1)[-1])
    if path.exists() and time.time() - path.stat().st_mtime < ttl:
        return path.read_text(encoding="utf-8")
    try:
        with requests.get(url, stream=True, timeout=timeout) as resp:
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            resp.encoding = resp.encoding or "utf-8"
            text = resp.text
    except requests.RequestException:
        # A network blip must not lose a cached copy that's merely stale.
        return path.read_text(encoding="utf-8") if path.exists() else None
    try:
        d.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError:
        pass
    return text
