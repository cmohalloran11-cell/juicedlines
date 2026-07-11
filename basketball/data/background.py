"""
Summer League player-background source — draft slot + pre-NBA league + pre-NBA
per-40 production, keyed to each SL player. Feeds the translated priors that
DOMINATE Summer-League projections (the SL sample can't).

Design (swappable, like every other adapter):
  1. A local seed file `sl_background.json` is the reliable, testable injection point
     — curated/known backgrounds go here and take precedence.
  2. Bart Torvik (college) is wired as a best-effort enrichment: the season CSV at
     `getadvstats.php?year=YYYY&csv=1` is fetchable (confirmed 200) and indexed by
     name to at least tag `pre_league=NCAA`. Its columns are advanced rates, so a
     full per-40 counting-stat mapping is the remaining wire-up (validate columns
     before trusting it) — hence conservative here.
  3. RealGM (international) — documented stub; wire against confirmed layout later.
Anything not found returns None, and the projector falls back to the generic SL
positional prior (wide intervals, low confidence — correct SL behaviour by design).
"""

from __future__ import annotations

import json
import time
import unicodedata
from pathlib import Path

import requests

from .base import BackgroundSource, PlayerBackground

_SEED_PATH = Path(__file__).parent / "sl_background.json"
_TORVIK_CSV = "https://barttorvik.com/getadvstats.php?year={year}&csv=1"

_S = requests.Session()
_S.headers.update({"User-Agent": "Mozilla/5.0"})


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace(".", "").replace("'", "").split())


class TorvikRealGMBackground(BackgroundSource):
    def __init__(self, torvik_year: int | None = None):
        self._seed = self._load_seed()
        self._torvik_year = torvik_year
        self._torvik_names: set | None = None      # lazy college-name index
        self._torvik_ts = 0.0

    # ── seed file (primary, reliable) ─────────────────────────────────────────
    def _load_seed(self) -> dict:
        try:
            raw = json.loads(_SEED_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
        out = {}
        for row in raw:
            nm = _norm(row.get("player", ""))
            if nm:
                out[nm] = row
        return out

    def _from_seed(self, name: str) -> PlayerBackground | None:
        row = self._seed.get(_norm(name))
        if not row:
            return None
        return PlayerBackground(
            player=row.get("player", name),
            draft_pick=row.get("draft_pick"),
            pre_league=row.get("pre_league", ""),
            archetype=row.get("archetype", ""),
            rates40=row.get("rates40", {}) or {},
            minutes_prior=row.get("minutes_prior"))

    # ── Torvik college name index (best-effort enrichment) ────────────────────
    def _torvik_index(self) -> set:
        if self._torvik_names is not None and time.time() - self._torvik_ts < 24 * 3600:
            return self._torvik_names
        names: set = set()
        year = self._torvik_year
        if year:
            try:
                r = _S.get(_TORVIK_CSV.format(year=year), timeout=20)
                if r.status_code == 200 and "text/csv" in r.headers.get("Content-Type", ""):
                    import csv
                    import io
                    for row in csv.reader(io.StringIO(r.text)):
                        if row:
                            names.add(_norm(row[0]))
            except Exception:
                pass
        self._torvik_names, self._torvik_ts = names, time.time()
        return names

    def background(self, player: str) -> PlayerBackground | None:
        seed = self._from_seed(player)
        if seed:
            return seed
        # best-effort: at least tag NCAA source when Torvik knows the name, so the
        # translation factor applies once real rates are wired. No fabricated rates.
        if _norm(player) in self._torvik_index():
            return PlayerBackground(player=player, pre_league="NCAA")
        return None
