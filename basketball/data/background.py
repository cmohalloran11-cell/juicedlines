"""
Summer League player-background source — draft slot + pre-NBA league + pre-NBA
per-40 production, keyed to each SL player. Feeds the translated priors that
DOMINATE Summer-League projections (the SL sample can't).

Design (swappable, like every other adapter):
  1. A local seed file `sl_background.json` is the reliable, testable override — curated
     backgrounds (international / G-League players, or manual draft slots) go here and
     take precedence. A partial row (just `draft_pick` / `pre_league`) overlays onto the
     auto-pulled college rates below.
  2. Bart Torvik (college) is the automatic feed: the season CSV at
     `getadvstats.php?year=YYYY&csv=1` (confirmed 200, no key) carries per-game box lines
     + minutes-share for every D-I player. We decode it to a per-40 line, newest college
     season first (a player's LAST season is the pre-NBA signal). This is what turns
     "generic rookie prior" into "AJ Dybantsa's actual BYU production."
  3. RealGM (international) — handled via the seed file until a confirmed layout is wired.
Anything not found (international/G-League with no seed) returns None, and the projector
falls back to the generic SL positional prior (wide intervals, low confidence).

Torvik `getadvstats` CSV column map (no header row; decoded + verified against known
players — e.g. Dybantsa 2PM·2+3PM·3+FTM = season pts):
    [3] GP   [4] Min%(of team min)   [6] usg   [12] TO%   [14] FTA   [17] 2PA   [19] 3PM(tot)
    [20] 3PA   per-game: [57] ORB [58] DRB [59] REB [60] AST [61] STL [62] BLK [63] PTS
    [64] archetype
Minutes/game = Min%/100 × 40 (NCAA games are 40 min); per-40 = per-game × 40 / MPG.
"""

from __future__ import annotations

import csv
import io
import json
import time
import unicodedata
from pathlib import Path

import requests

from .base import BackgroundSource, PlayerBackground

_SEED_PATH = Path(__file__).parent / "sl_background.json"
# committed static overlay: {normalized name: overall draft pick}, parsed once from the
# Wikipedia 2024–2026 draft tables (results are immutable → no need to refetch per build).
# Drives the showcase minutes bump (top-14) and the draft-slot minutes prior for 0-game players.
_DRAFT_PATH = Path(__file__).parent / "draft_picks.json"
_TORVIK_CSV = "https://barttorvik.com/getadvstats.php?year={year}&csv=1"
# search newest → oldest; a player's most recent college season is their pre-NBA signal
_TORVIK_YEARS = (2026, 2025, 2024)
_INDEX_TTL = 12 * 3600

# decoded getadvstats column indices
_C_GP, _C_MINPCT, _C_TOPCT = 3, 4, 12
_C_FTA, _C_2PA, _C_3PMTOT, _C_3PA = 14, 17, 19, 20
_C_ORB, _C_DRB, _C_REB, _C_AST, _C_STL, _C_BLK, _C_PTS, _C_ARCH = 57, 58, 59, 60, 61, 62, 63, 64

# quality gate: below this, the per-40 extrapolation is too noisy to trust as a prior
_MIN_GP, _MIN_MINPCT = 5, 25.0
# don't extrapolate a limited-minute season to a full-40 rate a player never sustained
_MPG_FLOOR = 26.0
# sane per-40 ceilings (elite college seasons) so low-minute bursts can't blow up a prior
_PER40_CAP = {"pts": 32.0, "reb": 15.0, "ast": 9.5, "stl": 3.2, "blk": 4.0, "3pm": 4.2, "to": 5.5}

_S = requests.Session()
_S.headers.update({"User-Agent": "Mozilla/5.0"})


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace(".", "").replace("'", "").split())


def _f(row: list, i: int) -> float:
    try:
        return float(row[i])
    except (ValueError, IndexError, TypeError):
        return 0.0


def _per40_from_row(row: list) -> dict | None:
    """Decode one Torvik row → raw college per-40 (pre-translation). None if too thin."""
    gp = _f(row, _C_GP)
    minpct = _f(row, _C_MINPCT)
    if gp < _MIN_GP or minpct < _MIN_MINPCT:
        return None
    mpg = max(_MPG_FLOOR, minpct / 100.0 * 40.0)
    scale = 40.0 / mpg                                   # per-game → per-40
    pg = {
        "pts": _f(row, _C_PTS), "reb": _f(row, _C_REB), "ast": _f(row, _C_AST),
        "stl": _f(row, _C_STL), "blk": _f(row, _C_BLK),
        "3pm": _f(row, _C_3PMTOT) / gp,                  # col 19 is a season total
    }
    # turnovers aren't a per-game column — recover from TO% (turnovers per possession used)
    topct = _f(row, _C_TOPCT) / 100.0
    fga = _f(row, _C_2PA) + _f(row, _C_3PA)
    fta = _f(row, _C_FTA)
    if 0.0 < topct < 0.9:
        to_total = topct * (fga + 0.44 * fta) / (1.0 - topct)
        pg["to"] = to_total / gp
    else:
        pg["to"] = 0.0
    return {k: round(min(_PER40_CAP[k], v * scale), 2) for k, v in pg.items()}


class TorvikRealGMBackground(BackgroundSource):
    def __init__(self):
        self._seed = self._load_seed()
        self._draft = self._load_draft()
        self._idx_cache: dict = {}          # year -> (ts, {norm_name: row})

    # ── seed file (override) ──────────────────────────────────────────────────
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

    def _load_draft(self) -> dict:
        try:
            d = json.loads(_DRAFT_PATH.read_text(encoding="utf-8"))
            return {_norm(k): int(v) for k, v in d.items()}
        except Exception:
            return {}

    def _bg_from_seed(self, row: dict, name: str) -> PlayerBackground:
        return PlayerBackground(
            player=row.get("player", name),
            draft_pick=row.get("draft_pick"),
            pre_league=row.get("pre_league", ""),
            archetype=row.get("archetype", ""),
            rates40=row.get("rates40", {}) or {},
            minutes_prior=row.get("minutes_prior"))

    # ── Torvik college index (per year, cached) ───────────────────────────────
    def _year_index(self, year: int) -> dict:
        hit = self._idx_cache.get(year)
        if hit and time.time() - hit[0] < _INDEX_TTL:
            return hit[1]
        idx: dict = {}
        try:
            r = _S.get(_TORVIK_CSV.format(year=year), timeout=25)
            if r.status_code == 200 and "csv" in r.headers.get("Content-Type", ""):
                for row in csv.reader(io.StringIO(r.text)):
                    if not row or len(row) <= _C_ARCH:
                        continue
                    nm = _norm(row[0])
                    # keep the higher-GP row on the rare in-year name collision
                    if nm and (nm not in idx or _f(row, _C_GP) > _f(idx[nm], _C_GP)):
                        idx[nm] = row
        except Exception:
            idx = {}
        self._idx_cache[year] = (time.time(), idx)
        return idx

    def _torvik(self, name: str) -> tuple[dict, str] | None:
        """(per40, archetype) from the player's most recent college season, or None."""
        key = _norm(name)
        for year in _TORVIK_YEARS:
            row = self._year_index(year).get(key)
            if not row:
                continue
            per40 = _per40_from_row(row)
            if per40:
                arch = row[_C_ARCH] if len(row) > _C_ARCH else ""
                return per40, arch
        return None

    # ── public ────────────────────────────────────────────────────────────────
    def background(self, player: str) -> PlayerBackground | None:
        seed = self._seed.get(_norm(player))
        # a seeded row WITH explicit rates is a full override (international / G-League)
        if seed and (seed.get("rates40")):
            return self._bg_from_seed(seed, player)
        # otherwise pull college rates from Torvik, overlaying any seeded draft slot
        got = self._torvik(player)
        if got:
            per40, arch = got
            pick = (seed or {}).get("draft_pick")
            if pick is None:
                pick = self._draft.get(_norm(player))     # static Wikipedia draft overlay
            return PlayerBackground(
                player=player,
                draft_pick=pick,
                pre_league=(seed or {}).get("pre_league") or "NCAA",
                archetype=(seed or {}).get("archetype") or arch,
                rates40=per40)
        # nothing usable (international/G-League with no seed) → generic SL prior
        return None
