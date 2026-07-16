"""
config.py — load configuration with safe defaults.

The whole system runs without a config.yaml; defaults below mirror
config.example.yaml. Edit config.yaml to override.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

_DEFAULTS: dict[str, Any] = {
    "paths": {"db": str(ROOT / "data.db"), "cache_ttl_hours": 12},
    "blend": {
        # recency weight FITTED on real 2023 data (was a guessed 0.60; the data
        # says single-game counts want the season average, recent form is noise).
        "recent_weight": 0.25,
        "season_weight": 0.65,
        "career_weight": 0.10,
        "recent_games": 15,
        "shrinkage_pa": 60,
        "shrinkage_min90": 8,
        # per-stat overrides (fitted): walks/runs reward recency, hits/HR/TB don't
        "recency_by_stat": {
            "hits": 0.0, "home_runs": 0.0, "total_bases": 0.0, "rbis": 0.0,
            "stolen_bases": 0.0, "strikeouts": 0.2, "walks": 0.6, "runs": 0.6,
        },
    },
    "keys": {"odds_api": ""},
    "mlb_park_factors": {
        "COL": 1.18, "CIN": 1.07, "BOS": 1.06, "TEX": 1.05, "KC": 1.04,
        "ARI": 1.03, "PHI": 1.03, "BAL": 1.02, "default": 1.00, "SD": 0.95,
        "SEA": 0.95, "SF": 0.94, "NYM": 0.96, "DET": 0.97, "OAK": 0.93,
    },
    "mlb_outdoor_parks": [
        "COL", "BOS", "CHC", "CIN", "KC", "DET", "BAL", "PHI", "PIT", "SD",
        "SF", "NYM", "NYY", "CLE", "ATL", "LAA", "LAD", "OAK", "SEA", "WSH",
        "STL", "TEX",
    ],
    "soccer": {
        "knockout_intensity": 0.94,
        "home_continent_boost": 1.04,
        "short_rest_penalty": 0.97,
    },
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@lru_cache(maxsize=1)
def load(path: str | None = None) -> dict[str, Any]:
    """Load merged config (defaults + optional config.yaml)."""
    cfg = dict(_DEFAULTS)
    p = Path(path) if path else (ROOT / "config.yaml")
    if p.exists():
        try:
            import yaml
            with open(p, "r", encoding="utf-8") as fh:
                cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})
        except Exception as exc:  # pragma: no cover
            print(f"[config] could not parse {p}: {exc}; using defaults")
    return cfg


def db_path() -> str:
    return os.environ.get("PROJECTOR_DB") or load()["paths"]["db"]


def park_factor(team: str | None) -> float:
    pf = load()["mlb_park_factors"]
    return float(pf.get((team or "").upper(), pf.get("default", 1.0)))


def is_outdoor(team: str | None) -> bool:
    return (team or "").upper() in set(load()["mlb_outdoor_parks"])
