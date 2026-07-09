"""Config loader — reads wc/config.yaml (falls back to built-in defaults), and
lets env vars override API keys. Tune model weights in the YAML, not the code."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_DEFAULTS = {
    "sources": {"fixtures": "sample", "strength": "sample", "players": "sample", "odds": "sample"},
    "keys": {"api_football": "", "odds_api": ""},
    "model": {"n_sims": 10000, "league_avg_goals": 1.35, "home_advantage": 1.25,
              "recency_decay": 0.55, "dc_rho": -0.05, "max_goals": 10, "tempo_ref_goals": 2.7},
    "intensity": {"knockout": 1.15, "rivalry": 1.10},
    "confidence": {"confirmed": 0.85, "probable": 0.55},
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base.get(k, {}), v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


@lru_cache(maxsize=1)
def load() -> dict:
    cfg = dict(_DEFAULTS)
    path = Path(__file__).parent / "config.yaml"
    if path.exists():
        try:
            import yaml
            cfg = _merge(_DEFAULTS, yaml.safe_load(path.read_text(encoding="utf-8")) or {})
        except Exception as exc:  # pragma: no cover
            print(f"[wc.config] using defaults ({exc})")
    # env overrides for keys
    cfg["keys"]["api_football"] = os.getenv("API_FOOTBALL_KEY") or cfg["keys"]["api_football"]
    cfg["keys"]["odds_api"] = os.getenv("ODDS_API_KEY") or cfg["keys"]["odds_api"]
    return cfg


def m(key: str, default=None):
    """Shorthand for a model weight."""
    return load()["model"].get(key, default)
