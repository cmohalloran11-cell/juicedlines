"""
fantasy.projections — the swappable projections seam.

get_provider() is the ONLY supported way to obtain a ProjectionsProvider. Nothing outside this
package may import a concrete adapter (e.g. nflverse_adapter) directly -- that's what keeps a
future commercial-provider swap (licensing still TBD) a one-file change: implement
ProjectionsProvider in a new adapter module, register it in _PROVIDERS below, done.
"""
from __future__ import annotations

import importlib
import os

from .base import ProjectionsProvider, CANONICAL_STAT_KEYS

_PROVIDERS: dict[str, str] = {
    "nflverse": "fantasy.projections.nflverse_adapter:NflverseProjectionsProvider",
}


def get_provider(name: str | None = None) -> ProjectionsProvider:
    """Resolve the active ProjectionsProvider. Defaults to the FANTASY_PROJECTIONS_PROVIDER
    env var, then 'nflverse' (the only adapter implemented so far)."""
    key = (name or os.getenv("FANTASY_PROJECTIONS_PROVIDER") or "nflverse").lower()
    path = _PROVIDERS.get(key)
    if not path:
        raise ValueError(f"Unknown projections provider '{key}'. Available: {sorted(_PROVIDERS)}")
    module_path, _, cls_name = path.partition(":")
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)()


__all__ = ["ProjectionsProvider", "CANONICAL_STAT_KEYS", "get_provider"]
