"""
Build the static board for the free/no-server deploy.

Runs the same pipeline as the live server's `refresh_lines` (pull all books →
enrich → attach projections), but writes the result to `static/board.json` instead
of an in-memory cache. A GitHub Action runs this on a schedule; the site is then
just static files (index.html + board.json), so it needs no running backend.

Only the fields the frontend reads are kept, to keep the file small.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pullers
import analytics

OUT = Path(__file__).parent / "static" / "board.json"

# Fields the frontend actually uses (keeps board.json small vs dumping every field).
_KEEP = (
    "id", "source", "sport", "player", "team", "position", "stat_type", "line",
    "odds_type", "matchup", "start_time", "status",
    "over_price", "under_price", "over_implied", "under_implied", "pickem_price",
    "headshot", "team_logo", "flag", "country",
    "model_proj", "model_edge", "model_prob", "proj_kind", "model_n",
    "bball_confidence", "tennis_confidence",
)


def _num(o):
    # numpy scalars (np.float64/int64) aren't JSON-serialisable by default.
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(repr(o))


def main() -> None:
    t0 = time.time()
    errors: dict[str, str] = {}

    ud, uerr = pullers.fetch_underdog()
    if uerr:
        errors["underdog"] = uerr
    pp, perr = pullers.fetch_prizepicks()
    if perr:
        errors["prizepicks"] = perr

    lines = ud + pp
    try:
        analytics.enrich_lines(lines)
    except Exception as exc:
        errors["enrich"] = str(exc)
    try:
        analytics.attach_projections(lines)
    except Exception as exc:
        errors["projections"] = str(exc)

    slim = [{k: l[k] for k in _KEEP if l.get(k) is not None} for l in lines]
    payload = {
        "lines": slim,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "errors": errors,
        "static": True,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":"), default=_num), encoding="utf-8")

    from collections import Counter
    by = Counter(l["sport"] for l in slim)
    mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT.name}: {len(slim)} lines, {mb:.1f} MB, {time.time()-t0:.0f}s")
    print(f"  by sport: {dict(by)}  errors: {list(errors)}")


if __name__ == "__main__":
    main()
