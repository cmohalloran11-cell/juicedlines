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

# Full board (with projections/edges). Keeps the name `board.json` so nothing breaks today;
# it is the PREMIUM payload and Phase 3 routes it through the auth gate instead of the public
# data branch. `board.free.json` is the free tier — live lines only, safe to serve publicly.
OUT = Path(__file__).parent / "static" / "board.json"
OUT_FREE = Path(__file__).parent / "static" / "board.free.json"
# Pre-computed research-drawer analytics (recent games, hit-rate, matchup) so the STATIC
# deploy can show the historical drawer without a live backend. Keyed by sport|player|stat.
OUT_ANALYTICS = Path(__file__).parent / "static" / "analytics.json"

# Rolling line-movement history. The live server keeps this in SQLite, but the Action is
# STATELESS (fresh checkout every run) — so the published file on the `data` branch IS the
# store: each build reads the previous one, appends today's values, and republishes it.
OUT_HISTORY = Path(__file__).parent / "static" / "history.json"
_HISTORY_URL = "https://raw.githubusercontent.com/cmohalloran11-cell/juicedlines/data/history.json"
_HIST_MAX_POINTS = 40          # per line; plenty for a movement chart, keeps the file small


def _load_prev_history() -> dict:
    """Previous rolling history from the data branch. Best-effort: first run starts empty."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{_HISTORY_URL}?t={int(time.time())}",
                                     headers={"User-Agent": "juiced-build"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return (json.load(r) or {}).get("history") or {}
    except Exception:
        return {}

# Fields the frontend actually uses (keeps the file small vs dumping every field).
_KEEP = (
    "id", "source", "sport", "player", "team", "position", "stat_type", "line",
    "odds_type", "matchup", "start_time", "status", "game_id",
    "over_price", "under_price", "over_implied", "under_implied", "pickem_price",
    "headshot", "team_logo", "flag", "country",
    "model_proj", "model_edge", "model_prob", "proj_kind", "model_n",
    "bball_confidence", "tennis_confidence",
)

# The paywall: everything a projection produces is PREMIUM. Stripping these leaves the free
# tier with the live lines only (player/team/stat/line) — a taste, no edges. The free file
# is safe to serve publicly; the premium file must only ever reach authenticated payers
# (Phase 3 routes it through the auth gate instead of the public data branch).
_PREMIUM_FIELDS = frozenset({
    "model_proj", "model_edge", "model_prob", "proj_kind", "model_n",
    "bball_confidence", "tennis_confidence", "model_floor", "model_ceiling",
    "model_proj_b", "model_prob_b",
})


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
        analytics.attach_game_ids(lines)   # lets the parlay tab detect correlated legs
    except Exception as exc:
        errors["game_ids"] = str(exc)
    try:
        analytics.attach_projections(lines)
    except Exception as exc:
        errors["projections"] = str(exc)

    slim = [{k: l[k] for k in _KEEP if l.get(k) is not None} for l in lines]
    updated = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _payload(rows, tier):
        return {"lines": rows, "updated_at": updated, "errors": errors,
                "static": True, "tier": tier}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # premium: the full board (with projections/edges) — auth-gated in production
    OUT.write_text(json.dumps(_payload(slim, "premium"), separators=(",", ":"), default=_num),
                   encoding="utf-8")
    # free: strip every projection-derived field → live lines only, safe to serve publicly
    free = [{k: v for k, v in row.items() if k not in _PREMIUM_FIELDS} for row in slim]
    OUT_FREE.write_text(json.dumps(_payload(free, "free"), separators=(",", ":"), default=_num),
                        encoding="utf-8")

    from collections import Counter
    by = Counter(l["sport"] for l in slim)
    print(f"wrote {OUT.name}: {len(slim)} lines, {OUT.stat().st_size/1e6:.1f} MB "
          f"| {OUT_FREE.name}: {OUT_FREE.stat().st_size/1e6:.1f} MB (lines only) | {time.time()-t0:.0f}s")
    print(f"  by sport: {dict(by)}  errors: {list(errors)}")

    # ── line-movement history (rolling; the data branch is the store) ────────────
    # A point is appended only when a line actually MOVES — so a line that never budges
    # keeps one seed point (chart correctly hides) and the file stays tiny. Stale ids drop
    # out naturally because we rebuild from the CURRENT lines each run.
    try:
        th = time.time()
        prev = _load_prev_history()
        hist: dict[str, list] = {}
        for l in lines:
            lid, lv = l.get("id"), l.get("line")
            if not lid or lv is None:
                continue
            pts = prev.get(lid) or []
            if not pts or pts[-1].get("line_value") != lv:
                pts = pts + [{"ts": updated, "line_value": lv}]
            hist[lid] = pts[-_HIST_MAX_POINTS:]
        OUT_HISTORY.write_text(
            json.dumps({"history": hist, "updated_at": updated}, separators=(",", ":"), default=_num),
            encoding="utf-8")
        movers = sum(1 for v in hist.values() if len(v) > 1)
        print(f"  wrote {OUT_HISTORY.name}: {len(hist)} lines ({movers} moved), "
              f"{OUT_HISTORY.stat().st_size/1e6:.2f} MB | +{time.time()-th:.0f}s")
    except Exception as exc:
        print(f"  history.json SKIPPED ({exc})")

    # ── research-drawer analytics (best-effort; never blocks the board) ──────────
    # Keyed by (sport, player, stat, LINE) — the LINE is part of the key because analyze()
    # computes everything against it: hit-rate, each recent game's cleared ✓/✗, P(over) and
    # proj−line. Keying by (player, stat) alone and analyzing one representative line made a
    # demon/alt line's drawer show the STANDARD line's card (e.g. a 1.5 demon rendering
    # "Line 4.5"). Lines that differ only by book/odds_type share a key, so one analyze()
    # still covers them. Emit a line_id → key index so the frontend never has to re-derive
    # the key (float formatting differs: python "2" vs js "2"). The pipeline already warmed
    # the game-log caches, so extra line variants are cheap. Line-movement lives in
    # history.json. Premium payload; gated with board.json.
    try:
        ta = time.time()
        groups: dict[str, list] = {}
        for l in lines:
            if l.get("line") is None:
                continue
            k = (f"{l.get('sport')}|{analytics._norm(l.get('player') or '')}"
                 f"|{l.get('stat_type') or ''}|{l.get('line')}")
            groups.setdefault(k, []).append(l)
        amap: dict[str, dict] = {}
        index: dict[str, str] = {}
        # `recent` is 62% of the payload and is IDENTICAL across a prop's line variants
        # (only each game's `cleared` ✓/✗ depends on the line) — so store it ONCE per
        # (sport, player, stat) and let the drawer recompute `cleared` against the line it
        # was opened with. Halves the file without losing anything.
        rmap: dict[str, list] = {}
        for k, gl in groups.items():
            try:
                a = analytics.analyze(gl[0])      # same player+stat+line → same analytics
            except Exception:
                a = None
            if not (a and a.get("available")):
                continue
            rec = a.pop("recent", None)
            if rec:
                rk = "|".join(k.split("|")[:3])   # sport|player|stat
                if rk not in rmap:
                    for g in rec:
                        g.pop("cleared", None)    # per-line → recomputed client-side
                    rmap[rk] = rec
                a["_r"] = rk
            amap[k] = a
            for l in gl:
                if l.get("id"):
                    index[l["id"]] = k
        OUT_ANALYTICS.write_text(
            json.dumps({"analytics": amap, "recent": rmap, "index": index, "updated_at": updated},
                       separators=(",", ":"), default=_num),
            encoding="utf-8")
        print(f"  wrote {OUT_ANALYTICS.name}: {len(amap)}/{len(groups)} groups, "
              f"{len(rmap)} recent-tables (deduped), {len(index)} lines indexed, "
              f"{OUT_ANALYTICS.stat().st_size/1e6:.1f} MB | +{time.time()-ta:.0f}s")
    except Exception as exc:
        print(f"  analytics.json SKIPPED ({exc})")


if __name__ == "__main__":
    main()
