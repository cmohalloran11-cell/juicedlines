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
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pullers
import analytics

# FAST refresh: rebuild only what actually changes minute-to-minute — the LINES (board.json)
# and their movement (history.json). Measured 2026-07-19, a full build is 433s of which
# analytics.json alone is 331s (77%); the drawer analytics are derived from game logs that
# update DAILY, and the CLV ledger upserts today's props, so neither needs a 5-minute cadence.
# Skipping both drops a cycle to ~100s, which is what makes a real 5-minute refresh possible.
# The refresh workflow runs one FULL build per hour and fast cycles in between; the previous
# analytics.json/clv.db stay on disk and are republished unchanged.
FAST = os.environ.get("FAST_REFRESH") == "1"

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
# Per-stat trust weights (γ) for the anchoring layer — tiny, published each full build.
OUT_TRUST = Path(__file__).parent / "static" / "trust.json"
_HISTORY_URL = "https://raw.githubusercontent.com/cmohalloran11-cell/juicedlines/data/history.json"
_HIST_MAX_POINTS = 40          # per line; plenty for a movement chart, keeps the file small

# ── CLV ledger (the research asset) ──────────────────────────────────────────────
# This is what answers "does our model beat the line" — the edge regression
# (y−L)=a+γ(m−L) runs on its GRADED rows. It used to be written ONLY by the live server's
# snapshot loop (main.py), so it grew only while someone happened to be running uvicorn —
# and props not logged on the day are gone forever. Now the Action maintains it, using the
# same data-branch-as-store trick as history.json.
#
# NOTE this is prop_clv ONLY. The local history.db is 1.8 GB, but that's almost entirely
# line_history (9.6M rows) which the static build doesn't need — line movement is served by
# history.json. The ledger alone is 22.5 MB, and ~0.9 MB once pruned to graded rows.
OUT_CLV = Path(__file__).parent / "static" / "clv.db"
SEED_CLV = Path(__file__).parent / "clv_seed.db"     # one-time bootstrap, committed to the repo
_CLV_URL = "https://raw.githubusercontent.com/cmohalloran11-cell/juicedlines/data/clv.db"


def _load_prev_clv() -> str:
    """Published ledger → static/clv.db. Falls back to the committed seed on first run."""
    OUT_CLV.parent.mkdir(parents=True, exist_ok=True)
    try:
        import urllib.request
        req = urllib.request.Request(f"{_CLV_URL}?t={int(time.time())}",
                                     headers={"User-Agent": "juiced-build"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        if data[:15] == b"SQLite format 3":        # don't write a 404 page over the ledger
            OUT_CLV.write_bytes(data)
            return "data-branch"
    except Exception:
        pass
    if SEED_CLV.exists():                          # first run: bootstrap from the seed
        import shutil
        shutil.copyfile(SEED_CLV, OUT_CLV)
        return "seed"
    return "empty"


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
    "lineup_status", "lineup_slot",     # the OUT badge + edge/parlay exclusions read these
    "workload_status", "layoff_days", "workload_outs",   # IL badge + "why" tooltip
)

# The paywall: everything a projection produces is PREMIUM. Stripping these leaves the free
# tier with the live lines only (player/team/stat/line) — a taste, no edges. The free file
# is safe to serve publicly; the premium file must only ever reach authenticated payers
# (Phase 3 routes it through the auth gate instead of the public data branch).
_PREMIUM_FIELDS = frozenset({
    "model_proj", "model_edge", "model_prob", "proj_kind", "model_n",
    "bball_confidence", "tennis_confidence", "model_floor", "model_ceiling",
    "model_proj_b", "model_prob_b", "model_proj_c", "model_prob_c",
    "lineup_slot", "lineup_status", "workload_status", "layoff_days", "workload_outs",
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

    # Per-game meta (probable pitchers + confirmed/projected lineups) for the game-detail view.
    # Cheap on FAST cycles — the today-schedule call is cached and the recent-lineup projection
    # is cached 3h — so it stays in the 5-minute path. Best-effort; never blocks the board.
    games_meta: dict = {}
    try:
        games_meta = analytics.mlb_game_meta()
    except Exception as exc:
        errors["game_meta"] = str(exc)[:40]
    try:
        games_meta.update(analytics.wnba_game_meta())    # WNBA projected starters
    except Exception as exc:
        errors["wnba_meta"] = str(exc)[:40]

    def _payload(rows, tier):
        return {"lines": rows, "updated_at": updated, "errors": errors,
                "static": True, "tier": tier, "games": games_meta}

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

    # ── CLV ledger: log today's props + grade yesterday's (never blocks the board) ──
    # Point db at the ledger-only file, NOT the 1.8GB local history.db, and never call
    # snapshot_lines here (that's what makes history.db huge; movement lives in history.json).
    try:
        if FAST:                       # reuse this block's own best-effort skip path
            raise RuntimeError("fast refresh")
        tc = time.time()
        import db
        db.DB_PATH = OUT_CLV
        src = _load_prev_clv()
        db.init_db()
        logged = db.log_clv(lines, updated)
        try:
            graded = analytics.grade_pending()          # MLB (statsapi game logs)
        except Exception as exc:
            graded = {"graded": 0, "voided": 0, "err": str(exc)[:40]}
        try:
            graded_bb = analytics.grade_basketball()     # WNBA + SL (ESPN box scores)
        except Exception as exc:
            graded_bb = {"graded": 0, "voided": 0, "err": str(exc)[:40]}
        pruned = db.prune_clv(keep_ungraded_days=3)
        import sqlite3 as _sq
        _c = _sq.connect(OUT_CLV)
        n_all = _c.execute("SELECT COUNT(*) FROM prop_clv").fetchone()[0]
        n_grd = _c.execute("SELECT COUNT(*) FROM prop_clv WHERE actual IS NOT NULL").fetchone()[0]
        _c.close()
        print(f"  wrote {OUT_CLV.name} [{src}]: logged {logged}, graded MLB {graded} + "
              f"BB {graded_bb}, pruned {pruned} | {n_grd} graded / {n_all} rows, "
              f"{OUT_CLV.stat().st_size/1e6:.1f} MB | +{time.time()-tc:.0f}s")
        # Per-stat trust (γ) for the anchoring layer — computed here where the graded ledger is
        # local, published as a TINY file that every build (incl. fast) reads so it never has to
        # re-download the multi-MB ledger just to anchor. NEXT build's attach_projections uses it.
        try:
            trust = {"MLB": db.stat_gammas("MLB", min_n=120),
                     "prob_cal": {"MLB": db.prob_calibration("MLB")},   # honest P(over)
                     "updated_at": updated}
            OUT_TRUST.write_text(json.dumps(trust, separators=(",", ":")), encoding="utf-8")
            print(f"  wrote {OUT_TRUST.name}: {len(trust['MLB'])} stats trusted, "
                  f"prob_cal {trust['prob_cal']['MLB'] or 'none'}")
        except Exception as exc:
            print(f"  trust.json SKIPPED ({exc})")
    except Exception as exc:
        print(f"  clv.db SKIPPED ({exc})")

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
        if FAST:                       # 331s of a 433s build — daily-cadence data, not 5-min
            raise RuntimeError("fast refresh")
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
