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
import books
import valuation
import dataos

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
# history.json. GRADED rows are kept forever by design (prune_clv's docstring: they're the
# only historical data multi-season calibration can be built from) so the ledger grows
# without bound — it reached 107.6 MB / 197k graded rows in August 2026 and every push past
# GitHub's 100 MiB per-file limit was silently rejected for 14+ hours (board/analytics froze
# because the whole publish() step fails atomically, not just the clv.db file), which is
# what actually caused that outage rather than any modeling or data-source bug. Publishing
# gzip-compressed (~5x on this data, see refresh.yml's publish()) buys a long runway without
# deleting anything — it's a wire format change only, never a retention decision.
OUT_CLV = Path(__file__).parent / "static" / "clv.db"
SEED_CLV = Path(__file__).parent / "clv_seed.db"     # one-time bootstrap, committed to the repo
_CLV_GZ_URL = "https://raw.githubusercontent.com/cmohalloran11-cell/juicedlines/data/clv.db.gz"
_CLV_URL = "https://raw.githubusercontent.com/cmohalloran11-cell/juicedlines/data/clv.db"


def _load_prev_clv() -> str:
    """Published ledger → static/clv.db. Falls back to the committed seed on first run."""
    OUT_CLV.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gzip
        import urllib.request
        req = urllib.request.Request(f"{_CLV_GZ_URL}?t={int(time.time())}",
                                     headers={"User-Agent": "juiced-build"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = gzip.decompress(r.read())
        if data[:15] == b"SQLite format 3":        # don't write a 404 page over the ledger
            OUT_CLV.write_bytes(data)
            return "data-branch"
    except Exception:
        pass
    try:                                            # legacy uncompressed copy, one-time migration
        import urllib.request
        req = urllib.request.Request(f"{_CLV_URL}?t={int(time.time())}",
                                     headers={"User-Agent": "juiced-build"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        if data[:15] == b"SQLite format 3":
            OUT_CLV.write_bytes(data)
            return "data-branch (legacy uncompressed)"
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
    "headshot", "team_logo", "flag", "country", "opponent", "is_home",
    "model_proj", "model_edge", "model_prob", "proj_kind", "model_n", "model_median",
    "model_floor", "model_ceiling",     # the p10-p90 range shown under the projection
    "prior_tier_reason",                 # CFB's 3-tier-fallback "why" (cfb.board.attach_cfb) --
                                         # mirrors dashboard.py's _drop, same additive pattern
    "model_raw",                        # pre-anchor model mean — juice_score's model-vs-market
                                         # agreement component (MLB/WNBA; tennis has its own
                                         # 3-way model_agreement below)
    "bball_confidence", "tennis_confidence",
    "surface", "model_agreement", "elo_eff_matches",   # tennis engine transparency fields
    # NFL engine transparency fields (nfl.board.NFL_FIELDS) — mirrors dashboard.py's _drop.
    "season_type", "season_type_confirmed", "expected_snaps", "snap_range",
    "expected_routes", "expected_routes_basis", "expected_targets", "expected_carries",
    "playing_time_confidence", "playing_time_probability", "role_confidence",
    "preseason_risk", "rotation_tier", "role", "depth_chart_position",
    "red_zone_opportunities", "pass_rush_matchup", "game_total", "team_total", "spread",
    "weather", "nfl_confidence", "nfl_confidence_factors",
    "snap_p10", "snap_p90", "snap_std_dev", "prior_influence",   # preseason snap-model diagnostics
    "lineup_status", "lineup_slot",     # the OUT badge + edge/parlay exclusions read these
    "workload_status", "layoff_days", "workload_outs",   # IL badge + "why" tooltip
    "market_book_count",                # juice_score's market-quality component (attach_market_quality)
    "stat_trust_gamma",                 # juice_score's measured-per-stat-trust component (attach_stat_trust)
    # juice_v2's inputs: the PRE-anchor simulated distribution + the total anchor weight left
    # on the model + the lock horizon. All four must survive into the static payload or the
    # static deploy scores a different (null) juice than the live server for the same prop.
    "model_pre_mean", "model_pre_median", "model_pre_sd", "model_pre_prob",
    "model_anchor_t", "minutes_to_lock",
    # 2026-08: projection version metadata (spec Principle 4 reproducibility) -- stamped on
    # every line by provenance.stamp_lines but previously stripped here before reaching any
    # served output, so a user could never actually see which model version produced their
    # pick, only the aggregate ledger could. Free tier too (not in _PREMIUM_FIELDS below) --
    # this is a transparency/trust disclosure, not a competitive edge signal.
    "model_version", "data_snapshot",
)

# The paywall: everything a projection produces is PREMIUM. Stripping these leaves the free
# tier with the live lines only (player/team/stat/line) — a taste, no edges. The free file
# is safe to serve publicly; the premium file must only ever reach authenticated payers
# (Phase 3 routes it through the auth gate instead of the public data branch).
_PREMIUM_FIELDS = frozenset({
    "model_proj", "model_edge", "model_prob", "proj_kind", "prior_tier_reason", "model_n", "model_raw", "model_median",
    "bball_confidence", "tennis_confidence", "model_floor", "model_ceiling",
    "surface", "model_agreement", "elo_eff_matches", "market_book_count", "stat_trust_gamma",
    "model_proj_b", "model_prob_b", "model_proj_c", "model_prob_c",
    "model_pre_mean", "model_pre_median", "model_pre_sd", "model_pre_prob", "model_anchor_t",
    "lineup_slot", "lineup_status", "workload_status", "layoff_days", "workload_outs",
    "season_type", "season_type_confirmed", "expected_snaps", "snap_range",
    "expected_routes", "expected_routes_basis", "expected_targets", "expected_carries",
    "playing_time_confidence", "playing_time_probability", "role_confidence",
    "preseason_risk", "rotation_tier", "role", "depth_chart_position",
    "red_zone_opportunities", "pass_rush_matchup", "game_total", "team_total", "spread",
    "weather", "nfl_confidence", "nfl_confidence_factors",
    "snap_p10", "snap_p90", "snap_std_dev", "prior_influence",
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
    extra, book_errs = books.fetch_extra_books()   # Sleeper (live) + any future books
    errors.update(book_errs)

    lines = ud + pp + extra
    try:
        analytics.enrich_lines(lines)
    except Exception as exc:
        errors["enrich"] = str(exc)
    try:
        analytics.attach_game_ids(lines)   # lets the parlay tab detect correlated legs
    except Exception as exc:
        errors["game_ids"] = str(exc)
    try:
        errors.update(analytics.attach_projections(lines))
    except Exception as exc:
        errors["projections"] = str(exc)
    try:
        analytics.attach_market_quality(lines)   # juice_score's cross-book coverage signal
    except Exception as exc:
        errors["market_quality"] = str(exc)
    try:
        analytics.attach_stat_trust(lines)   # juice_score's measured-per-stat-trust signal
    except Exception as exc:
        errors["stat_trust"] = str(exc)
    try:
        analytics.attach_lock_clock(lines)   # juice_v2's near-lock availability cap
    except Exception as exc:
        errors["lock_clock"] = str(exc)

    # Direction-invariant validation (2026-07-29 Over/Under bias audit): every exported
    # projection must satisfy Projection > Line ⇒ P(Over) > 50% (and the reverse). The engines
    # now report the MEDIAN of the same sample array model_prob comes from specifically to
    # GUARANTEE this (see dataos.direction_report's docstring) — this is the runtime safety
    # net, and reject=True means any violation found is actually dropped from the board
    # (dashboard._projected() already requires model_proj/model_prob, so nulling them here is
    # enough), not just logged. Should print ~0 violations; any it finds are worth investigating.
    dreport: dict = {}
    try:
        dreport = dataos.direction_report(lines, reject=True)
        dist = dreport["distribution"]
        print(f"::group::Direction check — {dreport['violations']}/{dreport['checked']} "
              f"violated the Projection/Probability invariant (rejected) | board: "
              f"{dist['over']} over ({dist['pct_over']}%) / {dist['under']} under ({dist['pct_under']}%)")
        for b in dreport["violations_by_sport_and_kind"].items():
            print(f"  {b[0]}: {b[1]}")
        for v in dreport["sample_violations"]:
            print(f"  REJECTED  {v['player']} — {v['stat']} ({v['source']}/{v['proj_kind']}) "
                  f"line {v['line']}, proj {v['projection']}, prob_over {v['prob_over']}")
        print("::endgroup::")
    except Exception as exc:
        errors["direction_audit"] = str(exc)

    # EV quality safeguard (Edge/EV audit): log any USER-FACING projection whose EV
    # exceeds EV_REVIEW_THRESHOLD (env, default 60% — see valuation.py for why not 15%).
    # Scoped to standard/boosted odds_type ONLY, matching dashboard._projected()'s own
    # filter — demon/goblin legs are unpriced (no real payout multiplier exposed by the
    # feed) and already excluded from everything a user can see, so auditing them just
    # buries the real signal in noise nobody will ever act on. Best-effort; never blocks.
    try:
        std = [l for l in lines if (l.get("odds_type") or "standard").lower() in ("standard", "boosted")]
        flagged = [f for f in (valuation.audit_ev(l) for l in std) if f]
        if flagged:
            print(f"::group::EV review — {len(flagged)} projections above "
                  f"{valuation.EV_REVIEW_THRESHOLD:.0%}")
            for f in sorted(flagged, key=lambda x: x["ev"], reverse=True)[:25]:
                print(f"  {f['ev']:+.1%}  {f['player']} — {f['stat']} ({f['source']}) "
                      f"{f['side']} {f['line']}, proj {f['projection']}, "
                      f"model_p={f['model_prob_for_side']:.3f}"
                      f"{' [pickem fallback]' if f['used_pickem_fallback'] else ''}")
            if len(flagged) > 25:
                print(f"  … and {len(flagged) - 25} more")
            print("::endgroup::")
    except Exception as exc:
        errors["ev_audit"] = str(exc)

    # Juice Score coherence review queue. A fault means the engine's own P(over) and its
    # median-vs-line displacement point opposite ways by more than the distribution's skew can
    # explain — an engine contradicting itself, not a weak prop. juice_v2 already nulls these
    # and dashboard._projected() drops them; this is where they get seen by a human. Runs
    # regardless of JUICE_VERSION, because the integrity check is worth having even while the
    # score itself is flag-gated off.
    try:
        faults = valuation.audit_juice_coherence(lines)
        if faults:
            print(f"::group::Juice coherence — {len(faults)} model-integrity faults "
                  f"(excluded from the board)")
            for f in faults[:25]:
                print(f"  {f['sport']}/{f['proj_kind']}  {f['player']} — {f['stat_type']} "
                      f"line {f['line']}: p={f['p']:.3f} vs b={f['b']:.3f} (e={f['e']:+.3f}) "
                      f"but median sits z={f['z']:+.2f} SD away, skew g={f['g']:+.2f}")
            if len(faults) > 25:
                print(f"  … and {len(faults) - 25} more")
            print("::endgroup::")
            errors["juice_coherence_faults"] = f"{len(faults)} lines excluded (model integrity)"
    except Exception as exc:
        errors["juice_coherence"] = str(exc)

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
        try:
            graded_tn = analytics.grade_tennis()          # Tennis (ESPN match results)
        except Exception as exc:
            graded_tn = {"graded": 0, "voided": 0, "err": str(exc)[:40]}
        pruned = db.prune_clv(keep_ungraded_days=3)
        import sqlite3 as _sq
        _c = _sq.connect(OUT_CLV)
        n_all = _c.execute("SELECT COUNT(*) FROM prop_clv").fetchone()[0]
        n_grd = _c.execute("SELECT COUNT(*) FROM prop_clv WHERE actual IS NOT NULL").fetchone()[0]
        _c.close()
        print(f"  wrote {OUT_CLV.name} [{src}]: logged {logged}, graded MLB {graded} + "
              f"BB {graded_bb} + Tennis {graded_tn}, pruned {pruned} | {n_grd} graded / "
              f"{n_all} rows, {OUT_CLV.stat().st_size/1e6:.1f} MB | +{time.time()-tc:.0f}s")
        # Per-stat trust (γ) for the anchoring layer — computed here where the graded ledger is
        # local, published as a TINY file that every build (incl. fast) reads so it never has to
        # re-download the multi-MB ledger just to anchor. NEXT build's attach_projections uses it.
        try:
            trust = {"MLB": db.stat_gammas("MLB", min_n=120),
                     "prob_cal": {"MLB": db.prob_calibration("MLB")},   # honest P(over)
                     # honest floor/ceiling: our p10-p90 band only held 56% of outcomes, not 80%
                     "width": {"MLB": db.interval_width("MLB")},
                     "updated_at": updated}
            OUT_TRUST.write_text(json.dumps(trust, separators=(",", ":")), encoding="utf-8")
            print(f"  wrote {OUT_TRUST.name}: {len(trust['MLB'])} stats trusted, "
                  f"prob_cal {trust['prob_cal']['MLB'] or 'none'}, "
                  f"interval width x{trust['width']['MLB']}")
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

    # ── Juiced 2.0 dashboard JSON — the new SPA reads these on the static deploy ──
    # Reuses the same modules the live API uses, so the static site serves the full
    # dashboard/projections/leaderboards/weather with no backend. db.DB_PATH was pointed
    # at the ledger (OUT_CLV) in the CLV block above, so the ledger-derived files are
    # correct on full builds; on FAST cycles they're left to the previous build.
    try:
        import dashboard as _dash
        import weather as _wx
        SD = OUT.parent

        def _w(name, obj):
            (SD / name).write_text(json.dumps(obj, separators=(",", ":"), default=_num),
                                   encoding="utf-8")

        # "Read every line" — limit high enough to cover the full priced (standard/boosted)
        # pool (measured ~4-5k on a full slate), not an arbitrary top-1000 cut that silently
        # dropped thousands of real lines.
        _w("projections.json", {"projections": _dash.projections(lines, limit=8000),
                                "updated_at": updated})
        # Demon/goblin lane, separate from the default (priced) projections payload — see
        # dashboard.projections()'s odds_types param. Kept in its own file so it doesn't
        # dominate the juice-sorted default list; the frontend lazy-loads this only when the
        # user opts into the PrizePicks Demon/Goblin toggle. Limit covers the full demon+
        # goblin pool (measured ~10-11k on a full slate) — "every line" applies here too.
        _w("boosted.json", {"projections": _dash.projections(
                                lines, limit=15000, sort="confidence",
                                odds_types=("demon", "goblin")),
                            "updated_at": updated})
        _w("injuries.json", {"injuries": _dash.injuries(lines)})
        _w("weather.json", _wx.slate(lines))
        _w("books.json", {"books": books.status()})
        _w("auth.json", {"auth_configured": bool(os.getenv("SUPABASE_URL")),
                         "supabase_url": os.getenv("SUPABASE_URL"),
                         "supabase_anon_key": os.getenv("SUPABASE_ANON_KEY"),
                         "ai": False, "dev_mode": False, "static": True})
        _w("dashboard.json", _dash.build(lines, updated, errors))
        # Diagnostics deliverable for the Over/Under bias audit — violation count, root cause
        # by sport/engine, and the before-rejection Over/Under distribution. Also the data
        # source for the planned admin Model Bias Monitor.
        _w("direction_report.json", {**dreport, "updated_at": updated})
        # Entry Optimizer (optimizer.py): today's-best combinations off the CURRENT board.
        # Runs every cycle (FAST included), not gated behind full-build-only like
        # model_health/backtest below — those are slow, day-level "how's the model doing"
        # metrics that don't need sub-hourly freshness; this is "what should I bet right
        # now", which is exactly as time-sensitive as projections.json/dashboard.json above.
        # Measured ~5s over a 3,700-prop slate (2026-08-04) — well inside the FAST cycle's
        # ~100s budget, nothing like the 331s analytics.json step the FAST/FULL split exists
        # for. Static deploy has no live /api/optimizer/today, so this file IS that endpoint
        # for the static SPA (see dashboard.html's staticApi()).
        import optimizer as _opt
        _w("optimizer.json", _opt.today_report(lines))
        _w("optimizer_books.json", {"books": [{"id": b, "available": bool(_opt.BOOK_PAYOUTS.get(b))}
                                              for b in _opt.SUPPORTED_BOOKS]})
        # Per-sport files (PrizePicks only — the only book with a real published payout
        # table) so the static site's sport filter changes what's actually shown, not just
        # the label — the live /api/optimizer/today would recompute this per-request, but
        # the static deploy has no live backend to re-query with different params.
        for _sp in ("MLB", "WNBA", "Tennis"):
            _w(f"optimizer_{_sp.lower()}.json", _opt.today_report(lines, sport=_sp))
        # Per-book files for books with no verified payout table yet (optimizer.BOOK_PAYOUTS)
        # — today_report's result is identical regardless of sport for these (always the
        # same unavailableReason), so one file per book covers every sport selection.
        for _bk in ("underdog", "sleeper"):
            _w(f"optimizer_{_bk}.json", _opt.today_report(lines, book=_bk))
        extra_json = ""
        if not FAST:
            import model_health as _mh
            import backtest as _bt
            _w("model_health.json", _mh.health())
            _w("backtest.json", {"current": _bt.current_accuracy("MLB")})
            _w("drift.json", _bt.drift("MLB"))
            extra_json = "/model_health/backtest/drift"
        print(f"  wrote SPA JSON: projections/dashboard/injuries/weather/books/auth/optimizer{extra_json}")
    except Exception as exc:
        print(f"  dashboard SPA JSON SKIPPED ({exc})")


if __name__ == "__main__":
    main()
