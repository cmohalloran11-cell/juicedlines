"""
backtest.py — the measurement ruler for projection models (spec Vol V Part 4).

Two capabilities:

  1. current_accuracy(sport) — how the LIVE model has actually done, read straight from the
     graded CLV ledger (recorded projection vs the real outcome). No re-run: this is the
     baseline every change must beat. Per-stat + overall MAE / RMSE / bias / hit-rate vs the
     line / probability calibration (ECE) / interval coverage.

  2. replay(sport, ...) — re-runs a projection model on historical prop-days, LEAKAGE-FREE
     (game logs are filtered to strictly before each prop's date), then scores it the same
     way. Run it with the current code to sanity-check the harness matches the ledger; run it
     again after a model change to get a measured before/after on the SAME prop-days.

compare(baseline, candidate) diffs the two so an improvement is provable, not hoped for.
"""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Callable, Optional

import db
import provenance

_TARGET_COVERAGE = 0.80


# ── ledger access ─────────────────────────────────────────────────────────────

def _ledger_rows(sport: str, stat: Optional[str] = None,
                 model_version: str | None = None, all_versions: bool = False,
                 proj_kind: str | None = None) -> list[dict]:
    """Graded, standard-odds prop rows, deduped to one per (player, stat, game_date).

    Scoped to `model_version` by default (current version for this sport, per
    provenance.model_version) — mirrors db.py's stat_gammas/prob_calibration/etc scoping.
    A deliberate math change to the engine (e.g. the 2026-08 two-stage uncertainty
    propagation, or the earlier MLB engine-crash fix) makes every accuracy/drift number
    computed here attributable to the CURRENT model, not a blend of pre- and post-change
    eras that would mask a regression or fake an improvement. Pass `all_versions=True`
    explicitly (e.g. for a long-horizon registry view) to intentionally pool across
    versions; a bare, unscoped read of the full ledger is never the default.

    `proj_kind` (default None = no filter) additionally scopes to one proj_kind, e.g.
    "nfl_regular" vs "nfl_preseason" — two engines that share `sport`="NFL" and the same
    `model_version` but must not have their measured accuracy pooled by default (see
    db.py's stat_gammas docstring for the same reasoning)."""
    q = ("SELECT player, stat_type, game_date, close_line AS L, close_proj AS P, "
         "actual AS Y, COALESCE(model_raw_prob, close_prob) AS RP, "
         "model_floor AS FL, model_ceiling AS CE, proj_kind AS PK, source AS SRC, "
         "model_version AS MV, model_n AS N "
         "FROM prop_clv WHERE sport=? AND actual IS NOT NULL AND close_line IS NOT NULL "
         "AND close_proj IS NOT NULL "
         "AND (odds_type IS NULL OR LOWER(odds_type) IN ('standard','boosted'))")
    args: list[Any] = [sport]
    if stat:
        q += " AND LOWER(stat_type)=?"
        args.append(stat.lower())
    if not all_versions:
        mv = model_version if model_version is not None else provenance.model_version(sport)
        q += " AND model_version=?"
        args.append(mv)
    if proj_kind is not None:
        q += " AND proj_kind=?"
        args.append(proj_kind)
    with db._lock, db._conn() as c:
        rows = [dict(r) for r in c.execute(q, args).fetchall()]
    seen: dict = {}
    for r in rows:
        seen.setdefault((r["player"], r["stat_type"], r["game_date"]), r)
    return list(seen.values())


# ── metrics ───────────────────────────────────────────────────────────────────

def _prob_outcome_pairs(P: list[dict]) -> list[tuple[float, float]]:
    """(predicted P(over), realized outcome 0/1) for every DECIDED row with a probability —
    the shared input to ECE, Brier, and the reliability diagram, so all three are always
    computed from the exact same population."""
    return [(p["RP"], 1.0 if p["Y"] > p["L"] else 0.0) for p in P
            if p.get("RP") is not None and p.get("L") is not None and abs(p["Y"] - p["L"]) > 1e-9]


def reliability_buckets(probs: list[tuple[float, float]], n_buckets: int = 10) -> list[dict]:
    """Bucket (predicted, outcome) pairs into `n_buckets` equal-width probability bins —
    the data behind a reliability diagram (x = predicted P(over), y = realized frequency).
    A perfectly calibrated model has every bucket's predicted/realized on the diagonal."""
    bins: list[list] = [[0.0, 0.0, 0] for _ in range(n_buckets)]
    for pr, o in probs:
        b = min(n_buckets - 1, int(pr * n_buckets))
        bins[b][0] += pr
        bins[b][1] += o
        bins[b][2] += 1
    out = []
    for i, (s, o, cnt) in enumerate(bins):
        lo, hi = i / n_buckets, (i + 1) / n_buckets
        out.append({
            "bucket": f"{lo:.1f}-{hi:.1f}", "n": cnt,
            "predicted": round(s / cnt, 4) if cnt else None,
            "actual": round(o / cnt, 4) if cnt else None,
        })
    return out


def metrics(pairs: list[dict]) -> dict:
    """pairs: dicts with L(line) P(projection) Y(actual) and optional RP(prob) FL/CE(band)."""
    P = [p for p in pairs if p.get("P") is not None and p.get("Y") is not None]
    n = len(P)
    if not n:
        return {"n": 0}
    ae = [abs(p["P"] - p["Y"]) for p in P]
    se = [(p["P"] - p["Y"]) ** 2 for p in P]
    bias = sum(p["P"] - p["Y"] for p in P) / n
    # hit-rate vs the line (pushes excluded): does the model's ACTUAL pick beat the line?
    # The product picks its side by P(over) vs 0.5, not by mean-vs-line — and for right-skewed
    # count stats (Total Bases, H+R+RBI) those disagree ~half the time (mean above the line but
    # P(over) < 0.5 → pick under). Score the real pick: prob-based when available, else mean-based.
    def _picks_over(p):
        rp = p.get("RP")
        return (rp > 0.5) if rp is not None else (p["P"] > p["L"])
    dec = [p for p in P if p.get("L") is not None and abs(p["Y"] - p["L"]) > 1e-9]
    hit = sum(1 for p in dec if _picks_over(p) == (p["Y"] > p["L"]))
    out = {
        "n": n,
        "mae": round(sum(ae) / n, 4),
        "rmse": round(math.sqrt(sum(se) / n), 4),
        "bias": round(bias, 4),
        "decided": len(dec),
        "hit_rate": round(hit / len(dec), 4) if dec else None,
    }
    # probability calibration: ECE (mean calibration gap) + Brier score (proper scoring
    # rule — mean squared error of the probability itself, rewards BOTH calibration and
    # sharpness/decisiveness, unlike ECE which only checks calibration). 0 = perfect,
    # 0.25 = the "always guess 50%" floor, 1.0 = maximally, confidently wrong.
    probs = _prob_outcome_pairs(P)
    if len(probs) >= 50:
        buckets = reliability_buckets(probs)
        ece = sum(abs(b["predicted"] - b["actual"]) * b["n"] for b in buckets if b["n"]) / len(probs)
        out["ece"] = round(ece, 4)
        out["brier"] = round(sum((pr - o) ** 2 for pr, o in probs) / len(probs), 4)
    # interval coverage (p10–p90 band should hold ~80%)
    band = [p for p in P if p.get("FL") is not None and p.get("CE") is not None and p["CE"] > p["FL"]]
    if len(band) >= 50:
        out["coverage"] = round(sum(1 for p in band if p["FL"] <= p["Y"] <= p["CE"]) / len(band), 4)
    return out


def current_accuracy(sport: str = "MLB", min_n: int = 50,
                     model_version: str | None = None,
                     proj_kind: str | None = None) -> dict:
    """Live model's measured accuracy from the ledger — the baseline. Overall + per-stat.
    Scoped to `model_version` (default: current) so a deliberate engine change can't have
    its accuracy quietly averaged with the era before the change — see _ledger_rows.
    `proj_kind` (default None = no filter) additionally scopes to one proj_kind, e.g.
    `current_accuracy("NFL", proj_kind="nfl_preseason")` vs `"nfl_regular"` — see
    _ledger_rows's docstring."""
    mv = model_version if model_version is not None else provenance.model_version(sport)
    rows = _ledger_rows(sport, model_version=mv, proj_kind=proj_kind)
    overall = metrics(rows)
    by_stat: dict = {}
    for r in rows:
        by_stat.setdefault(r["stat_type"], []).append(r)
    per_stat = {s: metrics(rs) for s, rs in by_stat.items() if len(rs) >= min_n}
    # worst-performing stats by MAE (where model work pays off most)
    ranked = sorted(((s, m) for s, m in per_stat.items()),
                    key=lambda kv: kv[1].get("mae", 0), reverse=True)
    return {
        "sport": sport, "model_version": mv, "proj_kind": proj_kind,
        "source": "graded ledger (recorded projections vs actuals)",
        "overall": overall,
        "per_stat": per_stat,
        "worst_by_mae": [{"stat": s, "mae": m["mae"], "hit_rate": m.get("hit_rate"), "n": m["n"]}
                         for s, m in ranked[:8]],
        "target_coverage": _TARGET_COVERAGE,
    }


def reliability_diagram(sport: str = "MLB", stat: Optional[str] = None,
                        n_buckets: int = 10, proj_kind: str | None = None) -> dict:
    """The full reliability-diagram data for one sport (optionally one stat): predicted
    P(over) vs realized frequency per bucket, plus Brier/ECE over the same population.
    This is the direct answer to "are the probabilities actually calibrated" — a
    well-calibrated model's buckets sit on the y=x diagonal; systematic over/under-
    confidence shows up as buckets bowing above or below it. `proj_kind` (default None)
    scopes to one proj_kind — see _ledger_rows's docstring."""
    rows = _ledger_rows(sport, stat=stat, proj_kind=proj_kind)
    probs = _prob_outcome_pairs(rows)
    if len(probs) < 50:
        return {"sport": sport, "stat": stat, "status": "insufficient_data", "n": len(probs)}
    buckets = reliability_buckets(probs, n_buckets)
    ece = sum(abs(b["predicted"] - b["actual"]) * b["n"] for b in buckets if b["n"]) / len(probs)
    brier = sum((pr - o) ** 2 for pr, o in probs) / len(probs)
    return {
        "sport": sport, "stat": stat, "n": len(probs), "buckets": buckets,
        "ece": round(ece, 4), "brier": round(brier, 4),
        "brier_baseline": 0.25,   # a model that always guesses 50% scores exactly this
    }


def by_sample_depth(sport: str = "MLB", min_n: int = 30,
                    proj_kind: str | None = None) -> dict:
    """Accuracy broken out by how much real history backed each projection at grading
    time (model_n — games/PA/BF of evidence, already logged on every row, see db.py's
    schema comment). This is the honest, MEASURED answer to "which player archetypes
    perform worst" — thin-sample rookies/call-ups/injury-returns vs deep-sample
    established regulars — without inventing a role/position taxonomy the ledger
    doesn't actually carry. `proj_kind` (default None) scopes to one proj_kind — see
    _ledger_rows's docstring."""
    rows = [r for r in _ledger_rows(sport, proj_kind=proj_kind) if r.get("N") is not None]
    if not rows:
        return {"sport": sport, "status": "insufficient_data", "n": 0,
                "note": "no rows carry model_n yet -- only graded AFTER the 2026-08 "
                        "model_n logging change will have it"}
    tiers = [("thin (<10 games)", lambda n: n < 10),
             ("medium (10-25 games)", lambda n: 10 <= n < 25),
             ("deep (25+ games)", lambda n: n >= 25)]
    out = []
    for label, pred in tiers:
        subset = [r for r in rows if pred(r["N"])]
        if len(subset) >= min_n:
            out.append({"archetype": label, **metrics(subset)})
    return {"sport": sport, "tiers": out, "n_total": len(rows)}


# ── diagnostics: does the model's OWN signal about itself hold up? ─────────────
# These answer the questions a real calibration system has to answer, using only what's
# already logged on every graded row — no new plumbing, no fabricated attribution.

def _bucket_by_interval_width(rows: list[dict], n_buckets: int = 3) -> list[tuple[str, list[dict]]]:
    """Split graded rows into n_buckets tertiles by RELATIVE p10-p90 band width
    (ceiling-floor)/|projection| — the model's own self-reported uncertainty, already
    logged on every row. Narrower band = the model claims more certainty on that pick."""
    scored = []
    for r in rows:
        fl, ce, p = r.get("FL"), r.get("CE"), r.get("P")
        if fl is None or ce is None or p is None or ce <= fl:
            continue
        scored.append(((ce - fl) / max(1e-9, abs(p)), r))
    if len(scored) < n_buckets * 20:
        return []
    scored.sort(key=lambda x: x[0])
    n = len(scored)
    labels = (["narrow (confident)", "medium", "wide (uncertain)"] if n_buckets == 3
              else [f"bucket {i + 1}" for i in range(n_buckets)])
    return [(labels[i], [r for _, r in scored[i * n // n_buckets:(i + 1) * n // n_buckets]])
            for i in range(n_buckets)]


def calibration_by_confidence(sport: str = "MLB", n_buckets: int = 3,
                              proj_kind: str | None = None) -> dict:
    """Does the model's OWN stated uncertainty (its p10-p90 band width) predict actual
    accuracy? A trustworthy signal means narrow-band props hit MORE than wide-band ones —
    that's the difference between 'confidence' meaning something and being decoration.
    `proj_kind` (default None) scopes to one proj_kind — see _ledger_rows's docstring."""
    buckets = _bucket_by_interval_width(_ledger_rows(sport, proj_kind=proj_kind), n_buckets)
    if not buckets:
        return {"sport": sport, "status": "insufficient_data"}
    out = [{"bucket": label, **metrics(subset)} for label, subset in buckets]
    narrow, wide = out[0], out[-1]
    trustworthy = (narrow.get("hit_rate") is not None and wide.get("hit_rate") is not None
                   and narrow["hit_rate"] >= wide["hit_rate"])
    return {"sport": sport, "buckets": out, "confidence_signal_trustworthy": trustworthy}


def calibration_by_method(sport: str = "MLB", min_n: int = 20) -> dict:
    """Does a full engine run actually beat the cheaper fallback method, measured —
    or is 'proj_kind: engine' just a label that doesn't earn its cost?"""
    groups: dict[str, list[dict]] = {}
    for r in _ledger_rows(sport):
        groups.setdefault(r.get("PK") or "unknown", []).append(r)
    return {"sport": sport,
            "by_method": {k: metrics(rs) for k, rs in groups.items() if len(rs) >= min_n}}


def calibration_by_book(sport: str = "MLB", min_n: int = 20,
                        proj_kind: str | None = None) -> dict:
    """Per-sportsbook accuracy — catches a book whose lines are systematically stale or
    mispriced relative to the others, which a sport-wide average would hide. `proj_kind`
    (default None) scopes to one proj_kind — see _ledger_rows's docstring."""
    groups: dict[str, list[dict]] = {}
    for r in _ledger_rows(sport, proj_kind=proj_kind):
        groups.setdefault(r.get("SRC") or "unknown", []).append(r)
    return {"sport": sport,
            "by_book": {k: metrics(rs) for k, rs in groups.items() if len(rs) >= min_n}}


def version_history(sport: str = "MLB", min_n: int = 30,
                    proj_kind: str | None = None) -> dict:
    """Accuracy broken out PER model_version that has graded history — lets you see, e.g.,
    whether the current version's measured MAE/hit-rate/coverage is actually better than
    the version it replaced, rather than assuming a deliberate math change helped. Pulls
    every version at once (all_versions=True) specifically to compare across them; every
    OTHER function in this module stays scoped to the current version by default.
    `proj_kind` (default None) additionally scopes to one proj_kind — see _ledger_rows's
    docstring; needed for NFL since one model_version covers both season types."""
    rows = _ledger_rows(sport, all_versions=True, proj_kind=proj_kind)
    by_version: dict = {}
    for r in rows:
        by_version.setdefault(r.get("MV") or "unknown", []).append(r)
    out = {v: metrics(rs) for v, rs in by_version.items() if len(rs) >= min_n}
    return {"sport": sport, "current_version": provenance.model_version(sport),
            "by_version": out}


def diagnostics(sport: str = "MLB", proj_kind: str | None = None) -> dict:
    """The full self-diagnostic bundle for one sport — confidence-signal honesty,
    method comparison, per-book accuracy. Everything measured, nothing attributed.
    `proj_kind` (default None) scopes confidence/book comparison to one proj_kind — see
    _ledger_rows's docstring. method_comparison is left unfiltered: for NFL, proj_kind
    itself IS the method split (nfl_regular vs nfl_preseason), so it already reports the
    breakdown a proj_kind filter would otherwise collapse to a single group."""
    return {
        "sport": sport,
        "confidence_calibration": calibration_by_confidence(sport, proj_kind=proj_kind),
        "method_comparison": calibration_by_method(sport),
        "book_comparison": calibration_by_book(sport, proj_kind=proj_kind),
    }


# ── leakage-free replay (MLB) ─────────────────────────────────────────────────

def replay(sport: str = "MLB", limit: int = 300,
           project_fn: Optional[Callable] = None, stat: Optional[str] = None) -> dict:
    """
    Re-run the projection model on `limit` random historical prop-days, using only game logs
    dated strictly BEFORE each prop. Scores the candidate the same way as the baseline so the
    two are directly comparable. MLB for now (its engine is the primary target); other sports'
    replays are a follow-on. `project_fn` lets a caller inject an alternate model; default is
    the live projector_bridge pipeline.
    """
    if sport != "MLB":
        return {"n": 0, "note": f"replay not yet implemented for {sport}"}
    import random
    import analytics
    import projector_bridge as pb

    rows = _ledger_rows(sport, stat)
    random.shuffle(rows)
    pairs: list[dict] = []
    attempted = 0
    log_cache: dict = {}

    for r in rows:
        if len(pairs) >= limit:
            break
        stat = r["stat_type"]
        gd = r["game_date"]
        try:
            is_p = analytics._prop_is_pitcher(stat)
            meta = analytics._resolve_player(r["player"], None, is_p)
            if not meta or not meta.get("id"):
                continue
            pid = meta["id"]
            is_pitcher = (meta.get("pos_type") == "Pitcher") if is_p is None else bool(is_p)
            group = "pitching" if is_pitcher else "hitting"
            attempted += 1
            key = (pid, group)
            logs = log_cache.get(key)
            if logs is None:
                logs = analytics._full_logs(pid, group)
                log_cache[key] = logs
            # LEAKAGE GUARD: only games before the prop's date.
            past = [g for g in logs if (g.get("date") or "9999") < gd]
            if len(past) < 5:
                continue
            pg = analytics._proj_games(past, is_pitcher)
            if not pg:
                continue
            fn = project_fn or pb.project_player
            projs = fn(pg, is_pitcher)
            eng = pb.for_stat(projs, stat, float(r["L"]), is_pitcher)
            if not eng or eng.get("projection") is None:
                continue
            pairs.append({"L": r["L"], "P": eng["projection"], "Y": r["Y"],
                          "RP": eng.get("prob_over"),
                          "FL": eng.get("floor"), "CE": eng.get("ceiling")})
        except Exception:
            continue

    m = metrics(pairs)
    m["attempted"] = attempted
    m["note"] = "leakage-free replay (logs filtered to before each prop's date)"
    return m


def record_run(sport: str = "MLB", kind: str = "baseline",
               metrics_dict: Optional[dict] = None) -> dict:
    """Persist a backtest run to the model registry (spec Ch 178), so a model version's
    measured accuracy is tracked over time. Defaults to recording the current baseline."""
    import uuid
    from datetime import datetime, timezone
    import provenance
    import store
    m = metrics_dict or current_accuracy(sport)["overall"]
    row = {
        "id": uuid.uuid4().hex,
        "model_version": provenance.model_version(sport),
        "sport": sport, "kind": kind,
        "n": m.get("n"), "mae": m.get("mae"), "rmse": m.get("rmse"),
        "bias": m.get("bias"), "hit_rate": m.get("hit_rate"),
        "ece": m.get("ece"), "coverage": m.get("coverage"), "brier": m.get("brier"),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        db_ = store.get_database()
        store.init_schema(db_)
        db_.execute(
            "INSERT INTO model_runs (id, model_version, sport, kind, n, mae, rmse, bias, "
            "hit_rate, ece, coverage, brier, recorded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            tuple(row[k] for k in ("id", "model_version", "sport", "kind", "n", "mae",
                                   "rmse", "bias", "hit_rate", "ece", "coverage", "brier",
                                   "recorded_at")))
    except Exception as exc:
        return {"recorded": False, "reason": str(exc)}
    return {"recorded": True, "run": row}


def registry(sport: Optional[str] = None, limit: int = 50) -> list[dict]:
    """Recorded backtest runs, newest first (the model registry / performance history)."""
    import store
    db_ = store.get_database()
    try:
        store.init_schema(db_)
        if sport:
            return db_.query("SELECT * FROM model_runs WHERE sport=? ORDER BY recorded_at DESC "
                             "LIMIT ?", (sport, limit))
        return db_.query("SELECT * FROM model_runs ORDER BY recorded_at DESC LIMIT ?", (limit,))
    except Exception:
        return []


def drift(sport: str = "MLB", recent_frac: float = 0.2, min_n: int = 60,
         proj_kind: str | None = None) -> dict:
    """
    Model-drift detection (spec Ch 182): split the graded ledger by game date into a RECENT
    window (the newest `recent_frac` of dated props) vs the PRIOR window, and compare accuracy.
    Flags drift when recent hit-rate drops materially or recent MAE inflates vs the prior — i.e.
    the live model is quietly getting worse and should be investigated/retrained.

    Scoped to the CURRENT model_version by default (via _ledger_rows) — a version bump
    always looks like a `recent` MAE spike/hit-rate drop against `prior` graded rows from
    a DIFFERENT model if the two eras were pooled; that would be a false drift alarm on the
    exact events that are supposed to reset the baseline, not a real regression. If there
    isn't yet enough graded history under the current version to split, this correctly
    reports insufficient_data rather than silently falling back to a cross-version compare.

    `proj_kind` (default None) additionally scopes to one proj_kind — see _ledger_rows's
    docstring; without it, an NFL preseason-vs-regular transition (weeks with only
    preseason rows, then only regular-season ones) would look identical to genuine drift.
    """
    mv = provenance.model_version(sport)
    rows = [r for r in _ledger_rows(sport, model_version=mv, proj_kind=proj_kind)
            if r.get("game_date")]
    d = _drift_from_rows(rows, recent_frac, min_n)
    d["sport"] = sport
    d["model_version"] = mv
    return d


def _drift_from_rows(rows: list[dict], recent_frac: float, min_n: int) -> dict:
    """Shared recent-vs-prior split + comparison, factored out of `drift()` so
    `drift_by_stat` can reuse the exact same logic per stat instead of duplicating it."""
    if len(rows) < 2 * min_n:
        return {"status": "insufficient_data", "n": len(rows)}
    dates = sorted({r["game_date"] for r in rows})
    cut = dates[int(len(dates) * (1 - recent_frac))]
    recent = [r for r in rows if r["game_date"] >= cut]
    prior = [r for r in rows if r["game_date"] < cut]
    if len(recent) < min_n or len(prior) < min_n:
        return {"status": "insufficient_data", "recent_n": len(recent), "prior_n": len(prior)}
    mr, mp = metrics(recent), metrics(prior)
    hit_drop = (mp.get("hit_rate") or 0) - (mr.get("hit_rate") or 0)
    mae_infl = (mr["mae"] - mp["mae"]) / mp["mae"] if mp.get("mae") else 0.0
    drifting = hit_drop > 0.05 or mae_infl > 0.15    # >5pp hit-rate drop or >15% MAE inflation
    return {
        "status": "drifting" if drifting else "stable",
        "since": cut,
        "recent": {"n": mr["n"], "hit_rate": mr.get("hit_rate"), "mae": mr["mae"]},
        "prior": {"n": mp["n"], "hit_rate": mp.get("hit_rate"), "mae": mp["mae"]},
        "hit_rate_drop": round(hit_drop, 4),
        "mae_inflation_pct": round(mae_infl * 100, 1),
    }


def drift_by_stat(sport: str = "MLB", recent_frac: float = 0.2, min_n: int = 30,
                  proj_kind: str | None = None) -> dict:
    """Per-stat drift (same recent-vs-prior comparison as `drift`, but broken out per
    stat_type) — answers "WHICH statistics are drifting", not just whether the sport
    overall is. A sport-wide 'stable' can hide one stat quietly degrading while others
    improve and average it out. `proj_kind` (default None) scopes to one proj_kind —
    see _ledger_rows's docstring."""
    mv = provenance.model_version(sport)
    rows = [r for r in _ledger_rows(sport, model_version=mv, proj_kind=proj_kind)
            if r.get("game_date")]
    by_stat: dict = {}
    for r in rows:
        by_stat.setdefault(r["stat_type"], []).append(r)
    out = {}
    for stat, rs in by_stat.items():
        d = _drift_from_rows(rs, recent_frac, min_n)
        if d.get("status") != "insufficient_data":
            out[stat] = d
    drifting_stats = [s for s, d in out.items() if d["status"] == "drifting"]
    return {"sport": sport, "model_version": mv, "by_stat": out,
            "drifting_stats": drifting_stats}


def compare(baseline: dict, candidate: dict) -> dict:
    """Diff two metric dicts (baseline = live, candidate = replayed new model)."""
    keys = ("mae", "rmse", "bias", "hit_rate", "ece", "coverage")
    delta = {}
    for k in keys:
        b, c = baseline.get(k), candidate.get(k)
        if b is None or c is None:
            continue
        # lower is better for mae/rmse/ece and |bias|; higher for hit_rate; toward 0.80 coverage
        if k == "hit_rate":
            better = c > b
        elif k == "coverage":
            better = abs(c - _TARGET_COVERAGE) < abs(b - _TARGET_COVERAGE)
        elif k == "bias":
            better = abs(c) < abs(b)
        else:
            better = c < b
        delta[k] = {"baseline": b, "candidate": c, "delta": round(c - b, 4), "better": better}
    return {"n_baseline": baseline.get("n"), "n_candidate": candidate.get("n"), "by_metric": delta}
