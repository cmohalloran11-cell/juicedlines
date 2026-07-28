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

_TARGET_COVERAGE = 0.80


# ── ledger access ─────────────────────────────────────────────────────────────

def _ledger_rows(sport: str, stat: Optional[str] = None) -> list[dict]:
    """Graded, standard-odds prop rows, deduped to one per (player, stat, game_date)."""
    q = ("SELECT player, stat_type, game_date, close_line AS L, close_proj AS P, "
         "actual AS Y, COALESCE(model_raw_prob, close_prob) AS RP, "
         "model_floor AS FL, model_ceiling AS CE "
         "FROM prop_clv WHERE sport=? AND actual IS NOT NULL AND close_line IS NOT NULL "
         "AND close_proj IS NOT NULL "
         "AND (odds_type IS NULL OR LOWER(odds_type) IN ('standard','boosted'))")
    args: list[Any] = [sport]
    if stat:
        q += " AND LOWER(stat_type)=?"
        args.append(stat.lower())
    with db._lock, db._conn() as c:
        rows = [dict(r) for r in c.execute(q, args).fetchall()]
    seen: dict = {}
    for r in rows:
        seen.setdefault((r["player"], r["stat_type"], r["game_date"]), r)
    return list(seen.values())


# ── metrics ───────────────────────────────────────────────────────────────────

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
    # probability calibration (ECE) — reliability of P(over)
    probs = [(p["RP"], 1.0 if p["Y"] > p["L"] else 0.0) for p in P
             if p.get("RP") is not None and p.get("L") is not None and abs(p["Y"] - p["L"]) > 1e-9]
    if len(probs) >= 50:
        bins = [[0.0, 0.0, 0] for _ in range(10)]
        for pr, o in probs:
            b = min(9, int(pr * 10))
            bins[b][0] += pr
            bins[b][1] += o
            bins[b][2] += 1
        ece = sum(abs(s / cnt - o / cnt) * cnt for s, o, cnt in bins if cnt) / len(probs)
        out["ece"] = round(ece, 4)
    # interval coverage (p10–p90 band should hold ~80%)
    band = [p for p in P if p.get("FL") is not None and p.get("CE") is not None and p["CE"] > p["FL"]]
    if len(band) >= 50:
        out["coverage"] = round(sum(1 for p in band if p["FL"] <= p["Y"] <= p["CE"]) / len(band), 4)
    return out


def current_accuracy(sport: str = "MLB", min_n: int = 50) -> dict:
    """Live model's measured accuracy from the ledger — the baseline. Overall + per-stat."""
    rows = _ledger_rows(sport)
    overall = metrics(rows)
    by_stat: dict = {}
    for r in rows:
        by_stat.setdefault(r["stat_type"], []).append(r)
    per_stat = {s: metrics(rs) for s, rs in by_stat.items() if len(rs) >= min_n}
    # worst-performing stats by MAE (where model work pays off most)
    ranked = sorted(((s, m) for s, m in per_stat.items()),
                    key=lambda kv: kv[1].get("mae", 0), reverse=True)
    return {
        "sport": sport, "source": "graded ledger (recorded projections vs actuals)",
        "overall": overall,
        "per_stat": per_stat,
        "worst_by_mae": [{"stat": s, "mae": m["mae"], "hit_rate": m.get("hit_rate"), "n": m["n"]}
                         for s, m in ranked[:8]],
        "target_coverage": _TARGET_COVERAGE,
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
        "ece": m.get("ece"), "coverage": m.get("coverage"),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        db_ = store.get_database()
        store.init_schema(db_)
        db_.execute(
            "INSERT INTO model_runs (id, model_version, sport, kind, n, mae, rmse, bias, "
            "hit_rate, ece, coverage, recorded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            tuple(row[k] for k in ("id", "model_version", "sport", "kind", "n", "mae",
                                   "rmse", "bias", "hit_rate", "ece", "coverage", "recorded_at")))
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


def drift(sport: str = "MLB", recent_frac: float = 0.2, min_n: int = 60) -> dict:
    """
    Model-drift detection (spec Ch 182): split the graded ledger by game date into a RECENT
    window (the newest `recent_frac` of dated props) vs the PRIOR window, and compare accuracy.
    Flags drift when recent hit-rate drops materially or recent MAE inflates vs the prior — i.e.
    the live model is quietly getting worse and should be investigated/retrained.
    """
    rows = [r for r in _ledger_rows(sport) if r.get("game_date")]
    if len(rows) < 2 * min_n:
        return {"sport": sport, "status": "insufficient_data", "n": len(rows)}
    dates = sorted({r["game_date"] for r in rows})
    cut = dates[int(len(dates) * (1 - recent_frac))]
    recent = [r for r in rows if r["game_date"] >= cut]
    prior = [r for r in rows if r["game_date"] < cut]
    if len(recent) < min_n or len(prior) < min_n:
        return {"sport": sport, "status": "insufficient_data",
                "recent_n": len(recent), "prior_n": len(prior)}
    mr, mp = metrics(recent), metrics(prior)
    hit_drop = (mp.get("hit_rate") or 0) - (mr.get("hit_rate") or 0)
    mae_infl = (mr["mae"] - mp["mae"]) / mp["mae"] if mp.get("mae") else 0.0
    drifting = hit_drop > 0.05 or mae_infl > 0.15    # >5pp hit-rate drop or >15% MAE inflation
    return {
        "sport": sport,
        "status": "drifting" if drifting else "stable",
        "since": cut,
        "recent": {"n": mr["n"], "hit_rate": mr.get("hit_rate"), "mae": mr["mae"]},
        "prior": {"n": mp["n"], "hit_rate": mp.get("hit_rate"), "mae": mp["mae"]},
        "hit_rate_drop": round(hit_drop, 4),
        "mae_inflation_pct": round(mae_infl * 100, 1),
    }


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
