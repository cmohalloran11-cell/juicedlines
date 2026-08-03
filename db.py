"""
db.py — SQLite history store for line snapshots.

Schema:
  line_history(ts TEXT, line_id TEXT, source TEXT, sport TEXT,
               player TEXT, team TEXT, stat_type TEXT,
               line_value REAL, over_implied REAL, under_implied REAL)
"""

import sqlite3
import threading
from pathlib import Path
from typing import Any

import provenance

DB_PATH = Path(__file__).parent / "history.db"
_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _lock, _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS line_history (
                ts           TEXT NOT NULL,
                line_id      TEXT NOT NULL,
                source       TEXT,
                sport        TEXT,
                player       TEXT,
                team         TEXT,
                stat_type    TEXT,
                line_value   REAL,
                over_implied REAL,
                under_implied REAL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_line_id ON line_history(line_id, ts)")
        # CLV / hit-rate ledger: one row per prop per game-day, tracking the line +
        # model projection from first sight (open) to last sight (close), plus the
        # graded actual once the game finishes. This is what lets us measure, over
        # time, whether the projections actually beat the market.
        c.execute("""
            CREATE TABLE IF NOT EXISTS prop_clv (
                line_id    TEXT NOT NULL,
                game_date  TEXT NOT NULL,
                sport      TEXT, source TEXT, player TEXT, stat_type TEXT,
                open_ts    TEXT, open_line REAL, open_prob REAL, open_proj REAL,
                close_ts   TEXT, close_line REAL, close_prob REAL, close_proj REAL,
                proj_kind  TEXT,
                actual     REAL,
                graded_at  TEXT,
                PRIMARY KEY (line_id, game_date)
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_clv_grade ON prop_clv(actual, game_date)")
        # A/B test columns: variant B = engine + matchup context + xBA prior (vs the
        # plain variant A in close_proj/close_prob). Added via ALTER so existing rows
        # keep their data (they'll have NULL B — excluded from the head-to-head).
        have = {r[1] for r in c.execute("PRAGMA table_info(prop_clv)")}
        # variant B = engine + matchup context + xBA prior; variant C = engine + confirmed
        # batting-order PA (2026-07-17). Separate columns so each enhancement is measured on
        # its own — reusing B's column for C would blend two feature definitions across time
        # and corrupt the head-to-head. NULL where the variant didn't differ (excluded from
        # the paired test), which for C is every prop without a posted lineup at build time.
        for col in ("close_proj_b", "close_prob_b", "close_proj_c", "close_prob_c"):
            if col not in have:
                c.execute(f"ALTER TABLE prop_clv ADD COLUMN {col} REAL")
        # ── audit fields (2026-07-16) ───────────────────────────────────────────
        # Without these the ledger cannot answer the only two questions that matter:
        #  * PRICE: profitability is unmeasurable without it. Scoring a hit-rate against a
        #    flat −110 is meaningless when we bet the same juiced side every time — e.g.
        #    HR 0.5 unders "won" 86% purely as a base rate (mean(m−L)=−0.37, sd 0.09), which
        #    reads as a huge edge and is actually just the price.
        #  * odds_type: demon/goblin lines are deliberately warped, so (m−L) vs (y−L) on them
        #    manufactures correlation and inflates the edge regression. Must be filterable.
        #  * model_raw / trust_weight: MLB doesn't anchor (close_proj IS the raw model), but
        #    WNBA/tennis blend toward the line, so their close_proj is NOT the model.
        #    The edge regression (y−L)=a+γ(m−L) needs the PRE-anchor m. Recovering it later via
        #    m=L+(final−L)/t is impossible at t=0 (snap-to-line) and numerically unstable at
        #    small t (tennis ~0.05 → 20× any rounding), so log it directly.
        for col, typ in (("odds_type", "TEXT"),
                         ("close_over_price", "TEXT"), ("close_under_price", "TEXT"),
                         ("close_over_implied", "REAL"), ("close_under_implied", "REAL"),
                         ("model_raw", "REAL"),        # pre-anchor projection m
                         ("model_raw_prob", "REAL"),   # pre-anchor P(over) at close_line
                         ("trust_weight", "REAL"),     # anchor weight actually applied
                         # the SHIPPED p10/p90 band. Without these, interval coverage can only be
                         # inferred from P(over) via a normal approx, which blows up on low-count
                         # discrete stats (runs vs a 0.5 line). Logged → coverage is a direct count.
                         ("model_floor", "REAL"), ("model_ceiling", "REAL"),
                         ("game_id", "TEXT"),
                         # ── provenance (2026-07-26) — spec Principle 4 (reproducibility).
                         # Records WHICH model/feature logic and WHICH data date produced each
                         # projection, so a historical row stays attributable to its exact logic.
                         ("model_version", "TEXT"),
                         ("feature_version", "TEXT"),
                         ("data_snapshot", "TEXT"),
                         # ── model-health archetype signal (2026-08) ─────────────────────
                         # model_n = games/PA/BF of history behind the projection at grading
                         # time -- already computed on every line for confidence_score, just
                         # never persisted. A REAL, measured "how much do we know about this
                         # player" signal, not a fabricated archetype label — lets
                         # backtest.by_sample_depth answer "which player archetypes (thin-
                         # sample rookies/call-ups vs deep-sample regulars) perform worst?"
                         ("model_n", "REAL")):
            if col not in have:
                c.execute(f"ALTER TABLE prop_clv ADD COLUMN {col} {typ}")
        # 2026-08 (production-readiness audit): every calibration/accuracy query in this
        # file and backtest.py filters on sport (always) and model_version (almost always,
        # after this session's model-version scoping work), then checks actual IS NOT NULL.
        # idx_clv_grade above is (actual, game_date) -- useless as a leading index for a
        # sport= filter. Without this, every stat_gammas/prob_calibration/interval_width/
        # stat_biases/_ledger_rows call does a full table scan, and prop_clv only grows
        # (graded rows are kept forever, db.py's own comment on line_history's retention
        # policy doesn't apply here). sport+model_version covers the common case; actual
        # as the third column lets SQLite use the index for the IS NOT NULL check too.
        c.execute("CREATE INDEX IF NOT EXISTS idx_clv_sport_version "
                 "ON prop_clv(sport, model_version, actual)")
        c.commit()


def snapshot_lines(lines: list[dict[str, Any]], ts: str) -> None:
    """Insert one snapshot row per line into history."""
    rows = [
        (
            ts,
            l["id"],
            l.get("source"),
            l.get("sport"),
            l.get("player"),
            l.get("team"),
            l.get("stat_type"),
            l.get("line"),
            l.get("over_implied"),
            l.get("under_implied"),
        )
        for l in lines
    ]
    with _lock, _conn() as c:
        c.executemany(
            "INSERT INTO line_history VALUES (?,?,?,?,?,?,?,?,?,?)", rows
        )
        c.commit()


def prune_clv(keep_ungraded_days: int = 3) -> int:
    """
    Drop stale UNGRADED prop_clv rows; keep every GRADED one forever.

    Graded rows are the research asset (they're what the edge regression runs on) and are
    only ~3% of the table — 3.4k rows / 0.9 MB vs 106k / 22.5 MB. The other 97% are props
    that never resolved (sports we don't grade yet, cancellations, or rows still waiting).
    A row only needs to survive open → close → grading (~1-2 days), so anything ungraded
    past a few days never will be. This is what keeps the published ledger small enough to
    live on the `data` branch.
    """
    with _lock, _conn() as c:
        cur = c.execute(
            "DELETE FROM prop_clv WHERE actual IS NULL AND graded_at IS NULL "
            "AND game_date < date('now', ?)", (f"-{int(keep_ungraded_days)} day",))
        n = cur.rowcount or 0
        c.commit()
    # VACUUM can't run inside a transaction — reclaim space on its own connection.
    try:
        c2 = sqlite3.connect(DB_PATH)
        c2.execute("VACUUM")
        c2.close()
    except Exception:
        pass
    return n


def prune_history(keep_days: int = 14) -> int:
    """
    Drop line_history snapshot rows older than `keep_days`.

    line_history grows by ~one row per live line per snapshot (~14k MLB rows every few
    minutes). It is ONLY read to draw a prop's recent line-movement chart (get_history),
    and a prop is settled within a day of its slate — nobody charts movement from weeks
    ago. Left unpruned it reached 9.6M rows / 1.77 GB in a month. A rolling window keeps
    the file bounded; freed pages are reused by later inserts, so the file stabilizes
    without a VACUUM every pass (call compact() for a one-time reclaim after shrinking
    the window). Returns the number of rows deleted.
    """
    with _lock, _conn() as c:
        cur = c.execute(
            "DELETE FROM line_history WHERE ts < strftime('%Y-%m-%dT%H:%M:%S+00:00', "
            "'now', ?)", (f"-{int(keep_days)} day",))
        n = cur.rowcount or 0
        c.commit()
    return n


def compact() -> None:
    """VACUUM the database to reclaim freelist pages to the OS. Slow and needs free disk
    (~the DB's size) — run sparingly (one-time after a big prune / retention change), not
    every loop. Runs on its own connection: VACUUM can't execute inside a transaction."""
    try:
        c2 = sqlite3.connect(DB_PATH)
        c2.execute("VACUUM")
        c2.close()
    except Exception:
        pass


def get_history(line_id: str, limit: int = 200) -> list[dict]:
    with _lock, _conn() as c:
        rows = c.execute(
            """SELECT ts, line_value, over_implied, under_implied
               FROM line_history WHERE line_id = ?
               ORDER BY ts DESC LIMIT ?""",
            (line_id, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def get_recent_snapshots(sport: str | None = None, limit: int = 50) -> list[dict]:
    with _lock, _conn() as c:
        if sport:
            rows = c.execute(
                """SELECT * FROM line_history WHERE sport = ?
                   ORDER BY ts DESC LIMIT ?""",
                (sport, limit),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM line_history ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────── CLV / hit-rate ledger ───────────────────────────

def log_clv(lines: list[dict[str, Any]], ts: str) -> int:
    """
    Upsert one row per (line_id, game_date) with the current line + model
    projection. First sight sets open_* and close_*; later snapshots only move
    close_*, so each row ends up holding the open→close move for that prop-day.
    game_date = the UTC date of the snapshot (props are for that day's slate).
    Only logs lines that carry a numeric line + a model projection + a player.
    """
    gd = ts[:10]
    rows = []
    for l in lines:
        if l.get("model_proj") is None or l.get("line") is None or not l.get("player"):
            continue
        if l.get("lineup_status") == "out":     # confirmed scratch → a DNP, don't grade it
            continue
        try:
            ln = float(l["line"])
        except (TypeError, ValueError):
            continue
        rows.append((
            l["id"], gd, l.get("sport"), l.get("source"), l.get("player"),
            l.get("stat_type"),
            ts, ln, l.get("model_prob"), l.get("model_proj"),
            ts, ln, l.get("model_prob"), l.get("model_proj"),
            l.get("proj_kind"),
            l.get("model_proj_b"), l.get("model_prob_b"),   # variant B (matchup ctx + xBA)
            l.get("model_proj_c"), l.get("model_prob_c"),   # variant C (batting-order PA)
            # audit fields — see the schema comment. Price makes profitability answerable;
            # odds_type lets demon/goblin be filtered out of the edge regression;
            # model_raw/trust_weight expose the PRE-anchor model on the blended sports.
            l.get("odds_type"),
            l.get("over_price"), l.get("under_price"),
            l.get("over_implied"), l.get("under_implied"),
            l.get("model_raw"), l.get("model_raw_prob"), l.get("trust_weight"),
            l.get("model_floor"), l.get("model_ceiling"),
            l.get("game_id"),
            l.get("model_version"), l.get("feature_version"), l.get("data_snapshot"),
            l.get("model_n"),
        ))
    if not rows:
        return 0
    with _lock, _conn() as c:
        c.executemany("""
            INSERT INTO prop_clv (line_id, game_date, sport, source, player, stat_type,
                open_ts, open_line, open_prob, open_proj,
                close_ts, close_line, close_prob, close_proj, proj_kind,
                close_proj_b, close_prob_b, close_proj_c, close_prob_c,
                odds_type, close_over_price, close_under_price,
                close_over_implied, close_under_implied,
                model_raw, model_raw_prob, trust_weight,
                model_floor, model_ceiling, game_id,
                model_version, feature_version, data_snapshot, model_n)
            VALUES (?,?,?,?,?,?, ?,?,?,?, ?,?,?,?, ?, ?,?, ?,?, ?,?,?,?,?, ?,?,?, ?,?,?, ?,?,?, ?)
            ON CONFLICT(line_id, game_date) DO UPDATE SET
                close_ts=excluded.close_ts, close_line=excluded.close_line,
                close_prob=excluded.close_prob, close_proj=excluded.close_proj,
                stat_type=excluded.stat_type, proj_kind=excluded.proj_kind,
                close_proj_b=excluded.close_proj_b, close_prob_b=excluded.close_prob_b,
                close_proj_c=excluded.close_proj_c, close_prob_c=excluded.close_prob_c,
                odds_type=excluded.odds_type,
                close_over_price=excluded.close_over_price,
                close_under_price=excluded.close_under_price,
                close_over_implied=excluded.close_over_implied,
                close_under_implied=excluded.close_under_implied,
                model_raw=excluded.model_raw, model_raw_prob=excluded.model_raw_prob,
                trust_weight=excluded.trust_weight,
                model_floor=excluded.model_floor, model_ceiling=excluded.model_ceiling,
                game_id=excluded.game_id,
                model_version=excluded.model_version,
                feature_version=excluded.feature_version,
                data_snapshot=excluded.data_snapshot,
                model_n=excluded.model_n
        """, rows)
        c.commit()
    return len(rows)


def pending_grades(today: str, sport: str = "MLB", limit: int = 400) -> list[dict]:
    """Un-attempted props from past game-days (gradeable via a game-log lookup).

    Ordered by RESEARCH VALUE, not just recency. The board now logs ~14k MLB rows/day but
    grading is rate-limited, and `prune_clv` deletes ungraded rows after a few days — so a
    plain `ORDER BY game_date` spent the whole budget on demon/goblin props that EVERY
    analysis filters out, while the rows that answer questions were pruned unqraded. Measured
    2026-07-19: of 24,586 pending MLB rows only 2,825 were standard odds and just 703 were
    standard AND carried variant C — i.e. the entire remaining batting-order sample was about
    to be deleted. Priority is on pre-treatment fields (odds type, which variant was logged),
    never on the outcome, so this concentrates power without biasing any estimate.
    """
    with _lock, _conn() as c:
        rows = c.execute("""
            SELECT line_id, game_date, player, stat_type FROM prop_clv
            WHERE graded_at IS NULL AND game_date < ? AND sport = ? AND player IS NOT NULL
            ORDER BY
                (odds_type IS NULL OR LOWER(odds_type) IN ('standard','boosted')) DESC,
                (close_proj_b IS NOT NULL OR close_proj_c IS NOT NULL) DESC,
                (close_over_price IS NOT NULL) DESC,
                game_date DESC
            LIMIT ?
        """, (today, sport, limit)).fetchall()
    return [dict(r) for r in rows]


def stat_gammas(sport: str = "MLB", min_n: int = 120, prior: float = 0.20,
                k: float = 300.0, model_version: str | None = None) -> dict:
    """Per-stat TRUST weight learned from the graded ledger — how much to trust the model over
    the market line, PER STAT. trust = the edge-regression slope γ of (actual−line) on
    (model−line): the optimal shrinkage of the model toward the line (proj = line + γ·(model−
    line)). Where the model beats the line (Hits/HR) γ is high → keep the model; where it
    doesn't (Runs/Total Bases) γ≈0 → defer to the line, so we stop surfacing noise edges.

    γ is shrunk toward `prior` by sample size (pseudo-count k) so a thin/noisy stat can't swing
    it, and read off `model_raw` (COALESCE with close_proj for pre-anchor rows) — the RAW model,
    never the anchored projection, so trust can't feed back on itself. Validated out-of-sample
    2026-07-21 (train 7d → test 4d): overall MAE 3.157→3.076, 18/19 stats better, ROI −11.2→−9.0%.

    Scoped to `model_version` (default: the CURRENT version for this sport, per
    provenance.model_version) so a deliberate model change — e.g. the 2026-08 MLB engine-crash
    fix, which silently swapped every live projection from a bare empirical fallback to the
    real Monte-Carlo engine — can't have its trust weights computed from a blend of the old
    model's graded history and the new one's. Until enough NEW-version rows accrue this
    honestly returns fewer/no stats rather than reusing a stale-model fit — the same "empty
    until we actually have enough to trust" behavior this function already had for thin data."""
    mv = model_version if model_version is not None else provenance.model_version(sport)
    with _lock, _conn() as c:
        rows = c.execute(
            """SELECT LOWER(stat_type) st, close_line L, COALESCE(model_raw, close_proj) m, actual y
               FROM prop_clv WHERE sport=? AND actual IS NOT NULL AND close_line IS NOT NULL
               AND stat_type IS NOT NULL AND model_version=?
               AND (odds_type IS NULL OR LOWER(odds_type) IN ('standard','boosted'))""",
            (sport, mv)).fetchall()
    import statistics
    by: dict = {}
    for r in rows:
        if r["m"] is None or r["L"] is None:
            continue
        by.setdefault(r["st"], []).append((r["m"] - r["L"], r["y"] - r["L"]))
    out: dict = {}
    for st, pts in by.items():
        n = len(pts)
        if n < min_n:
            continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        vx = statistics.pvariance(xs)
        if vx < 1e-9:                       # model never disagrees with the line here
            continue
        mx, my = statistics.mean(xs), statistics.mean(ys)
        g = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n) / vx
        gsh = (n * g + k * prior) / (n + k)         # shrink toward prior by sample size
        out[st] = round(max(0.0, min(1.0, gsh)), 3)
    return out


def prob_calibration(sport: str = "MLB", min_n: int = 400,
                     model_version: str | None = None) -> dict:
    """Platt calibration of P(over) — {"a":…, "b":…} for p_cal = sigmoid(a·logit(p)+b). The model's
    Monte-Carlo distributions are OVERCONFIDENT (measured 2026-07-21: a "62% over" hits ~51%), so a<1
    shrinks confidence toward 0.5. Fit by IRLS on the graded ledger's `model_raw_prob` (the PRE-
    calibration prob; COALESCE with close_prob for old rows) so it never feeds back on itself.
    Validated out-of-sample (train 7d→test 4d): ECE 0.070→0.045, log-loss + Brier both down.
    Empty {} until ≥min_n rows or a bad fit → probabilities ship uncalibrated, as before.

    Scoped to `model_version` (default: current, per provenance.model_version) — see
    stat_gammas's docstring for why pooling graded rows across a deliberate model change would
    apply one model's measured overconfidence correction to a different model's output."""
    import numpy as np
    mv = model_version if model_version is not None else provenance.model_version(sport)
    with _lock, _conn() as c:
        rows = c.execute(
            """SELECT close_line L, actual y, COALESCE(model_raw_prob, close_prob) p FROM prop_clv
               WHERE sport=? AND actual IS NOT NULL AND close_line IS NOT NULL
               AND COALESCE(model_raw_prob, close_prob) IS NOT NULL AND model_version=?
               AND (odds_type IS NULL OR LOWER(odds_type) IN ('standard','boosted'))""",
            (sport, mv)).fetchall()
    pts = [(r["p"], 1.0 if r["y"] > r["L"] else 0.0) for r in rows
           if r["p"] is not None and abs(r["y"] - r["L"]) > 1e-9]
    if len(pts) < min_n:
        return {}
    p = np.clip(np.array([q for q, _ in pts]), 1e-3, 1 - 1e-3)
    x = np.log(p / (1 - p))                       # logit(raw prob)
    y = np.array([o for _, o in pts])
    a, b = 1.0, 0.0
    for _ in range(50):                            # Newton / IRLS for 2-param logistic
        q = 1.0 / (1.0 + np.exp(-np.clip(a * x + b, -30, 30)))
        g = np.array([np.sum((q - y) * x), np.sum(q - y)])
        w = q * (1 - q)
        H = np.array([[np.sum(w * x * x), np.sum(w * x)], [np.sum(w * x), np.sum(w)]])
        try:
            step = np.linalg.solve(H + 1e-6 * np.eye(2), g)
        except Exception:
            return {}
        a -= float(step[0]); b -= float(step[1])
        if np.abs(step).max() < 1e-8:
            break
    if not (0.2 < a < 3.0):                         # implausible fit → don't calibrate
        return {}
    return {"a": round(a, 4), "b": round(b, 4)}


def interval_width(sport: str = "MLB", min_n: int = 300,
                   model_version: str | None = None) -> float:
    """How much to STRETCH the reported floor–ceiling band so it means what it says.

    Our p10–p90 band should contain 80% of outcomes. Measured 2026-07-21 on MLB it contained
    ~56% — the Monte-Carlo distributions are too narrow, so every floor/ceiling we showed was
    tighter than reality.

    Two ways to size the fix, in preference order:
      1. DIRECT — once `model_floor`/`model_ceiling` have accrued, just count how often the
         outcome landed inside the shipped band and scale the half-width by the ratio of the
         normal z-scores for target vs actual coverage. No distributional assumption at all.
      2. IMPLIED — before that data exists, borrow the Platt slope: shrinking the logit by `a`
         is algebraically the same correction as widening the SD by 1/a, so w = 1/a. (The
         standalone-SD estimate is NOT used: backing an SD out of a probability explodes on
         low-count discrete stats — runs against a 0.5 line — and read 4.3x against this 1.85x.)

    Returns 1.0 (no stretch) when there's nothing trustworthy to go on.

    Scoped to `model_version` (default: current) — see stat_gammas's docstring. A stretch
    factor measured against the OLD model's coverage gap is not the right correction for a
    newly-fixed/changed model's distributions, which can be narrow (or wide) by a different
    amount.
    """
    from statistics import NormalDist
    mv = model_version if model_version is not None else provenance.model_version(sport)
    with _lock, _conn() as c:
        rows = c.execute(
            """SELECT actual y, model_floor lo, model_ceiling hi FROM prop_clv
               WHERE sport=? AND actual IS NOT NULL AND model_floor IS NOT NULL
               AND model_ceiling IS NOT NULL AND model_ceiling > model_floor AND model_version=?
               AND (odds_type IS NULL OR LOWER(odds_type) IN ('standard','boosted'))""",
            (sport, mv)).fetchall()
    if len(rows) >= min_n:
        hit = sum(1 for r in rows if r["lo"] <= r["y"] <= r["hi"]) / len(rows)
        hit = min(0.98, max(0.30, hit))
        # band is ±z_actual sigma but should be ±z_target sigma → stretch by the ratio
        z_act = NormalDist().inv_cdf(0.5 + hit / 2.0)
        z_tgt = NormalDist().inv_cdf(0.90)            # p10–p90
        if z_act > 1e-6:
            return round(min(3.0, max(1.0, z_tgt / z_act)), 2)
    cal = prob_calibration(sport, model_version=mv)
    a = float(cal.get("a") or 0)
    if 0.2 < a < 1.0:
        return round(min(3.0, 1.0 / a), 2)
    return 1.0


def set_actual(line_id: str, game_date: str, actual: float | None, graded_at: str) -> None:
    """Record the graded outcome. actual=None marks the prop attempted-but-void
    (player didn't play that day), so it stops showing up as pending."""
    with _lock, _conn() as c:
        c.execute("UPDATE prop_clv SET actual=?, graded_at=? WHERE line_id=? AND game_date=?",
                  (actual, graded_at, line_id, game_date))
        c.commit()


def scorecard(sport: str | None = None, edge: float = 0.5) -> dict:
    """
    Aggregate the graded ledger into the numbers that matter:
      • hit-rate of the model's side vs the CLOSING line (pushes excluded),
      • the subset where the model had a real lean (|proj−line| ≥ edge) — the "plays",
      • CLV: how often / how much the line moved toward the model after we first saw it.
    """
    q = "SELECT * FROM prop_clv WHERE actual IS NOT NULL"
    args: list[Any] = []
    if sport:
        q += " AND sport = ?"; args.append(sport)
    with _lock, _conn() as c:
        rows = [dict(r) for r in c.execute(q, args).fetchall()]

    graded = len(rows)
    dec = play_n = play_hit = hit = 0
    clv_pos = clv_n = 0
    clv_sum = 0.0
    ab_n = ab_a = ab_b = 0        # A/B head-to-head on props that have both variants
    for r in rows:
        cl, cp = r.get("close_line"), r.get("close_proj")
        ol, op = r.get("open_line"), r.get("open_proj")
        act = r.get("actual")
        if cl is None or cp is None or act is None:
            continue
        if act == cl:            # push — no decision
            pass
        else:
            dec += 1
            # the model's ACTUAL pick is its P(over) vs 0.5, not mean-vs-line — they diverge on
            # right-skewed count stats (mean above the line while P(over) < 0.5 → pick under).
            _rp = r.get("model_raw_prob")
            if _rp is None:
                _rp = r.get("close_prob")
            pick_over = (_rp > 0.5) if _rp is not None else (cp > cl)
            if (act > cl) == pick_over:
                hit += 1
            if abs(cp - cl) >= edge:
                play_n += 1
                if (act > cl) == pick_over:
                    play_hit += 1
            # A (plain) vs B (enhanced) on the SAME prop, where B exists
            cpb = r.get("close_proj_b")
            if cpb is not None:
                ab_n += 1
                if (act > cl) == pick_over:
                    ab_a += 1
                if (act > cl) == (cpb > cl):
                    ab_b += 1
        # CLV: did the line move toward our open lean by close?
        if ol is not None and op is not None and cl is not None:
            lean_over = op > ol
            move = (cl - ol) if lean_over else (ol - cl)
            clv_n += 1
            clv_sum += move
            if move > 0:
                clv_pos += 1
    return {
        "graded": graded, "decided": dec,
        "hit_rate": round(hit / dec, 4) if dec else None,
        "plays": play_n, "play_hit_rate": round(play_hit / play_n, 4) if play_n else None,
        "clv_n": clv_n,
        "clv_positive_pct": round(clv_pos / clv_n, 4) if clv_n else None,
        "clv_avg_move": round(clv_sum / clv_n, 4) if clv_n else None,
        "breakeven": 0.5238,  # -110 vig breakeven
        # head-to-head: plain board engine (A) vs engine+matchup+xBA (B)
        "ab_n": ab_n,
        "ab_plain": round(ab_a / ab_n, 4) if ab_n else None,
        "ab_enhanced": round(ab_b / ab_n, 4) if ab_n else None,
    }


def recent_prop_moves(limit: int = 400) -> list[dict]:
    """Today's props with their open→close line and projection movement (for the dashboard's
    Top Movers / Biggest Line Move widgets). Reads the CLV ledger, which already tracks the
    open and close of each prop-day. Returns only rows where something actually moved."""
    with _lock, _conn() as c:
        rows = c.execute("""
            SELECT player, stat_type, sport, source,
                   open_line, close_line, open_proj, close_proj
            FROM prop_clv
            WHERE game_date = date('now') AND player IS NOT NULL
              AND open_line IS NOT NULL AND close_line IS NOT NULL
            ORDER BY ABS(COALESCE(close_line,0)-COALESCE(open_line,0)) DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def stat_biases(sport: str = "MLB", min_n: int = 60,
                model_version: str | None = None) -> dict:
    """
    Per-stat systematic bias = mean(projection − actual) from graded props,
    deduped to one point per (player, stat, game). A stable, sufficiently-sampled
    bias is a calibration error we can correct on future projections. Returns
    {stat_type_lower: bias} only for stats with ≥ min_n data points.

    Scoped to `model_version` (default: current) — see stat_gammas's docstring.
    """
    mv = model_version if model_version is not None else provenance.model_version(sport)
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT player, stat_type, game_date, close_proj, actual FROM prop_clv "
            "WHERE actual IS NOT NULL AND close_proj IS NOT NULL AND sport = ? "
            "AND model_version = ?",
            (sport, mv)).fetchall()
    seen: dict = {}                      # (player,stat,date) → (proj, actual), constant per group
    for r in rows:
        k = (r["player"], r["stat_type"], r["game_date"])
        seen.setdefault(k, (r["close_proj"], r["actual"]))
    agg: dict = {}
    for (pl, st, gd), (proj, act) in seen.items():
        agg.setdefault((st or "").lower(), []).append(proj - act)
    return {k: round(sum(d) / len(d), 3) for k, d in agg.items() if len(d) >= min_n}
