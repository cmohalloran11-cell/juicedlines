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
        for col in ("close_proj_b", "close_prob_b"):
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
        #    WNBA/SL/tennis/soccer blend toward the line, so their close_proj is NOT the model.
        #    The edge regression (y−L)=a+γ(m−L) needs the PRE-anchor m. Recovering it later via
        #    m=L+(final−L)/t is impossible at t=0 (snap-to-line) and numerically unstable at
        #    small t (tennis ~0.05 → 20× any rounding), so log it directly.
        for col, typ in (("odds_type", "TEXT"),
                         ("close_over_price", "TEXT"), ("close_under_price", "TEXT"),
                         ("close_over_implied", "REAL"), ("close_under_implied", "REAL"),
                         ("model_raw", "REAL"),        # pre-anchor projection m
                         ("model_raw_prob", "REAL"),   # pre-anchor P(over) at close_line
                         ("trust_weight", "REAL"),     # anchor weight actually applied
                         ("game_id", "TEXT")):
            if col not in have:
                c.execute(f"ALTER TABLE prop_clv ADD COLUMN {col} {typ}")
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
            l.get("model_proj_b"), l.get("model_prob_b"),   # variant B (enhanced)
            # audit fields — see the schema comment. Price makes profitability answerable;
            # odds_type lets demon/goblin be filtered out of the edge regression;
            # model_raw/trust_weight expose the PRE-anchor model on the blended sports.
            l.get("odds_type"),
            l.get("over_price"), l.get("under_price"),
            l.get("over_implied"), l.get("under_implied"),
            l.get("model_raw"), l.get("model_raw_prob"), l.get("trust_weight"),
            l.get("game_id"),
        ))
    if not rows:
        return 0
    with _lock, _conn() as c:
        c.executemany("""
            INSERT INTO prop_clv (line_id, game_date, sport, source, player, stat_type,
                open_ts, open_line, open_prob, open_proj,
                close_ts, close_line, close_prob, close_proj, proj_kind,
                close_proj_b, close_prob_b,
                odds_type, close_over_price, close_under_price,
                close_over_implied, close_under_implied,
                model_raw, model_raw_prob, trust_weight, game_id)
            VALUES (?,?,?,?,?,?, ?,?,?,?, ?,?,?,?, ?, ?,?, ?,?,?,?,?, ?,?,?,?)
            ON CONFLICT(line_id, game_date) DO UPDATE SET
                close_ts=excluded.close_ts, close_line=excluded.close_line,
                close_prob=excluded.close_prob, close_proj=excluded.close_proj,
                stat_type=excluded.stat_type, proj_kind=excluded.proj_kind,
                close_proj_b=excluded.close_proj_b, close_prob_b=excluded.close_prob_b,
                odds_type=excluded.odds_type,
                close_over_price=excluded.close_over_price,
                close_under_price=excluded.close_under_price,
                close_over_implied=excluded.close_over_implied,
                close_under_implied=excluded.close_under_implied,
                model_raw=excluded.model_raw, model_raw_prob=excluded.model_raw_prob,
                trust_weight=excluded.trust_weight, game_id=excluded.game_id
        """, rows)
        c.commit()
    return len(rows)


def pending_grades(today: str, sport: str = "MLB", limit: int = 400) -> list[dict]:
    """Un-attempted props from past game-days (gradeable via a game-log lookup)."""
    with _lock, _conn() as c:
        rows = c.execute("""
            SELECT line_id, game_date, player, stat_type FROM prop_clv
            WHERE graded_at IS NULL AND game_date < ? AND sport = ? AND player IS NOT NULL
            ORDER BY game_date DESC LIMIT ?
        """, (today, sport, limit)).fetchall()
    return [dict(r) for r in rows]


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
            pick_over = cp > cl
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


def stat_biases(sport: str = "MLB", min_n: int = 60) -> dict:
    """
    Per-stat systematic bias = mean(projection − actual) from graded props,
    deduped to one point per (player, stat, game). A stable, sufficiently-sampled
    bias is a calibration error we can correct on future projections. Returns
    {stat_type_lower: bias} only for stats with ≥ min_n data points.
    """
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT player, stat_type, game_date, close_proj, actual FROM prop_clv "
            "WHERE actual IS NOT NULL AND close_proj IS NOT NULL AND sport = ?",
            (sport,)).fetchall()
    seen: dict = {}                      # (player,stat,date) → (proj, actual), constant per group
    for r in rows:
        k = (r["player"], r["stat_type"], r["game_date"])
        seen.setdefault(k, (r["close_proj"], r["actual"]))
    agg: dict = {}
    for (pl, st, gd), (proj, act) in seen.items():
        agg.setdefault((st or "").lower(), []).append(proj - act)
    return {k: round(sum(d) / len(d), 3) for k, d in agg.items() if len(d) >= min_n}
