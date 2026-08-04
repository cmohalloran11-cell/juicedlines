"""
optimizer.py — Entry Optimizer ("the user should never have to manually decide what to bet").

Turns the already-scored projection pool (valuation.py's EV/Kelly/Confidence/Juice Score,
db.py's graded-ledger calibration) into ranked, legal PrizePicks entries across every
supported format — instead of a person scanning the board and building parlays by hand.

Pipeline, mirroring the spec's 5 steps:
  1. build_pool()        — every priced, projected prop + its real scoring fields (reuses
                            dashboard.py's enrichment; adds calibration/historical-accuracy/
                            model-agreement/volatility, which dashboard.py doesn't expose).
  2. candidate_legsets()  — legal leg combinations per format, filtered (no duplicate player,
                            no correlated pair — see correlation_tag) and quick-ranked.
  3. simulate_entry()     — Monte Carlo (Gaussian-copula correlated Bernoulli draws + the real
                            PrizePicks payout table) → EV, win probability, Kelly, risk of ruin.
  4. best_entries()       — two-stage: a fast scan pass ranks the shortlist, then only the
                            surfaced finalists are re-simulated at n_sims>=100,000.
  5. today_report()       — "Today's Best" per format + a cross-format Top Entries list, each
                            with a 0-100 Quality Score.

Design boundary (matches valuation.py — see CLAUDE.md "Architecture philosophy"): every
number here traces to a real, already-computed field on a line dict or a DB-backed query
passed in by the caller. Two exceptions are explicitly heuristic, not measured, and are
labeled as such below: the correlation TAGGING (categorical, from real team/player/time
fields) and its associated assumed rho constants (this codebase has never measured
cross-player prop correlation — projector_bridge's 0.49-0.59 numbers are intra-player
H/R/RBI only). The default generator (see PREFER_INDEPENDENT) avoids correlated pairs
entirely rather than lean on an assumed number, so a surfaced entry's simulation runs at
rho=0 (independent) by construction; the copula machinery still exists so a future,
measured correlation table can be dropped in without changing the simulation code.
"""
from __future__ import annotations

import itertools
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
from scipy.stats import norm

import dashboard as dashboard_mod
import valuation
import db
import backtest

# ── payout tables, per book ──────────────────────────────────────────────────────
# An entry's legs must all come from ONE book — a "combination" spanning PrizePicks and
# Underdog isn't a bet anyone can actually place. Multiplier paid on a $1 stake for N correct
# legs out of the number picked (0 = loses the full stake).
#
# PrizePicks: PUBLIC values from PrizePicks' own "How to Play" payout chart — a snapshot of a
# published reference table, not fetched/measured data, and PrizePicks can change it. If the
# live table drifts, update this constant (same maintenance model as an American-odds table).
# 2-Pick Flex IS a real PrizePicks product (2026-08-04, confirmed) — 2/2 correct pays 2.0x,
# 1/2 pays 0.5x (a partial-insurance payout even at the minimum pick count, unlike the 3+
# pattern the other Flex tiers follow).
#
# Underdog / Sleeper: this codebase only ever pulls a PER-LEG boost multiplier for these two
# (see underdog.py's payout_multiplier, books.py's Sleeper connector) — never their full
# multi-pick Pick'em payout chart. Rather than guess at numbers this repo has never verified,
# no table is populated for them; the book selector still lets a user scope the LEG POOL to
# Underdog/Sleeper (e.g. for browsing), but no entry can be built until a real table is added.
SUPPORTED_BOOKS: tuple[str, ...] = ("prizepicks", "underdog", "sleeper")
DEFAULT_BOOK = "prizepicks"

BOOK_PAYOUTS: dict[str, dict[tuple[int, str], dict[int, float]]] = {
    "prizepicks": {
        (2, "power"): {2: 3.0},
        (3, "power"): {3: 5.0},
        (4, "power"): {4: 10.0},
        (5, "power"): {5: 20.0},
        (2, "flex"): {2: 2.0, 1: 0.5},
        (3, "flex"): {3: 2.25, 2: 1.25},
        (4, "flex"): {4: 5.0, 3: 1.5},
        (5, "flex"): {5: 10.0, 4: 2.0, 3: 0.4},
        (6, "flex"): {6: 25.0, 5: 2.0, 4: 0.4},
    },
    "underdog": {},
    "sleeper": {},
}
# Backward-compatible aliases — PrizePicks is the default/only fully-supported book today.
PAYOUTS: dict[tuple[int, str], dict[int, float]] = BOOK_PAYOUTS[DEFAULT_BOOK]
FORMATS: list[tuple[int, str]] = list(PAYOUTS.keys())

# Legs considered per format before combining (top-N by juiceScore) — bounds the combination
# count to something a web request can score (C(35,2)=595 .. C(16,6)=8008), independent of
# how many props are actually on the board.
_POOL_CAP = {2: 35, 3: 30, 4: 24, 5: 20, 6: 16}

# Hard-avoid correlated pairs by default (see module docstring) rather than price them with
# an assumed, unmeasured rho.
PREFER_INDEPENDENT = True

# Assumed (NOT measured) pairwise correlation used only if a caller explicitly opts out of
# PREFER_INDEPENDENT via candidate_legsets(..., allow_correlated=True). Conservative and
# clearly smaller than the codebase's one real measured cross-stat number (intra-player
# H/R/RBI 0.49-0.59) precisely because this is a guess, not a fit.
_ASSUMED_RHO = {"teammate_stack": 0.20, "same_game_opponents": 0.08, "independent": 0.0}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Step 1: pool building ───────────────────────────────────────────────────────

def _sports_in(rows: list[dict]) -> list[str]:
    return sorted({(r.get("sport") or "").upper() for r in rows if r.get("sport")})


def _historical_accuracy_maps(sports: list[str]) -> dict[str, dict[str, Optional[float]]]:
    """{sport: {lowercased stat: hit_rate}} from the graded ledger (backtest.current_accuracy).
    Missing/thin stats simply aren't in the map — no invented accuracy number."""
    out: dict[str, dict[str, Optional[float]]] = {}
    for sp in sports:
        try:
            per_stat = backtest.current_accuracy(sp).get("per_stat", {})
            out[sp] = {stat.lower(): m.get("hit_rate") for stat, m in per_stat.items()}
        except Exception:
            out[sp] = {}
    return out


def _calibration_maps(sports: list[str]) -> dict[str, dict[str, float]]:
    """{sport: {lowercased stat: gamma trust}} — same source valuation.py's Juice Score
    already reads off the line (stat_trust_gamma), exposed per-stat for the optimizer."""
    out: dict[str, dict[str, float]] = {}
    for sp in sports:
        try:
            out[sp] = db.stat_gammas(sp)
        except Exception:
            out[sp] = {}
    return out


def build_pool(lines: list[dict]) -> list[dict]:
    """Every priced, projected prop as an optimizer "leg": dashboard.py's enriched row
    (juiceScore/confidence/edgePct/probability/side/volatility/...) plus calibration, real
    historical accuracy, and model agreement — the extra scoring fields the spec asks for
    that dashboard.py's own board rows don't need. Reuses dashboard._drop() rather than
    re-deriving its fields (volatility/kellyPct in particular already live there)."""
    priced = [l for l in dashboard_mod._projected(lines) if not valuation.is_unpriced(l)]
    dropped = [dashboard_mod._drop(l) for l in priced]
    sports = _sports_in(dropped)
    hist = _historical_accuracy_maps(sports)
    calib = _calibration_maps(sports)
    pool = []
    for l, row in zip(priced, dropped):
        sport = (row.get("sport") or "").upper()
        stat = (row.get("stat") or "").lower()
        row["calibration"] = calib.get(sport, {}).get(stat)
        if row.get("historicalAccuracy") is None:
            row["historicalAccuracy"] = hist.get(sport, {}).get(stat)
        row["modelAgreement"] = valuation.model_agreement_score(l)
        pool.append(row)
    return pool


# ── Step 2: correlation heuristic + candidate generation ───────────────────────

def correlation_tag(a: dict, b: dict) -> str:
    """Categorical risk tag from REAL fields already on the row (player/team/opponent) — not
    a fabricated correlation coefficient (see module docstring for why this codebase has no
    measured cross-player number to draw on)."""
    if a.get("player") and a.get("player") == b.get("player"):
        return "duplicate_player"
    at, ao = a.get("team"), a.get("opponent")
    bt, bo = b.get("team"), b.get("opponent")
    if at and bt and at == bt:
        return "teammate_stack"
    if (at and bo and at == bo) or (bt and ao and bt == ao):
        return "same_game_opponents"
    return "independent"


def correlation_summary(legs: list[dict]) -> dict:
    flags = []
    for a, b in itertools.combinations(legs, 2):
        tag = correlation_tag(a, b)
        if tag != "independent":
            flags.append({"a": a.get("player"), "b": b.get("player"), "type": tag})
    level = "high" if len(flags) >= 3 else "medium" if flags else "low"
    return {"level": level, "linkedPairs": len(flags), "flags": flags}


def candidate_legsets(pool: list[dict], picks: int, sport: Optional[str] = None,
                      shortlist: int = 8, allow_correlated: bool = False,
                      book: str = DEFAULT_BOOK) -> list[list[dict]]:
    """Legal, quick-ranked leg combinations for one format. Hard rules: unique players, no
    linked pair (unless allow_correlated), and every leg from the SAME book (`source`) — a
    combination spanning two books isn't a bet anyone can place — this IS "avoid duplicate
    players / highly correlated props / opposite sides of the same game" from the spec,
    enforced structurally rather than scored after the fact. Quick-ranks by mean juiceScore
    (already the product's real composite quality signal — see valuation.juice_score) to
    shortlist before the expensive Monte Carlo pass."""
    eligible = [l for l in pool if l.get("side") and l.get("probability") is not None
               and (l.get("source") or "").lower() == book.lower()]
    if sport and sport.lower() != "all":
        eligible = [l for l in eligible if (l.get("sport") or "").lower() == sport.lower()]
    if len(eligible) < picks:
        return []
    cap = _POOL_CAP.get(picks, 16)
    eligible.sort(key=lambda l: l.get("juiceScore") or 0, reverse=True)
    eligible = eligible[:cap]

    scored: list[tuple[float, list[dict]]] = []
    for combo in itertools.combinations(eligible, picks):
        if len({l.get("player") for l in combo}) != picks:
            continue
        if not allow_correlated and any(
                correlation_tag(a, b) != "independent" for a, b in itertools.combinations(combo, 2)):
            continue
        score = sum(l.get("juiceScore") or 0 for l in combo) / picks
        scored.append((score, list(combo)))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [combo for _, combo in scored[:shortlist]]


# ── Step 3: Monte Carlo simulation ──────────────────────────────────────────────

def _leg_prob(l: dict) -> Optional[float]:
    """Probability of the RECOMMENDED side hitting (dashboard rows always carry model_prob
    as P(over); flip it when the recommended side is under)."""
    p = l.get("probability")
    if p is None:
        return None
    return float(p) if l.get("side") == "over" else 1.0 - float(p)


def _corr_matrix(legs: list[dict], allow_correlated: bool) -> np.ndarray:
    n = len(legs)
    m = np.eye(n)
    if not allow_correlated:
        return m
    for i, j in itertools.combinations(range(n), 2):
        rho = _ASSUMED_RHO.get(correlation_tag(legs[i], legs[j]), 0.0)
        m[i, j] = m[j, i] = rho
    return m


def _nearest_corr_pd(m: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    vals, vecs = np.linalg.eigh(m)
    vals = np.clip(vals, eps, None)
    fixed = (vecs * vals) @ vecs.T
    d = np.sqrt(np.diag(fixed))
    fixed = fixed / np.outer(d, d)
    np.fill_diagonal(fixed, 1.0)
    return fixed


def _kelly_search(returns: np.ndarray, cap: float = 0.25, iters: int = 60) -> float:
    """Fraction of bankroll maximizing E[log(1+f*R)] over the SIMULATED return distribution —
    the Kelly criterion generalized to an arbitrary (non-binary) payout distribution, found by
    ternary search (the objective is concave in f). Quarter-Kelly cap, same rationale as
    valuation.kelly_fraction: full Kelly is too aggressive for a noisy model."""
    lo, hi = 0.0, cap
    for _ in range(iters):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        f1 = float(np.mean(np.log1p(m1 * returns)))
        f2 = float(np.mean(np.log1p(m2 * returns)))
        if f1 < f2:
            lo = m1
        else:
            hi = m2
    return round(max(0.0, (lo + hi) / 2.0), 4)


def _risk_of_ruin(returns: np.ndarray, kelly_f: float, paths: int = 1500, steps: int = 40,
                  ruin_frac: float = 0.2, seed: int = 99) -> float:
    """Bankroll simulation: bootstrap-resample this entry's own simulated returns to model
    staking `kelly_f` of a $1 bankroll on repeated plays of the same entry, and measure how
    often the bankroll ever drops below `ruin_frac` of the start within `steps` plays."""
    if kelly_f <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(returns), size=(paths, steps))
    step_returns = returns[idx]
    bankroll = np.ones(paths)
    ruined = np.zeros(paths, dtype=bool)
    for t in range(steps):
        bankroll = bankroll * (1.0 + kelly_f * step_returns[:, t])
        ruined |= bankroll < ruin_frac
        bankroll = np.maximum(bankroll, 0.0)
    return round(float(ruined.mean()), 4)


def simulate_entry(legs: list[dict], picks: int, kind: str, n_sims: int = 100_000,
                   seed: int = 42, allow_correlated: bool = not PREFER_INDEPENDENT,
                   book: str = DEFAULT_BOOK) -> Optional[dict]:
    """Monte Carlo the entry: correlated (Gaussian copula) or independent Bernoulli draws per
    leg at its own model probability, scored through the real payout table for (picks, kind)
    on `book`. Returns None if any leg is missing a probability, the format is unsupported, or
    `book` has no verified payout table (see BOOK_PAYOUTS)."""
    table = BOOK_PAYOUTS.get(book.lower(), {}).get((picks, kind))
    if table is None or len(legs) != picks:
        return None
    probs = [_leg_prob(l) for l in legs]
    if any(p is None for p in probs):
        return None
    probs_arr = np.array(probs, dtype=float)

    corr = _corr_matrix(legs, allow_correlated)
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(_nearest_corr_pd(corr))

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_sims, len(legs))) @ L.T
    u = norm.cdf(z)
    hits = u < probs_arr
    hit_counts = hits.sum(axis=1)

    mult_lookup = np.zeros(picks + 1)
    for h, mult in table.items():
        mult_lookup[h] = mult
    multipliers = mult_lookup[hit_counts]
    returns = multipliers - 1.0  # profit per $1 staked

    kelly_f = _kelly_search(returns)
    return {
        "picks": picks, "kind": kind, "book": book.lower(), "simCount": n_sims,
        "expectedValue": round(float(returns.mean()), 4),
        "expectedROI": round(float(returns.mean()), 4),   # EV per $1 stake IS the ROI here
        "probabilityAllHit": round(float((hit_counts == picks).mean()), 4),
        "chanceOfProfit": round(float((returns > 0).mean()), 4),
        "averageReturn": round(float(returns.mean()), 4),
        "medianReturn": round(float(np.median(returns)), 4),
        "worst10": round(float(np.percentile(returns, 10)), 4),
        "best10": round(float(np.percentile(returns, 90)), 4),
        "variance": round(float(returns.var()), 5),
        "kellyFraction": kelly_f,
        "riskOfRuin": _risk_of_ruin(returns, kelly_f),
    }


# ── Step 4/5: ranking + orchestration ───────────────────────────────────────────

_QUALITY_WEIGHTS = {"ev": 0.30, "profitChance": 0.20, "confidence": 0.20,
                    "calibration": 0.15, "diversification": 0.15}
_CORR_MULT = {"low": 1.0, "medium": 0.88, "high": 0.7}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _scale01(x: Optional[float], lo: float, hi: float) -> float:
    if x is None or hi == lo:
        return 0.5
    return _clamp01((x - lo) / (hi - lo))


def quality_score(sim: dict, legs: list[dict], corr: dict) -> int:
    """0-100 composite ranking one entry overall — EV, chance of profit, average leg
    confidence, average leg calibration (measured stat-trust gamma, 0.5 neutral default when
    unmeasured), and game/team diversification, discounted by the correlation heuristic. Every
    input is a real field already computed upstream; see module docstring for what's a
    heuristic (the correlation discount) vs. measured (everything else)."""
    ev_score = _scale01(sim.get("expectedROI"), -0.20, 0.40)
    profit_score = sim.get("chanceOfProfit") or 0.0
    conf_score = sum((l.get("confidence") or 0) for l in legs) / (100.0 * len(legs))
    calib_vals = [l.get("calibration") if l.get("calibration") is not None else 0.5 for l in legs]
    calib_score = sum(calib_vals) / len(calib_vals)
    distinct_games = len({(l.get("sport"), l.get("startTime"), l.get("team")) for l in legs})
    div_score = _clamp01(distinct_games / len(legs))
    composite = (_QUALITY_WEIGHTS["ev"] * ev_score
                + _QUALITY_WEIGHTS["profitChance"] * profit_score
                + _QUALITY_WEIGHTS["confidence"] * conf_score
                + _QUALITY_WEIGHTS["calibration"] * calib_score
                + _QUALITY_WEIGHTS["diversification"] * div_score)
    composite *= _CORR_MULT[corr["level"]]
    return int(round(_clamp01(composite) * 100))


def _leg_summary(l: dict) -> dict:
    keep = ("id", "player", "team", "opponent", "sport", "stat", "line", "projection",
           "side", "probability", "edgePct", "confidence", "juiceScore", "calibration",
           "historicalAccuracy", "modelAgreement", "volatility", "source", "headshot",
           "startTime", "oddsType")
    return {k: l.get(k) for k in keep}


def _reason(sim: dict, corr: dict) -> str:
    bits = [f"{sim['expectedROI']*100:+.1f}% expected ROI",
           f"{sim['chanceOfProfit']*100:.0f}% chance of profit"]
    bits.append("fully diversified legs" if corr["level"] == "low"
               else f"{corr['linkedPairs']} correlated pair(s) flagged")
    return "; ".join(bits)


# Quality safeguard, same pattern as valuation.EV_REVIEW_THRESHOLD/audit_ev — flag, don't
# silently rewrite. A combo's EV compounds every leg's probability multiplicatively, so a
# leg with genuine (if modest) tail overconfidence — the residual gap this codebase's own
# calibration diagnostics call out as unresolved even after Platt calibration ("the
# high-confidence TAIL stays overconfident", db.prob_calibration's docstring) — gets
# amplified into a combo EV that reads far more certain than it should be. Rather than
# invent an unvalidated compression constant for combos (the way _dampen_ev was tuned for
# single props off measured live examples), an extreme combo is flagged for human review;
# the underlying number is never altered.
ENTRY_EV_REVIEW_THRESHOLD = float(os.getenv("ENTRY_EV_REVIEW_THRESHOLD", "1.00"))


def audit_entry(sim: dict, threshold: Optional[float] = None) -> Optional[dict]:
    """Flag an entry whose simulated expectedROI exceeds `threshold` (+100% by default).
    Returns None when not flagged, else a dict a reviewer/UI can use to explain why."""
    th = ENTRY_EV_REVIEW_THRESHOLD if threshold is None else threshold
    roi = sim.get("expectedROI")
    if roi is None or roi <= th:
        return None
    return {
        "flagged": True, "threshold": th, "expectedROI": roi,
        "reason": "This entry's combined EV compounds each leg's individual probability — an "
                 "unusually high number here typically means one or more legs carry more "
                 "model confidence than history has validated, not a guaranteed edge.",
    }


def build_entry(legs: list[dict], picks: int, kind: str, n_sims: int = 100_000,
                book: str = DEFAULT_BOOK) -> Optional[dict]:
    sim = simulate_entry(legs, picks, kind, n_sims=n_sims, book=book)
    if sim is None:
        return None
    corr = correlation_summary(legs)
    return {
        "book": book.lower(), "picks": picks, "type": kind.capitalize(),
        "format": f"{picks} Pick {kind.capitalize()}",
        "qualityScore": quality_score(sim, legs, corr),
        "simulation": sim,
        "correlation": corr,
        "legs": [_leg_summary(l) for l in legs],
        "reason": _reason(sim, corr),
        "reviewFlag": audit_entry(sim),
    }


def best_entries(pool: list[dict], picks: int, kind: str, top_n: int = 1, shortlist: int = 8,
                 scan_sims: int = 20_000, final_sims: int = 100_000,
                 sport: Optional[str] = None, book: str = DEFAULT_BOOK) -> list[dict]:
    """Two-stage: shortlist candidates by quick juiceScore ranking, scan them at `scan_sims`
    to pick the real winner(s) by simulated EV, then re-simulate ONLY those finalists at
    `final_sims` (>=100,000 by default — the spec's simulation-count floor, applied to what
    actually gets surfaced rather than to every candidate, which would be computationally
    wasteful for no accuracy gain on the ones that get discarded). Empty when `book` has no
    payout table for (picks, kind) — see BOOK_PAYOUTS."""
    if (picks, kind) not in BOOK_PAYOUTS.get(book.lower(), {}):
        return []
    candidates = candidate_legsets(pool, picks, sport=sport, shortlist=shortlist, book=book)
    scanned = []
    for legs in candidates:
        seed = abs(hash(tuple(sorted(l.get("id") or "" for l in legs)))) % (2**31)
        sim = simulate_entry(legs, picks, kind, n_sims=scan_sims, seed=seed, book=book)
        if sim is not None:
            scanned.append((sim["expectedROI"], legs))
    scanned.sort(key=lambda t: t[0], reverse=True)
    entries = [e for legs in (l for _, l in scanned[:top_n])
              if (e := build_entry(legs, picks, kind, n_sims=final_sims, book=book)) is not None]
    entries.sort(key=lambda e: e["qualityScore"], reverse=True)
    return entries


def todays_best(pool: list[dict], sport: Optional[str] = None,
                book: str = DEFAULT_BOOK) -> dict[str, Optional[dict]]:
    """One top entry per format supported on `book` — the "Today's Best" grid. Empty dict
    when the book has no verified payout table yet."""
    out: dict[str, Optional[dict]] = {}
    for picks, kind in BOOK_PAYOUTS.get(book.lower(), {}):
        entries = best_entries(pool, picks, kind, top_n=1, sport=sport, book=book)
        out[f"{picks}_{kind}"] = entries[0] if entries else None
    return out


def top_overall(pool: list[dict], limit: int = 6, sport: Optional[str] = None,
                book: str = DEFAULT_BOOK) -> list[dict]:
    """Cross-format ranked leaderboard — the "#1 Best Entry" style list."""
    all_entries: list[dict] = []
    for picks, kind in BOOK_PAYOUTS.get(book.lower(), {}):
        all_entries.extend(best_entries(pool, picks, kind, top_n=2, sport=sport, book=book))
    all_entries.sort(key=lambda e: e["qualityScore"], reverse=True)
    return all_entries[:limit]


_CACHE: dict = {"key": None, "t": 0.0, "v": None}
_CACHE_TTL = 180.0  # seconds — matches the board's own ~5min freshness floor with headroom


def today_report(lines: list[dict], sport: Optional[str] = None,
                 book: str = DEFAULT_BOOK) -> dict:
    """Full payload for the Entry Optimizer page: today's-best grid + a cross-format top-entry
    leaderboard, scoped to one book (every leg in an entry must come from the same book — see
    candidate_legsets). Cached briefly (like dashboard.py's _OPP_CACHE) since a full pass
    re-runs the combination search + Monte Carlo across every format that book supports."""
    book = (book or DEFAULT_BOOK).lower()
    key = (sport or "all", book, len(lines))
    now = time.time()
    if _CACHE["key"] == key and now - _CACHE["t"] < _CACHE_TTL:
        return _CACHE["v"]
    has_payouts = bool(BOOK_PAYOUTS.get(book))
    pool = build_pool(lines)
    book_pool_size = sum(1 for l in pool if (l.get("source") or "").lower() == book)
    result = {
        "generatedAt": _now_iso(),
        "book": book,
        "poolSize": book_pool_size,
        "unavailableReason": (None if has_payouts else
                              f"No verified multi-pick payout table for {book} yet — "
                              f"switch to PrizePicks, the only book with a real published "
                              "payout chart in this codebase (see optimizer.BOOK_PAYOUTS)."),
        "todaysBest": todays_best(pool, sport=sport, book=book) if has_payouts else {},
        "topEntries": top_overall(pool, sport=sport, book=book) if has_payouts else [],
    }
    _CACHE.update(key=key, t=now, v=result)
    return result
