# JUICED

A live sports-projections platform for MLB, WNBA, and Tennis player props. Real Monte
Carlo simulation engines (not a market copy) produce a projection, a probability
distribution, and a Juice Score for every prop pulled from PrizePicks, Underdog, and
Sleeper — plus a full self-monitoring Model Health system that tracks its own accuracy,
calibration, and drift against graded outcomes.

## What's here

- **Three real projection engines**, one per sport, each a genuine per-player Monte Carlo
  simulation with empirical-Bayes shrinkage and two-stage (parameter + outcome)
  uncertainty propagation — not a market-derived number:
  - `projector/` — MLB (batters/pitchers), vendored from the sibling `stat-projector`
    project so a deploy that only checks out this repo still runs the real engine.
  - `basketball/` — WNBA, per-possession rate model with measured combo correlation.
  - `tennis/` — ATP/WTA, serve/return point model with an exact game/set/match DP.
- **`projector_bridge.py`** — the MLB glue layer (feature building, correlation-aware
  combo simulation, ensemble blending) between the raw engine and the board.
- **`valuation.py`** — EV, Kelly fraction, Confidence, and the 7-component Juice Score,
  computed as pure functions of already-simulated data (no I/O, fully unit-tested).
- **`optimizer.py`** — the Entry Optimizer: builds every legal same-book combination (2-6
  Pick, Power/Flex) from the priced projection pool, excludes duplicate players and
  correlated legs, and Monte Carlo simulates the survivors (>=100,000 sims on every entry
  actually surfaced) for EV/Kelly/risk-of-ruin and a 0-100 Quality Score. Only PrizePicks has
  a verified multi-pick payout table (`BOOK_PAYOUTS`) — Underdog/Sleeper are selectable but
  return `unavailableReason` until a real published table for them is added. Served at
  `GET /api/optimizer/today?book=` via `routes_optimizer.py`, mounted in `main.py`.
- **`db.py` / `backtest.py` / `model_health.py`** — the graded CLV ledger and the
  self-monitoring stack: MAE/RMSE/bias/Brier/ECE/coverage, reliability diagrams, drift
  detection (sport-wide and per-stat), accuracy by player sample-depth, and accuracy by
  model version, all scoped so a deliberate engine change can never be silently blended
  with the era before it.
- **`main.py`** — the FastAPI server: live board refresh loop, the CLV/grading pipeline,
  and every `/api/*` route.
- **`build_static.py`** — an alternate path that prebuilds the board to static JSON for a
  serverless/CDN deploy (no long-lived server needed) — see `DEPLOY.md`.
- **`static/dashboard.html`** — the entire frontend: one vanilla-JS, hash-routed SPA, no
  build step, no framework.
- **`store/`** — the Postgres-or-SQLite relational layer for users, watchlists, portfolio,
  alerts, and the model-run registry.
- **`fantasy/`** — the fantasy football draft assistant (`/api/fantasy/*`, `routes_fantasy.py`),
  all four planned phases: a canonical player table with source-id mapping + fuzzy-match
  review log, Sleeper league import, a pure league-scoring-rules engine, VOR/tiering and
  live-draft state (draft board + live draft mode), a pure optimal-lineup assignment
  (`lineup.py`), waiver-wire recommendations (reuses `draft_state`/`vor`, no separate module),
  and trade evaluation on VOR (`trade.py`). Projections come from a swappable
  `ProjectionsProvider` (`fantasy/projections/`) -- the only adapter implemented so far
  (`nflverse_adapter.py`) is a disclosed historical-performance baseline, not a licensed
  predictive model (a commercial provider is still TBD); `fantasy/projections_sync.py`
  populates `fantasy_projections` from it lazily, gated by a once-a-day staleness check.
  Sleeper's `/v1/players/nfl` dump is synced the same way by a background job, never on a
  request path. UI lives at `#/fantasy` in `static/dashboard.html` (live-server only — same
  `/api/ai/*`-style exception as AI Juice, not served from the static/prebuilt-JSON deploy).

## Quickstart (local)

```bash
pip install -r requirements.txt
python main.py            # or: uvicorn main:app --reload
```

Open `http://localhost:8001` (override with `PORT`). No API keys or credentials are required to run it — PrizePicks
reads a cookie-free partner API, MLB/ESPN need no keys, and auth/AI features simply stay
disabled until their env vars are set (see `DEPLOY.md`'s environment variable table).

Prefer a zero-server preview? `static/dashboard.html` also works opened directly as a file,
or served via `python -m http.server` from `static/` — it falls back to a prebuilt
`board.json` snapshot (see `build_static.py` / `DEPLOY.md` Option A).

## Running the tests

```bash
pip install -r requirements-dev.txt   # adds pytest + httpx on top of requirements.txt
python -m pytest -q
```

`requirements.txt` alone is NOT enough to run the suite — `pytest` and `httpx` (required
by FastAPI's `TestClient`) are test-only and deliberately kept out of the production
dependency list. Use `requirements-dev.txt` for local test runs; CI already does.

150+ tests across `tests/`, `basketball/tests/`, and `tennis/tests/` — pure-function unit
tests for every scoring/simulation component plus integration tests against a real
(temp-file) database and the live FastAPI app. CI (`.github/workflows/test.yml`) runs this
on every push and PR; `deploy-prod.yml` won't deploy to production unless it passes first.

## Deploying

See **[DEPLOY.md](DEPLOY.md)** — two paths: a free static build (GitHub Actions + Vercel,
~5 min data freshness, no server) or a full always-on server (Render/Railway/Fly/Docker,
continuous live updates). Includes the complete environment variable reference.

## Project conventions worth knowing before touching the code

- **No fabricated numbers.** Every metric shown to a user traces back to a real
  computation from real data — if the ledger is too thin to measure something, the UI
  says so explicitly rather than showing a plausible-looking placeholder.
- **Model versions are deliberate.** `provenance.py`'s `MODEL_VERSIONS` are hand-bumped
  only on a genuine math change, specifically so historical graded rows stay attributable
  to the exact logic that produced them and calibration queries never blend eras. See
  `provenance.MODEL_CHANGELOG` for what each bump actually changed.
- **Statistical changes need before/after evidence.** A model or scoring change should
  ship with a quantitative comparison (backtest, live measurement, or both), not just a
  plausible-sounding rationale.
