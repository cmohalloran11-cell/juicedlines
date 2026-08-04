# JUICED — Project Status

> **Living document.** Refresh with `/status`. Every number here should trace back to
> something actually checked in the session that last updated this file — never carried
> forward from memory. If you're reading this and something looks stale, run `/status`
> before trusting it.

**Last updated:** 2026-08-03
**Updated by:** manager (via `/status`, following the launch-readiness + agent-framework
build session)

---

## Launch status

**Free public beta — engineering-complete, launched.** Payment/subscription processing
does not exist and is explicitly deferred (product decision: free beta first).

## Test suite

- **153 tests passing** across `tests/`, `basketball/tests/`, `tennis/tests/`.
- Verified in a genuinely clean environment (fresh `venv` + fresh checkout +
  `pip install -r requirements-dev.txt`), not just the developer's own machine — this
  distinction matters: see [Known limitations / recent incidents](#recent-incidents-worth-remembering).
- Requires `requirements-dev.txt` (adds `pytest`, `httpx` on top of `requirements.txt`) —
  `requirements.txt` alone is intentionally insufficient to run tests.

## CI / CD

- `.github/workflows/test.yml` — runs the full suite on every push (any branch) and PR
  into `main`.
- `.github/workflows/deploy-prod.yml` — same suite gates the Vercel production deploy
  hook (`needs: test`). Verified: both the `test` and `deploy` jobs succeed on the latest
  commit to `main` (checked directly against the GitHub Actions API, not inferred from
  local results).
- `.github/dependabot.yml` — weekly automated PRs for `pip` and `github-actions`
  ecosystems.
- `pip-audit` runs in CI (non-blocking) — clean as of last check.

## Model versions (`provenance.MODEL_VERSIONS`)

| Sport | Version | Notes |
|---|---|---|
| MLB | `mlb-1.2.0` | Two-stage uncertainty propagation |
| WNBA | `wnba-1.1.0` | Two-stage uncertainty + measured combo correlation |
| Tennis | `tennis-1.2.0` | Two-stage uncertainty propagation |

All three engines fully implement two-stage (parameter + outcome) uncertainty and
empirical-Bayes shrinkage. Grading pipelines for all three sports (`grade_pending`,
`grade_basketball`, `grade_tennis`) are wired into the live background loop.

## Model Health

Self-monitoring system live at `/dashboard#/model-health` and `/api/model/dashboard`:
MAE/RMSE/bias/Brier/ECE/coverage, reliability diagrams, sport-wide and per-stat drift
detection, accuracy by player sample-depth, accuracy by model version, daily snapshots,
plain-language changelog (`provenance.MODEL_CHANGELOG`).

**Known transitional state:** interval-coverage self-correction resets on every
model-version bump and needs fresh graded history to re-measure — expected to stabilize
as games are graded under the current versions. Not a defect; nothing to act on beyond
watching it.

## Security / reliability posture

- Auth (Supabase JWKS) verified configured and working.
- Rate limiting on by default (300 req/min/IP over `/api/*`).
- Expensive Model Health/backtest endpoints cached (120s TTL); `/api/backtest`'s
  `replay` action requires login.
- XSS-escaping (`esc()`) applied on the highest-traffic frontend surfaces.
- Sentry error monitoring integrated, optional via `SENTRY_DSN`, verified working
  end-to-end with a real DSN (test event captured and flushed successfully).

## Compliance / legal

Terms of Service (`/terms.html`) and Privacy Policy (`/privacy.html`) live, linked from
the landing page footer and the app's Settings page. In-app Responsible Gaming
disclaimer present (Settings page + persistent "18+" sidebar mark).

## Recent incidents worth remembering

- **CI was silently broken for an extended period**: `pytest`/`httpx` were never declared
  as dependencies anywhere, so every local test run passed (both happened to be
  pre-installed globally on the dev machine) while a genuinely clean CI install failed
  before a single test could run — meaning the production deploy hook (gated on tests
  passing) never fired for a series of commits. Caught only by directly querying the
  GitHub Actions API instead of trusting local results. Fixed via `requirements-dev.txt`
  + a `tests/test_dashboard.py` fixture that had relied on ambient DB state. This is why
  `qa_engineer` and `release_manager`'s agent definitions both mandate checking the real
  CI run, not just local output — don't let this recur.

## Known limitations (not launch blockers)

- Single-process architecture: in-memory caches and the rate limiter are correct and
  effective for one instance; would need a shared store (Redis) if ever scaled
  horizontally. No current trigger to do this.
- Accessibility and XSS-escaping cover the highest-traffic surfaces by deliberate scope,
  not exhaustively.
- No per-user CLV (closing-line-value) tracking yet (flagged as a strong first
  post-launch candidate).

## How to refresh this file

Run `/status`. Don't hand-edit stale numbers back in.
