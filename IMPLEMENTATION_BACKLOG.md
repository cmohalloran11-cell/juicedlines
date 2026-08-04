# JUICED — Implementation Backlog

> **Living document.** The concrete, sequenced, owned queue behind `ROADMAP.md`'s
> direction. Refresh with `/roadmap`; individual items move through here as `/implement`
> and `/audit` produce and complete work. Every item needs an owning agent before it's
> actionable — an unowned item isn't ready to start.

**Last updated:** 2026-08-03

---

## How to read this file

- **Status**: `queued` (not started) → `in progress` → `blocked` (needs a decision or
  external input) → `done` (verified by `qa_engineer`, not just implemented).
- **Owner**: the specialist agent whose domain this is — see `.claude/agents/`.
- **Source**: how this item was identified (`/audit` finding, `/plan` design output,
  direct user request, `ROADMAP.md` near-term item).
- An item only moves to `done` after `qa_engineer` has verified it — an implementing
  specialist's own report of completion is not sufficient per `manager`'s rules.

---

## In progress

*(none — populate as `/implement` starts work)*

## Queued

| # | Item | Owner | Source | Priority |
|---|---|---|---|---|
| 1 | Per-user CLV tracking in Portfolio | `backend_engineer` + `frontend_engineer` | `ROADMAP.md` near-term | High |
| 2 | Set `SENTRY_DSN` on production host | `release_manager` (flags to user — needs host dashboard access) | Launch follow-up | High |

## Blocked (needs a decision or external input)

*(none currently — an item goes here when it needs product/business judgment, external
credentials, or another team's action before a specialist can proceed)*

## Done (recent)

| Item | Owner | Verified by | Notes |
|---|---|---|---|
| Fix CI: `pytest`/`httpx` never installable by CI | `release_manager` | `qa_engineer` (clean-venv repro + real GitHub Actions run check) | Production deploy hook had been silently blocked for a series of commits — see `JUICED_STATUS.md`'s "Recent incidents" |
| Fix `test_dashboard.py` relying on ambient DB state | `qa_engineer` | `qa_engineer` (clean-checkout repro) | Same investigation as above |
| Sentry error monitoring integration | `backend_engineer` / `release_manager` | Live test event captured + flushed with a real DSN | Optional, env-gated, test runs explicitly excluded |
| Terms of Service / Privacy Policy / Responsible Gaming disclaimer | `documentation_engineer` / `frontend_engineer` | Live page load + console check | Launch-readiness item |
| Dependency vulnerability scanning (Dependabot + `pip-audit`) | `release_manager` | Clean `pip-audit` run | Non-blocking in CI by design |
| Deployment rollback procedure documented | `release_manager` / `documentation_engineer` | N/A (documentation) | `DEPLOY.md` |
| Integration tests for primary public API endpoints | `qa_engineer` | Self (15 new tests passing) | `tests/test_api_endpoints.py` |
| Two-stage uncertainty propagation, all three sports | `sports_modeling_engineer` | Quantitative before/after per sport | See `provenance.MODEL_CHANGELOG` |
| Measured WNBA combo correlation (replacing incidental-only correlation) | `sports_modeling_engineer` | Live measurement + regression test | |
| Full Model Health self-monitoring framework | `backend_engineer` / `sports_modeling_engineer` | Live dashboard + API checks | |
| Production-readiness security/performance/UX/deployment fixes | multiple | `qa_engineer` per fix | ~16 scoped commits, see git history |

---

## Adding a new item

1. State it in one line: what, not how.
2. Assign an owner by domain (see `.claude/agents/` descriptions) — if it's genuinely
   cross-domain, run `/plan` first and let `architect` break it into per-owner sub-items.
3. Note its source.
4. Don't add speculative work without a real trigger (a user request, an `/audit`
   finding, measured data) — see `CLAUDE.md`'s implementation philosophy.
