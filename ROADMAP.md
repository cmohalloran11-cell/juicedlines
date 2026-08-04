# JUICED — Roadmap

> **Living document.** Refresh with `/roadmap`. This is direction and reasoning; the
> concrete, sequenced, owned queue behind it lives in `IMPLEMENTATION_BACKLOG.md`. Per
> `CLAUDE.md`: prioritize items driven by real usage, production metrics, or measured
> model-performance data over speculative "might be nice" work.

**Last updated:** 2026-08-03

---

## Near-term (next)

Driven by what real usage and Model Health data surface post-launch, not speculation.

- **Watch Model Health as real games get graded** under the current model versions
  (`mlb-1.2.0`, `wnba-1.1.0`, `tennis-1.2.0`) — interval coverage specifically needs fresh
  graded history to finish re-validating the two-stage uncertainty fix. This is
  observation, not a build task, but it gates whether anything else in calibration needs
  attention.
- **Per-user CLV (closing-line-value) tracking** in Portfolio — flagged in the
  production-readiness review as the highest-value addition for a professional-bettor
  audience; the raw data (logged plays with line/result) is already captured, this is
  aggregation + display.
- **Set `SENTRY_DSN` on the live production host** (currently only verified working
  locally) — an infra step, not a code task.

## Mid-term

Only pursue these once real usage or measured data justifies them — don't build ahead of
evidence.

- **Default side-by-side book comparison** on a prop (today: one line shown, "Compare
  Books" is a click away) — worth it if usage data shows people hitting that action often.
- **Broader accessibility sweep** beyond the primary-navigation fix already shipped —
  worth scoping once there's a concrete need (a specific reported barrier, or a
  deliberate accessibility audit), not proactively.
- **Broader XSS-escaping sweep** beyond the highest-traffic surfaces already covered —
  same logic: real remaining risk is currently assessed as low, revisit if that changes.

## Long-term / explicitly not started

- **Payment/subscription processing.** Deliberately deferred — this launch is a free
  public beta by product decision, not because billing was forgotten. Do not start this
  without an explicit go-ahead; it's a product/business decision, not an engineering
  backlog item waiting for capacity.
- **Horizontal scaling** (multi-instance deploy, Redis-backed shared cache/rate-limiter
  instead of in-memory single-process state). No current trigger — the app runs
  single-instance today and that's correct for current load. Revisit only if/when actual
  traffic data says otherwise; don't build for a scale that doesn't exist yet.
- **Generalizing correlation modeling** to markets/sports without real measured
  correlation data yet. Per `CLAUDE.md`: never fabricate a correlation constant — this
  waits on either real data becoming available or a deliberate measurement-pipeline
  project, not on engineering time alone.

## Decisions already made (don't re-litigate without new information)

- Free public beta first, billing later — explicit product decision.
- WNBA `wnba-1.1.0` model version was bumped covering two-stage uncertainty + combo
  correlation together; the earlier minutes-shrinkage fix was deliberately *not* bumped
  on its own (narrow effect, would have orphaned ~6,790 rows of still-valid calibration
  history for no real benefit). See `provenance.MODEL_CHANGELOG` for the full reasoning
  — this is the reference precedent for any future close call on whether a change
  warrants a version bump.
- No Slack/email/external notification channels for Model Health regressions yet —
  internal measurement/monitoring was built first, deliberately, so a notification
  channel can be layered on later without re-architecting the measurement layer.

## How to refresh this file

Run `/roadmap`. New items get sanity-checked by `architect` for fit against existing
patterns before being queued in `IMPLEMENTATION_BACKLOG.md`.
