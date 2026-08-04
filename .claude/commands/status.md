---
description: Generate or refresh JUICED_STATUS.md — the current, real state of the project.
---

Invoke **manager** to refresh `JUICED_STATUS.md`.

## Workflow

1. Gather real current state, not remembered/assumed state:
   - Test suite: run it, get the real count.
   - CI: check the real latest GitHub Actions run result.
   - Model versions: read `provenance.MODEL_VERSIONS` directly.
   - Model Health: query `/api/model/dashboard` (or call `model_health.dashboard_summary()`
     directly) for real per-sport accuracy/drift status.
   - Recent work: `git log` for what's actually landed since the last status update.
   - Open work: current state of `IMPLEMENTATION_BACKLOG.md`.
2. Update `JUICED_STATUS.md` following its template structure — every number in it must
   trace back to something actually checked in this pass, never carried forward from
   memory of an earlier session.
3. Call out anything that regressed since the last status snapshot (a previously-passing
   check now failing, a metric that got worse) prominently, not buried.

## Output

The refreshed `JUICED_STATUS.md`, plus a short summary of what changed since the last
snapshot (new sections, corrected numbers, resolved/new flags).
