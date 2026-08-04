---
description: Investigate a domain or the whole repo for real, high-impact issues — read-only, no fixes applied.
argument-hint: [domain: security | performance | testing | reliability | frontend | modeling | all] [optional: specific path/module]
---

Invoke the **repository_auditor** subagent to investigate: $ARGUMENTS

If no domain was given, default to a full production-readiness sweep across security,
performance, testing/CI health, correctness, and reliability — the same scope as this
project's own production-readiness reviews.

## Workflow

1. `repository_auditor` investigates the requested scope using direct code inspection —
   grep for real patterns, read actual functions, run verification commands
   (`EXPLAIN QUERY PLAN`, a clean-environment repro, `pip-audit`, the GitHub Actions API)
   where they'd settle a question cheaply. No speculation, no generic-checklist filler.
2. Every finding is reported with file:line, the concrete failure/cost scenario, and a
   severity grounded in actual exploitability/impact for this specific app.
3. Findings confirmed by more than one independent angle are called out as
   higher-confidence.
4. Anything investigated that turned out fine is reported explicitly, not omitted.
5. Hand the findings to **manager**, who routes each one to the specialist that owns the
   fix and files it in `IMPLEMENTATION_BACKLOG.md` — this command does not fix anything
   itself. Run `/implement` separately once you've reviewed the findings and decided what
   to act on.

## Output

A prioritized findings list (highest impact first), plus a short "investigated, no issue
found" section. If asked for a launch/production-readiness verdict specifically, end with
an explicit yes/no and the reasoning — don't leave it implied.
