---
description: Review and update ROADMAP.md and IMPLEMENTATION_BACKLOG.md — direction and the concrete queue behind it.
argument-hint: [optional: a new priority or item to fold in]
---

Invoke **manager** (with **architect** for anything structural) to update the roadmap:
$ARGUMENTS

## Workflow

1. Read the current `ROADMAP.md` and `IMPLEMENTATION_BACKLOG.md` as they stand.
2. Reconcile against real project state: what's actually shipped since the last update
   (`git log`, `JUICED_STATUS.md`), what `repository_auditor` findings are still open,
   what a recent `/plan` session produced that hasn't been filed yet.
3. If a new item is being added: `architect` sanity-checks whether it fits existing
   patterns or implies a structural change worth flagging before it's queued.
4. Prioritize by real impact — a finding confirmed by multiple independent angles, or a
   change with a clear quantitative case, ranks above a speculative nice-to-have.
   Explicitly avoid adding speculative work "in case it's needed later" — per
   `CLAUDE.md`, prefer being driven by real usage/data over guessed-ahead scope.
5. Update `ROADMAP.md` (near/mid/long-term direction) and `IMPLEMENTATION_BACKLOG.md`
   (the concrete, owned, sequenced queue) to match.
6. Flag anything that's actually a product/business decision for the user rather than
   silently prioritizing it either way.

## Output

The updated `ROADMAP.md` and `IMPLEMENTATION_BACKLOG.md`, plus a short summary of what
moved, what's new, and what's blocked on a decision only the user can make.
