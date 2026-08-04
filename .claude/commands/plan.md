---
description: Design a change before implementation — contracts, affected components, tradeoffs. No code written.
argument-hint: <what you want to build or change>
---

Invoke **manager** to scope the request, then **architect** to design it: $ARGUMENTS

## Workflow

1. `manager` reads the request and determines whether it's actually cross-cutting
   (touches more than one of: simulation engines, backend, frontend, storage) or
   self-contained enough to skip straight to `/implement` with a specialist.
2. If cross-cutting: `architect` defines the exact contract at every boundary the change
   crosses, checks whether an existing pattern already fits (shrinkage, model versioning,
   the pure-function valuation boundary, the two-deploy-path constraint) before proposing
   anything new, and states any real tradeoff explicitly with a recommendation.
3. Any question that's actually a product/business decision (not a technical one) gets
   flagged for you, not guessed at.
4. `manager` turns the finished design into a sequenced task list, each task tagged with
   the specialist who should own it, and stages it in `IMPLEMENTATION_BACKLOG.md`.

## Output

- The design: contract(s), affected components, which existing pattern it reuses (or
  justification for a new one), tradeoffs with a recommendation.
- A sequenced task breakdown ready for `/implement`.
- Anything flagged as needing your decision before work starts.

No source files are modified by this command — it produces a plan only.
