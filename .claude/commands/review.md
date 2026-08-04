---
description: Review a change (diff, PR, or recent commits) for correctness and adherence to project standards before it's trusted.
argument-hint: [optional: specific diff/PR/commit range — defaults to the current uncommitted diff]
---

Review: $ARGUMENTS

## Workflow

1. Identify what changed and which domain(s) it touches.
2. Route to the specialist who owns that domain for a standards review — **not** a
   different specialist rubber-stamping unfamiliar code:
   - Simulation/scoring changes → **sports_modeling_engineer** (quantitative evidence
     present? correct shrinkage/uncertainty pattern? model-version implications
     considered?)
   - Backend/API/storage changes → **backend_engineer** (parameterized SQL? auth
     appropriate to the action? `valuation.py`'s no-I/O boundary intact? works against a
     freshly-initialized DB?)
   - Frontend changes → **frontend_engineer** (untrusted strings escaped? displayed
     numbers stay consistent with each other? no console errors? mobile-safe?)
3. **qa_engineer** independently confirms the test suite actually passes (not just that
   the specialist claims it does) and checks for the specific failure classes this
   project has hit before: ambient test state, an undeclared dependency, a change that
   only works on one of the two deploy paths.
4. If the diff touches a shared boundary or contract, **architect** checks it against the
   existing design rather than just the local diff.
5. Report findings the same way `/audit` does: file:line, concrete scenario, severity —
   not vague "consider improving X" comments.

## Output

A findings list (if any) ranked by severity, plus an explicit verdict: ready to merge/
ship, or blocked with the specific reason. If blocked, route the finding back through
`manager` to the owning specialist for a fix, then re-review — don't fix it inline as
part of the review itself.
