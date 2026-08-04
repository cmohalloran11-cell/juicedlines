---
description: Implement a change end-to-end — routed to the right specialist(s), then verified by qa_engineer.
argument-hint: <task, or a task ID from IMPLEMENTATION_BACKLOG.md>
---

Invoke **manager** to route and sequence implementation of: $ARGUMENTS

## Workflow

1. `manager` identifies the correct owning specialist by domain — never a generalist:
   - Simulation/scoring math (`projector/`, `basketball/`, `tennis/`,
     `projector_bridge.py`, `valuation.py`'s scoring) → **sports_modeling_engineer**
   - API/storage/auth/middleware/grading pipeline → **backend_engineer**
   - `static/*.html` UI/UX → **frontend_engineer**
   - Speed/scale/cost as the primary concern → **performance_engineer**
2. If the task crosses a boundary between components and hasn't already been designed,
   `manager` routes to `architect` first (equivalent to running `/plan`) before any code
   is written.
3. The owning specialist implements the change following `CLAUDE.md`'s standards
   (no fabricated numbers, pure-function `valuation.py`, empirical-Bayes shrinkage
   pattern, model-version scoping, parameterized SQL, `esc()` on untrusted strings,
   etc.) and self-verifies before handing off.
4. A statistical/model change is not complete without `sports_modeling_engineer`'s
   quantitative before/after comparison attached.
5. **qa_engineer** verifies: full test suite, new/updated tests for the change, a live
   check if the change is user-observable, and — if `requirements*.txt` or CI config was
   touched — a genuinely clean-environment reproduction.
6. `manager` updates `IMPLEMENTATION_BACKLOG.md` to reflect completion, and notes
   anything discovered but out of scope for a future pass.

## Output

- What changed, per specialist, with the reasoning.
- `qa_engineer`'s verification result (real counts/output, not a summary claim).
- Anything flagged as needing `/release` next, or as its own follow-up task.

This command may modify application source files — that's its purpose. It should never
skip the `qa_engineer` verification step, regardless of how small the change looks.
