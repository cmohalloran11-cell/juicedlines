---
description: Ship a verified change to production, and confirm the deploy actually happened — not just that it should have.
argument-hint: [optional: specific commit/change being released]
---

Invoke **release_manager** to release: $ARGUMENTS (or the current state of `main` if
unspecified)

## Workflow

1. Confirm **qa_engineer** has already signed off (full suite passing; a genuinely
   clean-environment check if `requirements*.txt` or CI config changed recently). If not,
   route to `/test` first — this command does not release unverified work.
2. If the change touches `requirements*.txt`, a workflow file, or CI config specifically:
   reproduce the exact CI install+test steps in a fresh venv before pushing, matching
   CI's commands exactly.
3. Push (or confirm the relevant commit is already pushed to `main`).
4. Poll the real GitHub Actions API for the resulting run — wait for `status: completed`,
   then check `conclusion` for **both** the `test` job and the `deploy` job specifically.
   A workflow showing green overall is not sufficient confirmation on its own — the
   `deploy` job is the one that actually matters, and this project has a documented
   incident where it silently never ran despite the workflow "existing."
5. If it failed: investigate for real (job/step status via the API; a local clean-env
   reproduction of the exact failing step) — don't guess at the cause.
6. Report the verified outcome with the actual run ID/URL.
7. If anything requires action outside this repo (a real env var on a hosting
   dashboard, an external service account) — state exactly what's needed and where; this
   command cannot do those steps itself.

## Output

```
qa_engineer sign-off: [confirmed | NOT confirmed — stopping here]
Pushed commit: [sha]
CI run: [URL/ID] — test: [conclusion], deploy: [conclusion]
Needs your action: [external steps only you can do, or "none"]
```
