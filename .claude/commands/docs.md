---
description: Update or audit documentation — README, DEPLOY.md, CLAUDE.md, docstrings, model changelog.
argument-hint: [what changed, or "audit" for a stale-doc sweep]
---

Invoke **documentation_engineer** for: $ARGUMENTS

## Workflow

1. If documenting a recent change: read the actual current behavior (run the command,
   read the current code) before writing — never document from memory of what it used
   to do.
2. Update every doc the change actually affects, not just the most obvious one — a new
   env var needs `DEPLOY.md`'s table; a change to local setup needs `README.md`; a
   model-version bump needs a real `provenance.MODEL_CHANGELOG` entry with the actual
   reasoning, matching existing entries' depth.
3. If run as an audit ("audit" or no specific change given): sweep `README.md`,
   `DEPLOY.md`, and docstrings for drift against current code — a documented command
   that no longer works, an env var that no longer exists, a described behavior that's
   changed. Report drift found even where it's not fixed in this pass.
4. Never remove a documented caveat/limitation without confirming the underlying issue
   is actually resolved.
5. For legal/compliance pages (`static/terms.html`, `static/privacy.html`): if a
   described data practice has changed, update the page and its "last updated" date in
   the same pass — don't let these drift from what the app actually does.

## Output

- What was updated, and how it was verified accurate against current behavior.
- Stale documentation found but out of scope for this pass, reported explicitly rather
  than silently left.
