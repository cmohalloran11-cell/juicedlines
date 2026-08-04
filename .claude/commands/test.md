---
description: Run and verify the test suite — locally, and in a genuinely clean environment when the stakes warrant it.
argument-hint: [optional: specific test file/module, or "clean" to force a fresh-venv reproduction]
---

Invoke **qa_engineer** to verify: $ARGUMENTS (or the full suite if nothing specified)

## Workflow

1. Run the full suite locally: `pip install -r requirements-dev.txt && python -m pytest -q`
   — read the actual pass/fail count, not just whether it printed red text.
2. If a specific module was named, also run its subset directly for fast iteration.
3. If `requirements*.txt`, a CI workflow file, or a new test file was recently touched —
   or the argument explicitly requests it — reproduce in a genuinely clean environment:
   fresh `venv`, fresh checkout, `pip install -r requirements-dev.txt`, `pytest -q`. This
   project has a documented incident where local-only verification missed a dependency
   gap that broke CI for weeks; this step exists specifically to prevent a repeat.
4. If recent commits were pushed, also check the real GitHub Actions result via the API
   rather than assuming a local pass implies a CI pass:
   `curl -s "https://api.github.com/repos/<owner>/<repo>/actions/runs?branch=main&per_page=1"`,
   then the run's `/jobs` endpoint for step-level detail if it failed.
5. Report the exact evidence — counts, run IDs — not a summary claim.

## Output

```
Local: [N passed / M failed]
Clean environment: [checked, result | skipped, because Y]
CI: [checked, run ID + conclusion | not applicable]
Verdict: PASS | FAIL [+ exact failure detail]
```

This command does not fix failures — a failure gets routed back through `manager` to the
specialist who owns the failing code.
