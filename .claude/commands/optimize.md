---
description: Investigate and fix a speed/scale/cost issue — always measured before and after.
argument-hint: <endpoint, page, or operation that's slow/expensive>
---

Invoke **performance_engineer** to investigate and fix: $ARGUMENTS

## Workflow

1. Measure the actual current cost first — `EXPLAIN QUERY PLAN` for a SQL-bound issue,
   wall-clock timing for a Python function, network/render timing for a frontend
   concern. No optimization work starts without a real baseline number.
2. Identify the specific cause: unindexed query, uncached hot path with low volatility,
   a blocking call inside an `async def` route not wrapped in `run_in_executor`, a
   redundant client-side fetch, or (for simulation code) a genuinely slow loop that can
   be vectorized without changing output.
3. Implement the fix, scoped narrowly.
4. If the fix touches simulation/scoring code (a DP, a Monte Carlo loop, anything in
   `projector/`, `basketball/`, `tennis/`): **sports_modeling_engineer** confirms
   bit-exact or statistically-equivalent output before/after — a performance fix that
   silently changes a projection is not acceptable.
5. Measure again with the same method to confirm a real improvement.
6. **qa_engineer** confirms no behavior regression via the test suite.
7. Note explicitly if a further optimization exists but isn't warranted at current scale,
   and what would trigger revisiting it (e.g. "only matters once running >1 instance").

## Output

```
Before: [measurement]
Fix: [what changed]
After: [measurement, same method]
Behavior unchanged: [how confirmed]
Deferred: [anything not worth doing yet, and its trigger condition]
```
