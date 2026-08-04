# JUICED — Global Project Instructions

These instructions apply to **every agent, every subagent, and every session** working in
this repository, whether invoked directly or through `.claude/agents/*.md`. Agent-specific
files add detail; they never override what's here. If an agent's own file conflicts with
this document, this document wins.

## What JUICED is

A live sports-projections platform (MLB, WNBA, tennis player props) built on three real
Monte Carlo simulation engines — not a market copy. It pulls lines from PrizePicks,
Underdog, and Sleeper, projects every prop itself, and scores it (Juice Score, EV,
Confidence). A full self-monitoring "Model Health" system tracks the model's own accuracy,
calibration, and drift against graded outcomes. Currently a free public beta — no payment
processing exists yet, by deliberate product decision.

Read `README.md` for the file map and `DEPLOY.md` for how it ships before touching
anything you don't recognize.

## Architecture philosophy

- **Three independent simulation engines, one shared valuation layer.** `projector/`
  (MLB), `basketball/` (WNBA), `tennis/` each fit player rates and simulate a distribution.
  `valuation.py` turns any simulated distribution into EV/Kelly/Confidence/Juice Score as
  **pure functions of already-computed fields on a line dict — no I/O, no DB calls inside
  valuation.py.** Any DB-backed signal must be computed upstream (in `analytics.py`, at
  board-build time) and attached to the line as a plain field. This boundary is
  load-bearing; don't cross it for convenience.
- **Empirical-Bayes shrinkage everywhere a rate is estimated from a small sample.** The
  pattern is `(observed*n + prior*k) / (n+k)` — used in MLB's `regress_to_prior`, WNBA's
  `fit_rates`, tennis's `_shrink`. A new rate estimate with no equivalent shrinkage is a
  regression, not a feature.
- **Two-stage uncertainty**: every engine draws a per-trial parameter-uncertainty
  multiplier (scaled by how much real evidence backs the rate) *before* the outcome draw,
  not just outcome-only variance. See any of the three "Two-stage uncertainty propagation"
  commits in git history for the exact pattern and how it was validated.
- **No fabricated numbers, anywhere, ever.** Every metric shown to a user traces back to a
  real computation from real data. When the ledger is too thin to measure something, the
  answer is `null` + a plain-language "not enough data yet" note — never a plausible-
  looking placeholder, a hardcoded constant dressed up as measured, or a synthetic fallback
  presented as real. This is the single most-enforced rule in this codebase's history;
  violating it is treated as a correctness bug, not a style issue.
- **Model versions are deliberate, not automatic.** `provenance.MODEL_VERSIONS` is
  hand-bumped only on a genuine math change to a sport's engine, specifically so a
  historical graded row stays attributable to the exact logic that produced it, and
  calibration queries (`db.py`, `backtest.py`) never silently blend eras. Every
  calibration/accuracy query in the codebase filters on `model_version`. A new statistical
  change that doesn't consider whether it needs a version bump is incomplete — but bumping
  is also not automatic or free: it orphans graded history under the old version until new
  data accumulates. That tradeoff is a product decision, not a default — see
  `provenance.MODEL_CHANGELOG` for real precedent (bumped vs. deliberately-not-bumped
  cases, both with stated reasoning) before deciding either way, and ask if it's ambiguous.

## Coding standards

- **No comments explaining WHAT the code does** — well-named identifiers already do that.
  A comment earns its place only by explaining a non-obvious WHY: a hidden constraint, a
  workaround for a specific bug, a measured tradeoff, a prior incident. If deleting a
  comment wouldn't confuse a future reader, delete it.
- **Don't reference the current task/fix/caller in comments** ("used by X", "fixed for
  issue #123") — that belongs in the commit message, not the code, and rots as the
  codebase evolves.
- Match the file you're editing. This repo has no single formatter/linter config checked
  in; consistency with the surrounding function is the standard, not a personal
  preference.
- Prefer editing an existing file over creating a new one. Don't add abstractions,
  helpers, or config flags for a hypothetical future need — three similar lines beats a
  premature abstraction.
- Don't add error handling, validation, or fallbacks for scenarios that can't happen.
  Trust internal invariants; validate only at real boundaries (external API responses,
  user input, third-party data).

## Repository rules

- **Never commit a secret.** `.env` is gitignored; real credentials live there or in the
  host's environment variables, never hardcoded, never in a docstring or example. Before
  every commit, scan the diff for anything that looks like a key/token/password — even in
  a comment or test fixture.
- **`requirements.txt` is pinned (`~=X.Y.Z`), not floor-only.** A dependency bump is a
  deliberate PR (Dependabot opens these), reviewed and tested, not a side effect of a
  fresh install. Test-only packages (`pytest`, `httpx`) live in `requirements-dev.txt`,
  never `requirements.txt` — production installs must never pull test tooling.
- **The MLB engine is vendored from a sibling repo** (`../stat-projector`). Local dev with
  the sibling present silently prefers it over the vendored copy in `projector/` — a fix
  made only in one location can be masked entirely in local testing and still ship broken.
  When you touch `projector/models/mlb_model.py`, `montecarlo.py`, or
  `projector/features/mlb_features.py`, check whether the sibling exists locally and sync
  both copies, or verify explicitly (a subprocess-isolated test that can't see the sibling
  is the proven pattern — see `test_vendored_engine_works_without_the_sibling_stat_projector_repo`).
- **Two deploy paths exist and both must keep working**: a static prebuilt-JSON path
  (`build_static.py`, served by Vercel/GitHub Pages) and a full FastAPI server path
  (`main.py`, Render/Railway/Fly/Docker). A change to a shared module can break one path
  while appearing to work in the other — verify both before calling a change complete.

## Testing requirements

- **Local pytest passing is not sufficient evidence that CI passes.** This burned the
  project once already: `pytest` and `httpx` were never declared as dependencies, so every
  local run "passed" only because both happened to be pre-installed globally on the dev
  machine — CI failed on every single push for weeks, silently, because a genuinely clean
  install never had them. Before claiming "tests pass" as a completion criterion, verify
  against a state that doesn't already have your personal environment's leftover
  packages/files (a fresh venv + fresh checkout, or — more directly — the actual GitHub
  Actions run result for the pushed commit, which is checkable via the public API even
  without `gh` CLI: `curl -s "https://api.github.com/repos/<owner>/<repo>/actions/runs?branch=main&per_page=1"`).
- Every new DB-touching test isolates itself with a temp `DB_PATH`/`store` singleton and
  calls `init_db()`/`init_schema()` explicitly — never assume a table already exists
  because it does on your machine. (`tests/test_dashboard.py`'s fixture is the reference
  pattern; it didn't do this for a long time and only broke on a genuinely clean checkout.)
- A statistical/model change ships with a **quantitative before/after comparison**
  (synthetic scenario, live measurement, or both) in the commit message — not just a
  plausible-sounding rationale. "This should improve calibration" is not evidence;
  "SD increased from X to Y for a thin sample, mean unchanged" is.
- Run the full suite (`python -m pytest -q` after `pip install -r requirements-dev.txt`)
  before every commit that touches Python code. Don't ship a red suite.

## Documentation requirements

- `README.md` is the landing doc (what the repo is, quickstart, test instructions) —
  keep it in sync with any change to how the app is run or tested.
- `DEPLOY.md` is the deployment/ops reference (env vars, both deploy paths, rollback
  procedure) — update it in the same commit as any change to deployment config,
  env var defaults, or the rollback mechanism.
- `provenance.MODEL_CHANGELOG` is the model's own audit trail — every version bump gets a
  real entry there, not just a code comment.
- Docstrings on public functions explain contracts and non-obvious behavior (see
  `valuation.py`'s module docstring, or any `db.py` function scoped to `model_version`, for
  the house style) — not a restatement of the function name.

## Implementation philosophy

- **Audit before you fix.** Ground every claimed issue in the actual code (file:line, the
  concrete failure scenario) before proposing a change. Don't invent work, and don't
  report a "finding" you haven't verified still applies to the current code.
- **Prioritize by real, measurable impact** — a finding confirmed independently by more
  than one investigation angle (e.g., both a security review and a performance review
  flagging the same endpoint) is higher-confidence than a single-source guess.
- **Scoped fixes, not sprawling rewrites.** A bug fix doesn't need surrounding cleanup; a
  security fix doesn't need an unrelated refactor riding along. Commit logical groups of
  related changes, not everything-at-once.
- **Verify live, not just in theory.** Where the change is observable (an API response, a
  rendered page, a computed statistic), actually run it and check the real output before
  calling it done — this repo's history has more than one case where "should work" turned
  out to be wrong until someone actually looked.
- **A finding that turns out to already be fixed, or not actually a bug on inspection, gets
  reported as a non-issue** — don't force a change to justify having looked.

## Communication guidelines

- State findings and decisions directly. Don't narrate internal deliberation ("let me
  think about whether...") in output meant for a human or another agent to read.
- A short, complete update beats a long one. Say what changed and what's next; skip the
  guided tour of how you got there.
- When a decision requires product/business judgment (pricing, legal language, scope
  tradeoffs, anything outside "what does the code/data say") — surface it as a decision
  for a human, don't guess and proceed as if it were purely technical.
- When reporting on multiple issues, rank by impact first. Don't bury the one blocking
  finding under five cosmetic ones.
