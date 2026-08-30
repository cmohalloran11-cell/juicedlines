# 00 — Final Summary

Four-phase run: diagnosis, Juice Score rebuild, model improvements, add College Football.
Commits `29bf81f`..`54e8a3d` on `claude/juiced-model-improvement-r6jnoe`, one commit range per
phase (plus one small cross-phase backend fix, §4). 62 files changed, +7,444/−50 lines total.
Full detail lives in `reports/01-diagnosis.md`, `reports/02-juice.md`, `reports/03-improvements.md`,
and `cfb/README.md`; this file is the cross-phase index.

**The single fact that shaped every phase**: this session's only accessible real graded-outcome
data is `clv_seed.db` — 3,312 MLB rows over a 13-day window (2026-06-29→07-12). The real
production ledger (`history.db`) is gitignored and was never reachable from this container.
WNBA/Tennis/NFL have **zero** accessible graded rows; CFB (new this run) has zero by
definition. Nothing below claims a live-accuracy result beyond what that one 13-day MLB slice
can honestly support — everywhere else, the report says "unmeasurable" instead of guessing.

---

## 1. Phase 1 — Diagnosis (`29bf81f`, `reports/01-diagnosis.md`)

No model changes. Findings:

- Ledger inventory: MLB 3,446 rows / 3,312 graded / 13-day window; WNBA/Tennis/NFL 0 rows.
  `model_raw`/`model_raw_prob`/`trust_weight` are 100% NULL in the accessible MLB data —
  traced to the fact that MLB's per-stat anchoring (`_mlb_trust()`) went live 2026-07-21,
  *after* this data ends, so `close_proj` is a valid pre-anchor `m` proxy for this specific
  window (not evidence the logging path is broken going forward).
- sd(m−L) per stat type (MLB): no stat literally triggers R1's 0.15 threshold; `Doubles`
  flagged NO-SIGNAL in spirit (model outputs a literal constant, 0/0 degenerate) and descoped.
- Edge regression: pooled γ=0.306, CI **[−0.120, 0.650]** (crosses 0, n=3,312, dominated by
  mixed stat scales). Per-stat: only Hits (γ=0.629, CI[0.385,0.854]), Home Runs (γ=0.510,
  CI[0.108,0.986]), Hits+Runs+RBIs (γ=0.533, CI[0.149,0.937]), Runs Allowed (γ=1.597,
  CI[0.134,2.991]) exclude zero.
- MLB trade-deadline degradation test: **UNMEASURABLE** — accessible data ends 2026-07-16,
  15 days before the 2026-07-31 deadline. R3 never fired; counter-bias layer left unchanged.
- WNBA pace mismatch (config `league_pace=96` vs. computed ~80–85): traced and found
  **already fixed** — used only as a relative matchup adjustment, doesn't leak into
  projection level, measured before/after already on record in-code (−2.76 bias when tried
  raw). R4 did not fire.
- t=0/t<0.2 fraction: **unmeasurable** — `trust_weight` unpopulated in every accessible row.

## 2. Phase 2 — Juice Score rebuild (`c9e2a3b`, `reports/02-juice.md`)

Rebuilt as one signed value: `juice = 100·sign(e)·S(|e|)·c`, `e = p − b`, all inputs
pre-anchor. `z`/`g` (normalized projection differential / skew gap) are computed on every
prop as the coherence gate but **do not scale the score** — the spec's own ablation rule
fired: e-only beat e×z on every cut tested (AUC 0.6997 vs 0.6970 pooled and on 6/8 stat
types; decile Spearman +0.981 vs +0.976).

**Validation (MLB, n=2,677 of 3,312 graded, Wilson CIs):**
- Squash `S` fit by KS-minimization against Uniform[0,1] over the 3,312 real rows: KS=0.031
  (still above the n's 5% critical value 0.024 — reported as "roughly uniform," not proven).
- Signed deciles: Spearman **+0.988**. Top decile OVER hit rate **56.7%** CI[50.7,62.5] vs.
  50% break-even; bottom decile UNDER 93.3%.
- Per-stat-type: all 8 positive, none strictly monotone (weakest: Walks +0.300).
- **WNBA/Tennis/NFL: UNMEASURABLE, no monotonicity table fabricated for them** (zero graded
  rows). `c` and `g`/mean-median split are also unmeasurable from the ledger (only one
  central projection value was ever stored historically).

**Shipped flag-gated OFF** (`JUICE_VERSION=1` default). Three stated reasons: 3 of 4 sports
UNMEASURABLE, MLB's only window is R5-blocked (13 days), and `static/dashboard.html` assumed
an unsigned never-null score at ~20 call sites (named as a prerequisite, not done in this
run). **Model version not bumped** — purely additive read-only fields, no calibration query
reads the new fields, nothing to orphan.

**Found, not fixed**: Tennis measures its stat-gamma off the *anchored* `close_proj`, not a
raw `model_raw` — the exact self-feedback risk `db.py:353-354` warns about. Needs its own
version-bump decision; flagged for a human/architect follow-up, not smuggled in.

Tests: 472 passed, fresh venv, under both `JUICE_VERSION` values.

## 3. Phase 3 — Model improvements (`f9fb538`, `reports/03-improvements.md`)

Audited MLB/WNBA/Tennis recency-weighting and regime-break handling per Phase 1's priority
order (WNBA/Tennis/NFL skipped — UNMEASURABLE, R2; MLB constrained by R5's 13-day window).

**One bug fixed**: WNBA's `EspnBasketball._athlete_gamelog` returned games unsorted; both
consumers (`fit_rates`, `project_minutes`) weight purely by list index, so an unsorted feed
doesn't degrade recency weighting, it **inverts** it. Fixed with a sort, matching the other
three data sources which already sort for this reason. Proof (3-game fixture, correctness
test, not a live-accuracy claim): `fit_rates` pts/poss **0.0992 → 0.2302** (unweighted
reference 0.1574); tests failed pre-fix (2/5), passed after (5/5).

**Found, measured, deliberately not shipped** (below noise or arithmetically inert, not
worth a version bump on a 13-day sample this project's own R5 forbids tuning against):
1. MLB `_eff_pa` ignores the recency blend it describes — measured effect (8M sims):
   SD +0.10%, mean unchanged.
2. MLB/WNBA draw `Gamma(exposure)` where the shape should be the outcome count — understated
   CV up to 5.0× for HR; measured effect ≤+2.74% SD, ≤0.47pp P(over).
3. MLB's two-stage uncertainty layer is near-inert at realistic sample sizes (6×2M-sim
   seeds): dropping effective sample 544→92 moves SD only +0.07% to +0.49%, seed SE
   0.02–0.05% — arithmetic (outcome variance dominates parameter variance ~100:1 at 4.4
   PA/game), not a defect.

**Model version not bumped, any sport** — the fixed WNBA path is currently unwired (ESPN
403s in production; `balldontlie` is the live default source), so the fix affected zero
graded rows.

Tests: 474 passed, fresh venv, both `JUICE_VERSION` values.

## 4. Phase 4 — Add College Football (`01e7285`, `b133e5a`, `88e2fd1`, `54e8a3d`)

New sport, four commits (data layer → engine → frontend → one small cross-cutting fix), no
`CFBD_API_KEY`/`ODDS_API_KEY` anywhere in this environment — **nothing here was ever
verified against a live CFBD or Odds API response.** Everything below is a code-correctness
claim, not a calibration claim.

- **`01e7285` (data/plumbing)**: `cfb/` package — CFBD REST client, swappable `OddsProvider`
  interface + The Odds API adapter (player props; CFBD carries none), canonical player table
  with CFBD-id fuzzy-match mapping + review log (reused `fantasy/`'s pattern rather than
  reinventing it), manual player-status override table, and the full `prop_clv` ledger
  schema with pre-anchor fields logged **from day one** (the exact gap Phase 1 found MLB
  went without for months). Registered in `provenance.py` (`cfb-0.1.0`, "plumbing only"),
  `main.py`, `books.py`, `analytics.py`, `model_health.py`. One test-fixture bug fixed
  (wrong line-dict key name) and the missing `players_sync` test added. 531 passed.
- **`b133e5a` (engine)**: `cfb/model/` + `cfb/sim/` — garbage-time blowout-probability layer
  (symmetric snap discount for heavy favorites *and* underdogs), explicit plays×usage×
  efficiency pace model, empirical-Bayes 3-tier prior fallback (A: returning production, B:
  transfer with a measured FCS/G5→P4 level discount, C: recruiting rating), opponent
  down-weighting fit against CFBD's efficiency metrics. Mechanism proofs (planted-effect
  recovery / hand-computed closed forms, explicitly **not** accuracy claims): two-stage
  uncertainty SD +63% when effective sample drops 2000→12 (mean unchanged); garbage time
  −9.45% carries at +35 spread, identical magnitude at −35; fitted opponent coefficient
  recovers a planted 1.25× FCS effect to within 0.06; median-anchoring returns exactly 0.500
  probability at zero trust (the same regression guard NFL shipped). Model version bumped
  `cfb-0.1.0→cfb-1.0.0` — orphans nothing (0.1.0 never wrote a projection). **Found, not
  fixed**: the Odds API adapter's outcome-shape assumption (`point` + over/under) doesn't
  match anytime-TD markets' documented `Yes`/`No` shape, so `player_anytime_td` can't reach
  the board yet — flagged in `cfb/README.md` rather than guessed at without a real response
  to verify against. 566 passed.
- **`88e2fd1` (frontend)**: `static/dashboard.html` — CFB wired into every sport
  selector (Projections, Model Health, Entry Optimizer), a CFB Prior Tier filter
  (returning/transfer/freshman), a drawer section + row badge for the tier, the required
  `"Data provided by CollegeFootballData.com"` attribution shown inline whenever CFB is
  selected, and honest "not enough data yet" treatment everywhere Model Health/Scorecard
  would otherwise imply a track record. Verified live via headless Playwright against both
  a running server and static-fixture JSON (not against a real CFBD/Odds response — none
  has ever existed). Found a real gap: `dashboard.py`'s `_drop()` never exposed
  `proj_kind`/`prior_tier_reason` to any sport's frontend consumer, so the new tier badge
  had no data to read.
- **`54e8a3d` (fix, this session)**: closed the gap above — added `projKind`/
  `priorTierReason` to `dashboard.py`'s `_drop()` and `prior_tier_reason` to
  `build_static.py`'s `_KEEP`/`_PREMIUM_FIELDS`, so both deploy paths carry it. Verified a
  synthetic CFB line round-trips correctly. 566 passed.

---

## Sports skipped, and the rule that skipped them

| Sport | Rule | Why |
|---|---|---|
| WNBA | R2 | 0 graded rows accessible in this environment, every phase |
| Tennis | R2 | 0 graded rows accessible in this environment, every phase |
| NFL | R2 | 0 graded rows accessible in this environment, every phase |
| CFB | R2 (implicit) | New sport, UNMEASURABLE by definition (task spec item 8) — 0 graded rows possible, and no live API access ever, this run |

MLB was never skipped, but R5 fired against it (§ below) — no tuning decision in any phase
used its 13-day window as justification.

## Decision rules that fired, across all four phases

| Rule | Fired? | Where / outcome |
|---|---|---|
| R1 (sd(m−L)<0.15·m_sd → NO-SIGNAL) | Yes, in spirit, once | MLB `Doubles` (Phase 1) — degenerate constant projection, descoped from Phase 3 |
| R2 (<400 graded → UNMEASURABLE) | Yes | WNBA/Tennis/NFL every phase; CFB by definition (Phase 4) |
| R3 (ablation shows removal helps → remove) | Never | MLB counter-bias layer's ablation could not run (no post-deadline data) — layer preserved unchanged through Phase 3 |
| R4 (WNBA pace leaks into level → fix priority one) | No | Traced and found already fixed (Phase 1) — non-issue, no fix forced |
| R5 (never tune <28d or <400 outcomes) | Yes | MLB's entire accessible window is 13 days — blocked every point-estimate-driven tuning decision in Phases 1 and 3 |

## Before/after measurements, with n (everything claimed as "improved" or "validated" anywhere in this run)

| What | n | Before | After |
|---|---|---|---|
| Juice Score signed-decile monotonicity (MLB) | 2,677 scored / 3,312 graded | — (new metric) | Spearman +0.988; top decile 56.7% CI[50.7,62.5] vs 50% breakeven |
| Juice squash-function fit (KS vs U[0,1]) | 3,312 | exponential-MLE alt: KS 0.110 | Weibull fit: KS 0.031 |
| z-component ablation (pooled AUC) | 2,677 | e×z: 0.6970 | e-only: 0.6997 → simplified to e-only |
| WNBA recency-weighting fix (synthetic fixture) | 3-game fixture | pts/poss 0.0992 (below unweighted ref 0.1574) | pts/poss 0.2302 |
| MLB `_eff_pa` recency-blend fix (simulated, not shipped) | 8M sims | SD 0.63631 | SD 0.63697 (+0.10%, below noise, not shipped) |
| MLB two-stage layer sensitivity (simulated) | 6×2M-sim seeds | eff. sample 544 | eff. sample 92 → SD +0.07–0.49%, arithmetic, not a defect |
| CFB two-stage uncertainty (planted-effect proof) | 200k trials | eff_n=2000: SD 25.67 | eff_n=12: SD 41.93 (+63%, mean unchanged) |
| CFB garbage-time discount (planted-effect proof) | — | no discount | +35 spread: −9.45% carries; −35 spread: same magnitude |
| CFB opponent-adjustment fit (planted-effect recovery) | — | planted 1.25× FCS coefficient | recovered 1.25× (to 0.06); planted 0.60 PPA slope → recovered 0.652 |

Every row above is real and reproducible in this repo's current test suite. None is a
live-market-accuracy claim except the Juice Score's MLB decile table, which is explicitly
scoped to a 13-day, single-sport, R5-flagged window.

## What could not be measured, and why (the important list)

- **All real-time/live-accuracy claims for WNBA, Tennis, NFL, and CFB** — zero graded rows
  accessible anywhere in this environment for any of them, in any phase.
- **MLB trade-deadline degradation** (bias/Brier/γ split, counter-bias-layer ablation) —
  accessible data ends 2026-07-16, 15 days before the 2026-07-31 deadline. No amount of
  reasoning substitutes for the missing 6+ weeks of data; genuinely impossible here.
- **MLB tuning beyond the 13-day window** — R5 blocks it by the project's own rule,
  independent of n (3,312 ≫ 400, window still only 13 days).
- **t=0/t<0.2 fraction, any sport** — `trust_weight` is unpopulated in every row this
  session could reach.
- **Juice Score's `c` (confidence) and `g` (skew gap) validated against real history** — the
  ledger only ever stored one central projection value per historical row; `g` needs both
  mean and median, `c` needs inputs that aren't ledger columns at all.
- **CFB — everything.** No `CFBD_API_KEY`/`ODDS_API_KEY` has existed in any environment this
  code has run in. Every number in Phase 4's engine report is a planted-effect recovery or a
  hand-computed closed form proving the *mechanism* works, never a live accuracy number.
  `model_health`/`backtest` will correctly report `insufficient_data` for CFB until real
  games grade under `cfb-1.0.0`.
- **GitHub Actions CI, every phase** — this session's GitHub API access returned "not
  enabled." Every phase substituted the strongest available check: a fresh Python
  venv + `pip install -r requirements.txt -r requirements-dev.txt` + `pytest -q`,
  reproducing CI's own steps exactly, rather than claiming an unverifiable CI pass.

## Open items for a human (not resolved in this run, by design)

1. **MLB counter-bias / stat-gamma shrink layer**: never validated (no post-deadline data),
   never removed (removing on zero evidence is its own unvalidated risk). Needs real
   `history.db` access to resolve either way.
2. **Tennis's stat-gamma measured off anchored `close_proj`, not raw `model_raw`** (Phase 2
   finding) — a real self-feedback risk, needs its own model-version-bump decision.
3. **`JUICE_VERSION=2` flip** — validated as far as data allows, blocked on a frontend pass
   over ~20 call sites in `static/dashboard.html` that assume the old unsigned/never-null
   score shape. Not attempted in this run (out of scope for Phase 4's frontend work, which
   only added CFB's new UI, not migrated the existing juice display).
4. **CFB's `player_anytime_td` market** — cannot reach the board today (Odds API adapter
   shape mismatch); needs one real API response to fix correctly rather than guess at.
5. **A regime-break detector for recency weighting** (Phase 3's audit) — needs a threshold
   fitted against outcomes that don't exist in this environment, and is blocked further
   upstream: WNBA/MLB currently discard the per-game team context a regime-break detector
   would need. Flagged as a data-plumbing prerequisite, not attempted.
