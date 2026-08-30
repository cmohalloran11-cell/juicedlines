# Phase 2 — Juice Score rebuild

**Shipped behind a flag (`JUICE_VERSION`, default `1` = the old score).** The rebuilt score
is implemented, tested, wired into both deploy paths, and validated as far as the available
data allows — which is MLB only, over 13 days, i.e. a window this project's own tuning rule
(Phase 1 R5) forbids acting on. Reasoning for the flag is in §7.

---

## 0. Data-access disclosure (inherited from Phase 1, re-verified)

The real production ledger (`history.db`) is gitignored and not present in this container.
The only real graded-outcome data reachable anywhere in this environment is
`clv_seed.db.prop_clv`: **3,446 rows, 3,312 graded, MLB only, `game_date` 2026-06-29 →
2026-07-12**. WNBA / Tennis / NFL have **zero** graded rows. There is also no outbound
network to any sports/book API from here (`statsapi.mlb.com` and
`api.underdogfantasy.com` both return `403 Tunnel connection failed` through the proxy), so
a fresh live board pull was not possible; the "live" run in §5 is the shipped
`valuation.juice_v2()` executed end-to-end over those 3,312 real graded rows, plus a
full-stack smoke run through `db.log_clv` → `dashboard.build` → `build_static` fields.

Two corrections to assumptions carried into this phase:

* `close_over_implied` / `close_under_implied` are **100% NULL** in this table, not
  "present". Every graded row is Underdog, `odds_type` NULL.
* `close_prob_b` is **not** a de-vigged book probability. `db.py:66-73` documents it as
  **A/B variant B's model probability** (engine + matchup context + xBA prior). It was not
  used as `b`.
  So `b = 0.5` for every validation row — correct for a pick'em book, whose flat payout is
  identical on both sides and therefore cannot move which side is right.

---

## 1. The score

One **signed** value in `[-100, +100]`. Positive = over, negative = under, near zero = no
play, `null` = no model opinion to score. Implemented as pure functions in `valuation.py`
(`juice_v2`, `juice_v2_factors`, `breakeven_prob`, `juice_confidence`,
`audit_juice_coherence`) — no I/O, no DB calls, no clock read.

### Inputs — all PRE-anchor

| symbol | field on the line | source |
|---|---|---|
| `p` | `model_pre_prob` | P(over `line`) from the **un-anchored** simulated sample array (Platt-calibrated on MLB, same as `model_prob`) |
| `b` | — | `breakeven_prob(line)`: de-vigged `over/(over+under)` when the book prices both sides, `0.5` for pick'em, `None` (→ null juice) for demon/goblin |
| `m_med` | `model_pre_median` | median of the un-anchored array |
| `m_mean` | `model_pre_mean` | mean of the un-anchored array |
| `m_sd` | `model_pre_sd` | SD of the un-anchored array |
| `L` | `line` | posted line |
| `c` | — | `juice_confidence(line)` ∈ [0,1] |
| `t` | `model_anchor_t` | **total** anchor weight left on the model |

Why pre-anchor matters concretely: every engine produces `t·model + (1−t)·line`, so
`model_proj − line` is mechanically shrunk by `t` and is identically zero at `t = 0`. Scoring
the anchored number measures how much the board trusted the model, not what the model said.
Threading these four moments was the bulk of the plumbing work — see §6.

`t` is the **product** of both anchors on MLB (the engine's per-stat `trust` from
`_mlb_trust()` **times** `attach_stat_trust`'s γ shrink, since both blend toward the same
line). Reading only one of the two overstates the surviving model signal.

### Components

```
e = p − b                          probability edge
z = (m_med − L) / m_sd             normalized projection differential (SD units)
g = (m_mean − m_med) / m_sd        skew gap
c = cbrt(sample · availability · reliability)
```

`c` deliberately excludes decisiveness (unlike `confidence_score`), because decisiveness *is*
`|e|` — multiplying them would square the edge. Its three factors:
`sample = clamp(model_n / 30)` (the saturation `confidence_score` has shipped since
2026-07-29), `availability` ∈ {1.0 confirmed/unknown, 0.5 questionable, 0.0 out},
`reliability = clamp(2 · stat_trust_gamma)` (the measured per-stat edge-regression slope from
`db.stat_gammas`; the neutral default 0.5 → 1.0, so an *unmeasured* stat is never punished as
if it were a *proven-bad* one). Geometric mean, so a stat the model has proven zero edge on
cannot be rescued by a deep sample.

### Scoring

```
juice = 100 · sign(e) · S(|e|) · c
S(x)  = 1 − exp( −(x / 0.2033) ** 1.384 )
```

`z` and `g` are still computed on every prop — they are what the coherence gate tests `p`
against — but they **do not scale the score**. That is a measured decision, not a
simplification for its own sake: see §4.

---

## 2. Fitting the squash function

`S` is a Weibull CDF, monotone, `S(0)=0`, `S(∞)=1`. Both constants were **fit**, not chosen:
a 2-D grid search minimising the Kolmogorov–Smirnov distance between `S(|e|)` and
`Uniform[0,1]` over the **3,312 real graded MLB close snapshots** in `clv_seed.db`.

| candidate family | parameters | KS vs U[0,1] |
|---|---|---:|
| Exponential (1-param, MLE `λ = mean|e| = 0.1815`) | λ | 0.1098 |
| **Weibull (2-param, KS-fit)** | **shape 1.384, scale 0.2033** | **0.0310** |

Resulting `S(|e|)` deciles: `0.078 / 0.183 / 0.284 / 0.389 / 0.488 / 0.579 / 0.710 / 0.823 /
0.912` against the `0.1 … 0.9` an exactly-uniform score would produce.

**Honest caveats, stated because the data is thin:**

* KS = 0.031 still **exceeds** the n=3,312 5% critical value of 0.0236. This is "roughly
  uniform", not "proven uniform". The spec asked for roughly uniform; that is what was
  achieved and that is what is claimed.
* Fit on **MLB only** — no other sport has a single graded row in any accessible ledger. The
  constants are shared across all sports today because `e` is a probability and therefore
  dimensionless/portable, but they are **not validated** for WNBA/Tennis/NFL. Refit per sport
  once each has ≥28 days / ≥400 graded outcomes of its own.
* The **uniformity target was `S(|e|)`, not `|juice|`**, because `c` could not be evaluated on
  the ledger at all (`model_n`, `lineup_status` and `stat_trust_gamma` are not columns in
  `prop_clv`). `c` is a monotone shrink in [0,1], so `|juice|` on a live board will be
  left-shifted relative to `S(|e|)`. That distribution is genuinely unmeasured, and no number
  is claimed for it.
* Fit on a 13-day window. Phase 1's R5 says do not treat a pattern in this window as a tuning
  signal. Fitting a display-scale transform is not the same act as tuning a projection, but
  the constraint is real and is the main reason the score ships flag-off.

The z-squash used in the ablation (§4) was `1 − exp(−|z|/0.5450)`, the exponential MLE on the
same rows. It is not in the shipped code.

---

## 3. Null handling and the coherence gate

`juice_v2` never returns a plausible-looking small number in place of "we don't know". Every
null path carries a machine-readable `reason`:

| `reason` | condition | why null rather than a low score |
|---|---|---|
| `no_model_signal` | `t < 0.2` | Every board hard-defers to the market line below 0.2 (`basketball/board.py:119`, `tennis/board.py:163`, `nfl/board.py:219`). The projection **is** the market's line. There is nothing of the model's left to score. |
| `degenerate_distribution` | `m_sd ≤ 0` | A zero-spread "distribution" is a stub. Phase 1 found MLB `Doubles` emitting one identical constant across all 90 graded rows. No scale to normalize against, no real opinion. |
| `unpriced` | demon/goblin | The boosted multiplier is not in the feed, so `b` cannot be computed. Inventing one would be a fabricated number. |
| `no_pre_anchor_probability` / `no_distribution_moments` | fields absent | The integrity check cannot run without both moments; scoring without it is worse than not scoring. |
| `coherence_fault` | see below | **Model integrity error** |

### `coherence_fault`

Raised when the model's own `P(over)` and its own median-vs-line displacement point in
opposite directions by **more than the distribution's skew explains**:

```
fault  ⟺  sign(p − 0.5) ≠ sign(z)  and  |z| > |g|
```

`juice = null`, `coherence_fault` carries the full diagnostics (`p, b, e, z, g, direction,
m_mean, m_median, m_sd, line, player, stat_type, sport, proj_kind, id`),
`dashboard._projected()` drops the line from every user-facing surface, and
`valuation.audit_juice_coherence()` collects them into the build's review queue —
printed as a `::group::` block by `build_static.py`, the same flag-don't-block pattern
`audit_ev` already uses. It runs regardless of `JUICE_VERSION`, because the integrity check is
worth having even while the score is flagged off.

**One deliberate deviation from the spec's literal wording.** The spec says to test
`sign(e)` vs `sign(z)`. The gate tests `sign(p − 0.5)` vs `sign(z)` instead. Reason: `e = p − b`
mixes the model with the *market's price*. With a de-vigged `b = 0.60`, a model at `p = 0.58`
whose median sits above the line has `e < 0` and `z > 0` — but that is not the engine
contradicting itself, it is the book having priced past the model's lean. Testing `e` would
fault every such prop. `p − 0.5` versus the median-vs-line displacement is the model's genuine
internal consistency, and is exactly the invariant `dataos.direction_report` already enforces
at build time (`Projection > Line ⇒ P(Over) > 50%`), restated in SD units with a skew
tolerance. Where `b = 0.5` (every row in the validation set) the two tests are identical, so
this does not affect any number below.

### Skew is what makes the gate non-trivial

For a right-skewed counting stat the mean sits above the median, so **"mean above the line but
`p < 0.5`" is correct model behaviour** when the book prices at the median. That case is
handled twice: `z` is built on `m_med` (not `m_mean`) so it does not arise in the first place,
and a residual sign disagreement smaller in SD units than `g` is treated as explained. Both
behaviours are pinned by tests (`test_no_fault_when_the_distribution_skew_explains_the_disagreement`,
`test_coherence_fault_when_probability_and_median_disagree_beyond_skew`).

### Near-lock availability

`minutes_to_lock ≤ 60` **and** `lineup_status` unknown (not `in` / `questionable` / `out`) →
`stale = True` and `|juice|` capped at 50 (`capped = True` when the cap actually bit). The
clock is read once at board-build time by `analytics.attach_lock_clock()` (injectable `now`,
called from **both** `main.py` and `build_static.py`) so `valuation.py` stays pure and
deterministic. **The cap value of 50 is a product safety rule, not a measured constant, and is
labelled as such in the code** — there is no data anywhere in this environment from which a
"correct" cap could be fit, and the honest thing is to say so rather than dress a policy
choice up as a measurement.

---

## 4. Validation — z-ablation

**The spec's own decision rule fired: the full `e × z` score does not beat the `e`-only score,
so the score was simplified to `e`-only.**

Measured on the 2,677 coherent rows the shipped scorer covers (identical `c = 1` on all of
them, so this isolates the `e`/`z` core):

| variant | AUC (score vs. over/under outcome) | Spearman(decile, over-hit%) |
|---|---:|---:|
| **`e`-only (shipped)** | **0.6997** | **+0.981** |
| `sign(e)·√(e_norm·z_norm)` (spec draft) | 0.6970 | +0.976 |
| `z`-only | 0.6844 | +0.969 |

Per stat type (same rows):

| stat_type | n | `e`-only | `e × z` | `z`-only |
|---|---:|---:|---:|---:|
| Hits Runs Rbis | 334 | **0.5596** | 0.5541 | 0.5406 |
| Home Runs | 382 | 0.6267 | **0.6304** | 0.6142 |
| Hits | 332 | **0.5905** | 0.5898 | 0.5845 |
| Runs | 185 | 0.5628 | **0.5661** | 0.5248 |
| Rbis | 205 | **0.6038** | 0.5678 | 0.5139 |
| Walks | 280 | **0.5224** | 0.5119 | 0.4884 |
| Fantasy Points | 273 | **0.5797** | 0.5710 | 0.5553 |
| Stolen Bases | 201 | **0.6381** | 0.6170 | 0.5356 |

`e`-only wins pooled and on 6 of 8 stat types; the two it loses (Home Runs +0.0037, Runs
+0.0033) are inside noise.

Conditional test — does `z` carry anything once `e` is fixed? (over-rate gap between the
high-`z` and low-`z` half of each `e`-quintile, and symmetrically):

| quintile | `z`'s residual gap inside an `e`-quintile | `e`'s residual gap inside a `z`-quintile |
|---:|---:|---:|
| 1 | +9.7 pp | +8.8 pp |
| 2 | +1.5 pp | +13.0 pp |
| 3 | +1.8 pp | +6.6 pp |
| 4 | **−2.1 pp** | +7.9 pp |
| 5 | +2.7 pp | +8.2 pp |

`e` discriminates strongly at every level of `z`; `z` barely discriminates at all once `e` is
held fixed, and flips sign in one quintile.

This is the *expected* result, not an anomaly: `p = P(X > L)` is the **sufficient statistic**
for a binary over/under outcome, and `z` is a lossier summary of the same simulated
distribution. `z`'s stated purpose in the spec was portability across stats/sports — but a
probability is already dimensionless and already portable, so `z` was buying normalization
that `e` did not need while adding noise.

**Caveat that must travel with this result:** in the ledger, `m_sd` had to be proxied
(see §5) and `m_med` had to be proxied by the **mean**, so the `z` tested here is a degraded
version of the `z` production computes from real Monte Carlo samples. The decision rule the
task specified is unconditional and every cut points the same way, so it was followed — but
`z` should be re-ablated with real per-row moments once `model_raw_median`/`model_raw_sd` (the
new ledger columns, §6) have accumulated graded history.

---

## 5. Validation — decile monotonicity

Method: the **shipped** `valuation.juice_v2()` was run over all 3,312 graded rows.
2,677 scored; 635 (19.2%) returned `coherence_fault`. Wilson 95% intervals throughout.

Ledger gaps and how each was handled — all four are validation-data limitations, **not**
approximations in the production code path, which computes all of them for real from the
engines' Monte Carlo sample arrays:

| input | gap | handling |
|---|---|---|
| `m_sd` | never stored | proxy = **per-stat residual SD `sd(actual − close_proj)`**, measured on these same graded rows. Real, measured, per-stat — but pooled, not per-row. |
| `m_med` | ledger stores one central value (`close_proj`, the **mean**) | `m_med := m_mean`, which **forces `g = 0`**. Consequence: the skew gap and the mean/median distinction **could not be back-tested at all**, and the 19.2% fault rate below is a **`g = 0` upper bound**, not the expected production rate. |
| `c` | `model_n`/`lineup_status`/`stat_trust_gamma` are not columns in `prop_clv` | `c = 1.0` on every row. This validates the `e` core, not the confidence multiplier. |
| `t` | `trust_weight` 100% NULL | Phase 1 §1 established MLB anchoring went live 2026-07-21, after this window ends, so every row is genuinely `t = 1`. No rows needed the `t < 0.2` exclusion. |

### A. Signed juice deciles — hit = the OVER wins, break-even 50.0% (MLB, n=2,677)

| decile | n | juice range | realized hit % | 95% CI |
|---:|---:|---|---:|---|
| 1 | 267 | −96.9 … −92.6 | 6.7% | [4.3, 10.4] |
| 2 | 268 | −92.5 … −86.5 | 14.6% | [10.8, 19.3] |
| 3 | 268 | −86.5 … −77.8 | 22.0% | [17.5, 27.4] |
| 4 | 267 | −77.6 … −66.1 | 31.8% | [26.5, 37.6] |
| 5 | 268 | −66.1 … −54.8 | 33.2% | [27.8, 39.0] |
| 6 | 268 | −54.8 … −37.9 | 37.7% | [32.1, 43.6] |
| 7 | 267 | −37.9 … −4.8 | 50.2% | [44.2, 56.1] |
| 8 | 268 | −4.8 … +13.4 | 47.8% | [41.9, 53.7] |
| 9 | 268 | +13.7 … +38.6 | 53.7% | [47.8, 59.6] |
| 10 | 268 | +38.9 … +88.3 | 56.7% | [50.7, 62.5] |

**Spearman(decile, hit%) = +0.988; 8 of 9 steps non-decreasing** (the single inversion is
decile 7 → 8, 50.2% → 47.8%, well inside overlapping CIs).

* **Top decile: OVER hit 56.7%** (n=268), CI [50.7, 62.5] — clears 50.0% break-even, but the
  lower bound sits only 0.7 pp above it.
* **Bottom decile: UNDER hit 93.3%** (n=267), CI [89.6, 95.7].
* Note for anyone reading the top-decile number as profitability: 50.0% is the *spec's* `b`
  for a pick'em leg. A real Underdog/PrizePicks 2-pick entry at 3× needs ≈57.7% per leg, so
  56.7% is **not** a demonstrated profitable edge.

### B. |juice| deciles — hit = the side the score points at wins (MLB, n=2,638)

| decile | n | \|juice\| range | realized hit % | 95% CI |
|---:|---:|---|---:|---|
| 1 | 263 | 0.1 … 11.7 | 52.9% | [46.8, 58.8] |
| 2 | 264 | 11.7 … 26.9 | 47.3% | [41.4, 53.4] |
| 3 | 264 | 26.9 … 39.6 | 57.2% | [51.2, 63.0] |
| 4 | 264 | 39.6 … 50.3 | 64.4% | [58.4, 69.9] |
| 5 | 264 | 50.3 … 58.1 | 59.8% | [53.8, 65.6] |
| 6 | 263 | 58.1 … 68.4 | 64.3% | [58.3, 69.8] |
| 7 | 264 | 68.6 … 78.1 | 67.8% | [61.9, 73.1] |
| 8 | 264 | 78.1 … 86.7 | 78.4% | [73.1, 82.9] |
| 9 | 264 | 86.7 … 92.6 | 84.5% | [79.6, 88.3] |
| 10 | 264 | 92.6 … 96.9 | 93.2% | [89.5, 95.6] |

**Spearman = +0.952; 7 of 9 steps non-decreasing.**

### C. Per-stat-type monotonicity (quintiles — n per stat is 185–382)

Deciles would put 19–38 rows in a bucket; quintiles are the honest granularity here.

| stat_type | n | Spearman(quintile, over-hit%) | quintile hit rates |
|---|---:|---:|---|
| Rbis | 205 | +1.000 | 24% → 27% → 29% → 34% → 46% |
| Runs | 185 | +0.949 | 38% → 38% → 43% → 49% → 49% |
| Hits | 332 | +0.900 | 36% → 50% → 54% → 65% → 55% |
| Stolen Bases | 201 | +0.872 | 2% → 8% → 8% → 12% → 12% |
| Home Runs | 382 | +0.600 | 5% → 16% → 13% → 12% → 23% |
| Fantasy Points | 273 | +0.500 | 30% → 40% → 61% → 38% → 51% |
| Hits Runs Rbis | 334 | +0.400 | 35% → 58% → 46% → 54% → 55% |
| Walks | 280 | +0.300 | 21% → 38% → 27% → 30% → 29% |

All eight are **positive**, none is **strictly** monotone, and Walks (+0.300) / Hits Runs Rbis
(+0.400) are weak. Stat types below n=100 after fault exclusion (Total Bases 85, Singles,
Batter Strikeouts, Doubles, and all four pitcher stats) are **not reported** rather than
reported with a meaningless CI. `Doubles` is separately descoped by Phase 1 (degenerate
`m_sd = 0`), and the shipped code nulls it out for exactly that reason.

The pooled numbers in A/B are partly carried by cross-stat base-rate differences (HR unders
hit ~86% as a base rate), which is precisely why this per-stat table is the load-bearing one.

### D. Per sport

| sport | graded rows available | decile monotonicity |
|---|---:|---|
| MLB | 2,677 scored (of 3,312 graded) | measured — §5A/B/C, **descriptive and underpowered** (13-day window, Phase 1 R5) |
| **WNBA** | **0** | **UNMEASURABLE** — zero graded rows in any accessible ledger |
| **Tennis** | **0** | **UNMEASURABLE** — zero graded rows in any accessible ledger |
| **NFL** | **0** | **UNMEASURABLE** — zero graded rows in any accessible ledger |

No monotonicity table was fabricated for the three UNMEASURABLE sports. The score is
implemented and unit-tested for all four; it is validated for one.

### E. Backtest against `backtest.py`

Not run. `backtest.py` reads the production ledger, which is not in this container, and the
seed table lacks `model_version`, `model_n`, `model_floor`/`model_ceiling` entirely. The three
new ledger columns added in §6 are what make a real juice backtest possible later; today there
is nothing to run one against.

---

## 6. Plumbing (all additive)

Threading real pre-anchor moments onto every line, since none of them existed before:

* `projector_bridge.py` — `_payload_from_samples()` now reads `raw_median`, `raw_sd`,
  `raw_prob_over` off the sample array **before** the anchor shift-and-clip (which is why they
  cannot be recovered afterwards); `_payload()`'s no-samples branch reports the same fields,
  with `raw_sd` approximated from the p10–p90 band. **No sibling `../stat-projector` repo
  exists in this container** (checked), and `projector_bridge.py` is JUICED's own glue layer,
  not one of the three vendored engine files CLAUDE.md calls out — no sync needed.
* `analytics.py` — stamps `model_pre_mean/median/sd/prob` + `model_anchor_t` on MLB engine
  **and** empirical-fallback lines; `attach_stat_trust` multiplies γ into `model_anchor_t`
  (idempotently, via a stashed `model_engine_anchor_t`, matching the existing `model_raw`
  guard); new `attach_lock_clock()`.
* `basketball/board.py`, `nfl/board.py`, `tennis/board.py` — same five fields off each board's
  own untouched `arr`, with `model_anchor_t = 0.0` on the snap-to-market branch.
* `db.py` — three new columns via the existing `PRAGMA table_info` / `ALTER TABLE` guard:
  `model_raw_median`, `model_raw_sd`, `model_anchor_t`. **Without these a graded row can never
  be re-scored, so juice can never be backtested** — which is exactly why §5 had to fall back
  to a proxy `m_sd`. Also fixed the stale `db.py:82` comment Phase 1 flagged ("MLB doesn't
  anchor" — untrue since 2026-07-21).
* `dashboard.py` / `optimizer.py` — `_juice_rank()` ranks on **magnitude** with `None` sorting
  last, so every "best juice" / "juice ≥ 80" surface keeps working under a signed, nullable
  score (v1 is non-negative, so `abs()` is the identity there — v1 behaviour is byte-identical).
* `build_static.py` — the five new fields in `_KEEP` and `_PREMIUM_FIELDS` so the static
  deploy does not silently score a different (null) juice than the live server, plus the
  coherence review-queue block.

**One thing deliberately NOT done:** `tennis/board.py` now has the pre-anchor array in hand
and could trivially also stamp `model_raw`/`model_raw_prob`/`trust_weight`. It does not.
Those three feed `db.stat_gammas` / `db.prob_calibration`, which today measure tennis off the
**anchored** `close_proj`/`close_prob`; switching the quantity they read mid-model-version
would blend two definitions of `m` inside one calibration fit. **This is a real finding worth
its own decision** — measuring γ on the anchored value is the self-feedback failure mode
`db.py:353-354` warns about — and is handed to `manager`/`architect` rather than fixed
silently inside a Juice Score commit.

---

## 7. Ship decision: flag-gated, default OFF

`JUICE_VERSION` (env, default `"1"`). `juice_score()` / `juice_factors()` dispatch on it;
`juice_v2()` / `juice_v2_factors()` / `audit_juice_coherence()` are always callable regardless,
so tests, backtests and later phases can use v2 without flipping anything. The full suite is
green under **both** values (472 passed each).

Monotonicity did **not** fail outright for MLB, so the spec's "do not ship as a user-facing
number" trigger did not strictly fire. It ships flag-off anyway, for three concrete reasons:

1. **Three of four sports are UNMEASURABLE.** WNBA/Tennis/NFL have zero graded rows. Making a
   score user-facing on three sports where its monotonicity is entirely unknown is the kind of
   claim this codebase's "no fabricated numbers" rule exists to prevent.
2. **The one measured sport is R5-blocked.** 13 days < 28. Phase 1 recorded that no tuning
   decision may rest on this window; the same logic applies to a shipping decision.
   Per-stat Spearman is positive but weak on two stats (Walks +0.300, HRR +0.400), and the
   top-decile CI lower bound clears break-even by 0.7 pp.
3. **The sign convention is a breaking UI change.** `static/dashboard.html` reads `juiceScore`
   at ~20 call sites that assume unsigned 0–100: `jColor()`'s colour scale, `sort desc`,
   `juiceScore >= 80` ("High Juice"), the portfolio CSV export, the player/team leaderboards.
   The Python side is fully updated; **the frontend pass is an explicit prerequisite for
   flipping the flag** and is not attempted here (half-updating a flag-off path is worse than
   a named blocker).

### Criteria for flipping `JUICE_VERSION` to `2`

1. ≥28 days and ≥400 graded outcomes **per sport**, re-run §5A/B/C per sport, monotone.
2. Re-run the §4 ablation with real per-row `model_raw_median`/`model_raw_sd` from the new
   ledger columns (the `z` tested here was a proxy).
3. Refit `_JUICE_E_SHAPE`/`_JUICE_E_SCALE` per sport on that data.
4. Frontend pass for the signed/nullable convention.

### The old score

`juice_score()`'s v1 body moved to `_juice_v1_score` / `_juice_v1_factors`, unchanged, still
the default. It is a **flag**, not a permanent second live score — it exists to satisfy the
spec's own "ship behind a flag" clause, and `_juice_v1_*` should be deleted at the same commit
that flips the default.

---

## 8. Model version: **NOT bumped** — reasoning

`provenance.MODEL_VERSIONS` is unchanged for all four sports.

**What a bump would protect against:** a historical graded row being attributed to logic that
did not produce it, so a calibration query blends two eras.

**What this change actually did to the engines:** nothing. Every fitted rate, every simulated
distribution, every anchor weight, every `model_proj` / `model_prob` / `model_median` /
`model_floor` / `model_ceiling` is byte-identical before and after. The engine-side diff is
purely **additive read-only fields** describing the same distribution (`raw_median`, `raw_sd`,
`raw_prob_over` and the `model_pre_*` mirrors). Juice Score is a downstream valuation/ranking
number; **no calibration or accuracy query in `db.py` or `backtest.py` reads it**, and none of
them reads any field this change introduced.

**What a bump would cost:** `db.stat_gammas`, `db.prob_calibration`, `db.interval_width` and
`db.stat_biases` are all `model_version`-scoped. Bumping orphans the entire graded calibration
history for all four sports until new data accumulates — and `attach_stat_trust` falls back to
γ = 0.5 for every stat in the meantime, which would materially change live recommendations.

That is the same tradeoff `provenance.MODEL_CHANGELOG` already records for the WNBA
minutes-shrinkage fix (deliberately not bumped: "narrow effect, didn't justify discarding
~6,790 rows of still-valid calibration history"). Here the effect on engine output is not
merely narrow, it is **zero**, so the case is stronger.

**The one place this was close, and how it was resolved:** stamping
`model_raw`/`model_raw_prob` on tennis lines *would* have changed what `db.stat_gammas` reads
for tennis going forward (pre-anchor instead of anchored `close_proj`) — a genuine mid-version
definition change that would have forced a Tennis bump. That was deliberately dropped from
this commit (§6) so the no-bump decision stays clean, and flagged for a separate decision.

**No changelog entry** is added, since `MODEL_CHANGELOG` documents version bumps. If
`JUICE_VERSION=2` is ever made the default, that is still not an engine change and still
should not bump — but it *is* a user-facing product change that belongs in release notes.
