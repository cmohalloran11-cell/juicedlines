# Phase 3 — Recency weighting, regime breaks, and the two-stage uncertainty layer

**One code change shipped** (a WNBA data-adapter ordering bug, unit-proven, labelled below as
a code-correctness fix and *not* a tuned/measured accuracy improvement). **Three real
statistical mis-specifications found and deliberately NOT shipped**, each with a measured
magnitude showing the fix would be inert or unvalidatable. No `MODEL_VERSIONS` bump.

Phase 1's data constraints are inherited unchanged and were re-verified, not re-derived:
`history.db` in this container has **1 row** (a test artifact — `line_history` is empty), so
the only real graded data remains `clv_seed.db` (3,312 rows, MLB only, 13 days). **No number
in this report comes from `clv_seed.db`** — every figure below is either a synthetic
mechanism measurement (clearly labelled) or a closed-form derivation.

---

## 1. What was audited

How each engine weights recent vs. old observations, and whether anything detects a regime
break (trade, role change, injury return, rotation change).

| Engine | Mechanism | file:line | Regime-break handling |
|---|---|---|---|
| MLB | Three-window **blend**, not decay: `recent = logs[-15:]`, `season = all logs`, `career`. Per-stat recency weight `rw` from `recency_by_stat` | `projector/features/mlb_features.py:44-76`, `projector/config.py:19-32`, `projector/models/base.py:74-78` | **None.** No team/lineup/layoff signal reaches the rate model. |
| WNBA | Exponential decay on **list index**, `w = 0.5**(i/halflife)`, halflife 6 games (rates) / 3 games (minutes) | `basketball/model/rates.py:50`, `basketball/model/minutes.py:30` | **None**, and structurally impossible — see §2.2. |
| Tennis | Exponential decay on **calendar days**, half-life 730 d | `tennis/model/rates.py:21,34-40,95,128` | **None.** A 2-year half-life means an injury layoff or a mechanics change is ~invisible. |

### 1.1 MLB: recent form is switched off for most batter props

`recency_by_stat` (`projector/config.py:29-32`) sets `rw = 0.0` for hits, home_runs,
total_bases, rbis, stolen_bases; `0.2` for batter strikeouts; `0.6` for walks and runs.
Pitchers ignore the table entirely and use the blend default `0.25`
(`mlb_features.py:131` passes `cfg`, not a per-stat weight).

So for **five of the eight batter stats the model projects, the last 15 games carry literally
zero weight**. This is deliberate and documented as fitted on 2023 data ("the data says
single-game counts want the season average, recent form is noise", `config.py:20-21`) — I
have no basis to change it and did not. It is recorded because it is the single biggest
reason MLB cannot react to a regime break: for hits/HR/TB/RBI/SB the projection is a
whole-season average, full stop.

Two further facts that compound it, both verified in code:

* **`career` is never supplied in production.** `projector_bridge.py:193-195` calls both form
  builders without a `career` argument, so `c = career or s` (`mlb_features.py:63,128`)
  always resolves to the season window. The 10–15% "career" weight is really extra season
  weight. `mlb_features.py`'s module docstring ("recent 60% / season 30% / career 10%") is
  stale against both `config.py` and this.
* **Lineup slot never reaches the primary projection.** `form["exp_pa"]` reads
  `predictive["lineup_pa"]` (`mlb_features.py:78`), which `analytics.py` only supplies to
  **A/B variant C** (`analytics.py:1756-1765`), never to variant A. That is by design —
  variant C exists precisely to measure the signal on the ledger before merging it — so this
  is a *pending measurement*, not a bug, and not mine to force.

`build_batter_form`/`build_pitcher_form` sort the log by date themselves
(`mlb_features.py:44,116`), so MLB is immune to the ordering class of bug found in §2.

---

## 2. Shipped: WNBA ESPN adapter returned the game log in feed order

**Bug fix, not a tuned/measured improvement. WNBA is otherwise out of scope (UNMEASURABLE,
Phase 1 R2).** Evidence is a test that fails before and passes after — not a live-accuracy
claim.

### 2.1 The bug

`GameLogSource.gamelog`'s contract is **most-recent-first**
(`basketball/data/base.py:100`), and both consumers weight purely by **list index**:

* `basketball/model/rates.py:50` — `w = 0.5 ** (i / halflife)`
* `basketball/model/minutes.py:30` — `w = 0.5 ** (i / _MINUTES_HALFLIFE)`

Three of the four implementations enforce the contract with an explicit sort
(`balldontlie.py:212`, `wnba_stats.py:277`, and ESPN's own `_boxscore_index` at
`espn.py:275`). `EspnBasketball._athlete_gamelog` did not — it returned games in ESPN's
nested `seasonTypes → categories → events` iteration order, unsorted.

That is not a degraded weighting, it is a potentially **inverted** one: an oldest-first feed
puts weight 1.0 on the oldest game and `0.5**(n/6)` on the newest. The nesting makes it worse
than a single ordering assumption — `seasonTypes` is a list, so regular-season and postseason
blocks are concatenated in feed order regardless of date.

### 2.2 Why this also closes off regime-break detection for WNBA

Worth recording alongside the fix: `basketball/projections.py:127-128` overwrites every
game's `team`/`team_id` with the player's *current* team, and `_athlete_gamelog` never
populates them anyway (`espn.py:319` — `team_id="", team=""`). balldontlie does carry a
per-game `team` (`balldontlie.py:202`) but it is discarded at the same line. **A mid-season
trade is therefore not observable anywhere in the WNBA rate model**, even in principle. That
is a data-plumbing prerequisite for any future regime-break work, not something I could fix
inside this phase's scope.

### 2.3 Fix and evidence

One defensive sort in `basketball/data/espn.py`, matching the three sibling adapters, plus
two regression tests in `basketball/tests/test_espn_gamelog.py`.

Fixture: three games (2026-07-05, -12, -19) served **oldest-first, split across two
`seasonTypes`** — ESPN's real nesting shape. The newest game is the 26-point one.

| measurement | before | after | unweighted reference |
|---|---:|---:|---:|
| returned date order | `07-05, 07-12, 07-19` | `07-19, 07-12, 07-05` | — |
| `fit_rates(...).per_poss["pts"]`, half-life 1 game, no shrinkage | **0.0992** | **0.2302** | 0.1574 |

Before the fix the recency-weighted rate landed **below** the flat average (the model was
anchored on the *oldest* game); after, it correctly sits above it. The same inversion applies
to minutes: `project_minutes([34,33,32,20,18,16], ...)` → `27.0` min, the reversed list →
`23.1` min (`minutes.py`, half-life 3, k=1.5).

Both tests fail on the pre-fix code (verified by stashing the one-line change: 2 failed, 3
passed) and pass after (5 passed).

### 2.4 Impact and model-version reasoning

`basketball/data/__init__.py:10` wires **balldontlie**, not ESPN — ESPN returns HTTP 403 on
every `basketball/wnba` endpoint (2026-08) and is kept in-tree with passing tests but
unwired. So **this bug produces no wrong number in production today**; it is a latent
contract violation that would silently invert WNBA recency weighting the next time the source
is switched — and this repo has switched WNBA sources three times in two months
(`ffb173d`, `f980005`, `7b87e18`).

**`MODEL_VERSIONS["WNBA"]` NOT bumped.** A bump exists so a graded row stays attributable to
the logic that produced it. No graded row's value changes, because the changed code path
produced no graded row: the adapter is not wired in. Bumping would orphan WNBA's entire
graded calibration history (`db.stat_gammas` / `prob_calibration` / `interval_width` /
`stat_biases` are all `model_version`-scoped, and `attach_stat_trust` falls back to γ = 0.5
for every stat meanwhile) in exchange for protecting zero rows. Same reasoning shape as the
`MODEL_CHANGELOG`'s deliberately-not-bumped WNBA minutes-shrinkage precedent, and stronger
here: the effect on any produced number is exactly zero, not merely narrow. No
`MODEL_CHANGELOG` entry, since that file documents bumps.

---

## 3. Found, measured, deliberately NOT shipped

All three are real. All three were rejected on a measured magnitude, not a hunch.

### 3.1 MLB `_eff_pa` ignores the recency blend it is supposed to describe

`mlb_features.py:85` sets `form["_eff_pa"] = season_pa + shrinkage_pa` — the full-season PA
count — and `mlb_model.py:150,162,167,…` uses it as the Gamma shape for every stat's
parameter-uncertainty draw (`_theta`, CV = `1/sqrt(eff_pa)`).

But the point estimate it describes is `rw·recent + (1-rw)·season`, and `recent ⊂ season`. For
iid per-PA outcomes the variance of that blend is closed-form (with `a = rw`, `d = 1-a`,
`n_r` = recent-window PA, `N` = season PA; exact in production because `career` is never
supplied, §1.1):

```
Var(blend) = v · [ a²/n_r + d(2a+d)/N ]        ⇒   n_eff = 1 / (a²/n_r + d(2a+d)/N)
```

Sanity: `a=0 → n_eff = N` ✓ (so the five `rw=0.0` stats are already correct);
`a=1 → n_eff = n_r` ✓.

For a 110-game/484-PA batter with a 15-game/66-PA recent window:

| stats | `rw` | `n_eff` | `_eff_pa` shipped | corrected | CV(θ) shipped → corrected |
|---|---:|---:|---:|---:|---|
| hits, HR, TB, RBI, SB | 0.00 | 484.0 | 544.0 | 544.0 | 0.0429 → 0.0429 (already right) |
| batter strikeouts | 0.20 | 386.2 | 544.0 | 446.2 | 0.0429 → 0.0473 |
| **walks, runs** | 0.60 | **147.6** | 544.0 | **207.6** | **0.0429 → 0.0694** |
| all pitcher stats | 0.25 | 346.7 | 544.0 | 406.7 | 0.0429 → 0.0496 |

So the parameter uncertainty on walks/runs is understated by **62%** — a real error, in the
dangerous direction (too narrow).

**Why it is not shipped — measured effect on the output** (8,000,000 sims per cell, walks,
p=0.092, 4.4 PA/game):

| | CV(θ) | mean | **SD** |
|---|---:|---:|---:|
| shipped (`_eff_pa` 544.0) | 0.0429 | 0.40485 | 0.63631 |
| corrected (`_eff_pa` 207.6) | 0.0694 | 0.40487 | **0.63697** |

**+0.10% on SD, mean unchanged.** At ~4.4 PA/game the outcome variance term `Tp(1-p)`
dominates the parameter term `(Tp)²·CV²` by roughly two orders of magnitude, so a 62% error
in CV(θ) moves the distribution by a tenth of a percent. Shipping it would require new
per-stat `_eff_pa` plumbing through `form` and `mlb_model.py`, and would be a genuine (if
tiny) math change to a live engine — i.e. a `MODEL_VERSIONS["MLB"]` decision that orphans all
four sports' graded calibration history — to buy an effect below Monte Carlo noise. Recorded
here so a human with the real ledger can decide; not taken unilaterally.

### 3.2 MLB and WNBA draw parameter uncertainty with the wrong Gamma shape (tennis does it right)

`_theta` (`mlb_model.py:42-46`) draws `Gamma(eff_pa, 1/eff_pa)` → CV = `1/sqrt(eff_pa)`, and
applies it as a multiplier on a per-PA **probability**. The docstring calls this "the standard
Gamma-Poisson effective-sample-size approximation" — but in a Gamma-Poisson posterior the
shape is the observed **event count**, not the **exposure**. For a rate `p` shrunk by
`regress_to_prior` (which is exactly a Beta-Binomial posterior mean with `n+k` total counts),
the posterior CV is:

```
CV = sqrt( (1-p) / (p·(n+k)) )        not      1/sqrt(n+k)
```

Since every rate in the engine has `p < 0.5`, the shipped CV is **always too small**, by
`sqrt((1-p)/p)`. This is not my invention — it is **already the codebase's own pattern in
tennis**, which draws `Beta(rate·eff_n, (1-rate)·eff_n)` (`provenance.py`, Tennis 1.2.0;
`tennis/model/matchup.py::_beta_draw`, `tennis/sim/engine.py:19-20`). WNBA has the same
mismatch as MLB (`basketball/sim/engine.py:67`, `Gamma(eff_poss, 1/eff_poss)` on a
per-possession count rate, where the correct shape is `rate × eff_poss`).

Measured (400,000 sims per cell; per-game counts corrected to shape = observed count):

| | CV(θ) shipped → correct | SD change, deep sample (`_eff_pa` 544) | SD change, thin call-up (`_eff_pa` 92) |
|---|---|---:|---:|
| home_runs (p=0.038) | 0.0429 → 0.2157 (**5.0×**) | +0.29% | **+2.74%** |
| walks (p=0.092) | 0.0429 → 0.1347 (3.1×) | +0.53% | +1.86% |
| hits (p=0.245) | 0.0429 → 0.0753 (1.8×) | +0.22% | +1.31% |
| total_bases | — | +0.28% | +1.70% |
| H+R+RBI combo | — | +0.32% | +0.93% |
| pitcher strikeouts (p=0.235, 23 BF) | 0.0397 → 0.0716 (1.8×) | +0.93% | — |

Largest `P(over)` shift anywhere in the grid: **0.47 pp** (hits, thin sample).

**Not shipped.** A 5× error in a stated CV is a real defect and is reported as one, but the
realised effect on any user-facing number is ≤2.7% of SD and ≤0.5 pp of `P(over)`. That does
not justify a `MODEL_VERSIONS` bump for MLB *and* WNBA, and — critically — there is no
accessible graded data to confirm the widened intervals actually calibrate better
(`db.interval_width` is `model_version`-scoped and this container has no live ledger). Per
Phase 1 R5 and the standing "no change without a before/after" rule, a mechanism argument
alone is not enough to move a live engine.

### 3.3 The MLB two-stage uncertainty layer is close to inert at realistic sample sizes

This is the headline audit finding and it subsumes §3.1/§3.2. `MODEL_VERSIONS["MLB"] = 1.2.0`
says the layer "widens intervals for thin samples (call-ups, injury returns)". It does — by
under half a percent.

Measured: same form, only `_eff_pa` changed, from a 110-game veteran (544) to an 8-game
call-up (92) — a **5.9× reduction in the evidence behind every rate**. 6 seeds × 2,000,000
sims per cell.

| stat | SD @ `_eff_pa` 544 | SD @ `_eff_pa` 92 | change | seed SE |
|---|---:|---:|---:|---:|
| hits | 1.03957 | 1.04463 | **+0.49%** | 0.02% |
| home_runs | 0.40883 | 0.40912 | +0.07% | 0.03% |
| walks | 0.63656 | 0.63794 | +0.22% | 0.01% |
| runs | 0.82624 | 0.82848 | +0.27% | 0.04% |
| rbis | 0.85595 | 0.85846 | +0.29% | 0.05% |
| total_bases | 2.08884 | 2.09185 | +0.14% | 0.04% |
| H+R+RBI | 1.57981 | 1.58615 | +0.40% | 0.02% |

The existing characterization test only detects the layer at all because it compares
`_eff_pa = 8` against `200_000` — a 25,000× range
(`tests/test_engine_characterization.py:202-203,228-229`). That is a legitimate mechanism
test; it should not be read as evidence the layer moves realistic projections.

This is arithmetic, not a defect: for a single game at ~4.4 PA, `Var = T·p(1-p) + (T·p)²·CV²`
and `T·p ≈ 0.4`, so the parameter term is ~100× smaller than the outcome term. **Even
applying §3.1 and §3.2 together** (correct blend `n_eff` *and* correct Beta CV) the layer's
full realistic range stays under ~3% of SD. Any future work on MLB parameter uncertainty
should start from that ceiling rather than assume the layer is a meaningful lever.

**Nothing shipped from this section.** No engine math changed, no version bumped.

---

## 4. Considered and explicitly not done

| Item | Why not |
|---|---|
| **Removing the MLB stat-gamma / counter-bias layer** (`analytics.py:1213-1226`) | Unchanged, exactly as Phase 1 §4 left it. R3 never fired because the ablation cannot run — no data exists past 2026-07-16, 15 days before the 2026-07-31 deadline split. Removing a live layer on zero evidence is itself an unvalidated change. |
| **Refitting WNBA's fixed dispersion constants** (`min_sd_frac` 0.13, `pace_sd_frac` 0.05, `disp` 0.10, `basketball/config.py:58-60`) | WNBA is UNMEASURABLE (R2, zero graded rows). The per-stat fit the task describes is real and worth doing — `min_sd_frac` in particular could be replaced by each player's own measured minutes SD shrunk toward the league constant, exactly the `_empirical_combo_corr` pattern (`projector_bridge.py:129-153`) — but "measure the improvement" is impossible with zero outcomes, so it would ship as an unvalidated engine change on a sport with no calibration history. Skipped, not estimated. |
| **A regime-break detector (changepoint / trade-aware weighting) in any engine** | Every version needs a threshold or discount fitted against outcomes. There is no accessible data to fit one, and the codebase's rule is explicit that a plausible-sounding constant is a correctness bug. WNBA additionally cannot see a trade at all (§2.2) and MLB's log carries no team either (`projector_bridge.py:88-106` drops it; `analytics.py:422-430` keeps only `opp_id`). **The honest prerequisite is a data-plumbing change (preserve per-game team/lineup-slot through `_to_form_logs` and `PlayerGame`), which belongs to `backend_engineer`, followed by a measurement — not a model change now.** |
| **Building a new walk-forward harness** | `basketball/backtest/calibration.py` already is one (date-strict, `gl[i+1:]` on a recent-first list — verified correct), and `backtest.py` covers the ledger. Adding a third for a hypothetical future dataset is the abstraction-for-a-hypothetical-need the coding standards forbid. |
| **Tennis `_recency_weight` fail-open** (`tennis/model/rates.py:34-40,85-88`) | An unparseable date returns weight **1.0** (the maximum), and `_latest_date` returning `None` silently disables recency weighting entirely. Latent only: the sole `PlayerMatch` producer is Sackmann (`sackmann.py:61`, `tourney_date` = `YYYYMMDD`, always parses); ESPN tennis data feeds Elo/grading, never `fit`. Reported, not changed. |
| **Backtest/production parameter drift in WNBA** (`basketball/backtest/calibration.py:50-56` defaults `shrink_poss=200`, `minutes_shrink_games=3`, `min_sd_frac=0.15`, `disp=0.12` vs `projections.py:135-143` `300`/`4`/`0.15` and `config.py` `120`/`1.5`/`0.13`/`0.10`) | Dead today — WNBA's config sets all four explicitly, so both paths agree. It would silently diverge the moment a second league is added without those keys, which would mean the backtest stops validating production. Reported as a latent hazard, not fixed (no second league exists). |
| **Stale MLB docstring** (`mlb_features.py:1-13` says "recent 60% / season 30% / career 10%"; config says 25/65/10 and `career` is never passed) | Real but cosmetic; deliberately not bundled into a phase whose only code change is a scoped bug fix. |

---

## 5. Model version

**Nothing bumped, for any sport.** `provenance.MODEL_VERSIONS` and `MODEL_CHANGELOG` are
untouched, and that is the correct outcome rather than an omission:

* The one shipped change (§2) is in an **unwired** adapter — `basketball/data/__init__.py:10`
  returns `BallDontLie`. It cannot have produced any graded row, so no historical row is
  mis-attributed and no calibration query can blend eras.
* Everything that *would* have been a math change (§3.1, §3.2) was measured, found to move the
  output by ≤2.7% of SD with no way to validate the direction against real outcomes, and was
  therefore not made.

The cost of a bump, for the record, is the same one `MODEL_CHANGELOG` already weighs:
`db.stat_gammas`, `db.prob_calibration`, `db.interval_width` and `db.stat_biases` are all
`model_version`-scoped, so a bump orphans every sport's graded calibration history and drops
`attach_stat_trust` back to γ = 0.5 until new data accumulates.

## 6. MLB engine sync

No MLB engine file was modified (`projector/models/mlb_model.py`, `montecarlo.py`,
`projector/features/mlb_features.py` are all unchanged — `git diff` touches only
`basketball/data/espn.py` and `basketball/tests/test_espn_gamelog.py`). The sibling
`../stat-projector` does not exist in this container (re-checked), so no sync was required.

## 7. Tests

Fresh venv built from `requirements.txt` + `requirements-dev.txt` (not the ambient
environment — this container's system Python has no numpy at all, so a stale-package false
pass was impossible): **474 passed**, and **474 passed again under `JUICE_VERSION=2`**.
Phase 2 left the suite at 472; the two new tests are §2.3's.
