# Phase 1 — Diagnosis

**No model changes made in this phase.** Report only.

## 0. Data-access disclosure (read this before anything else below)

This session runs in an ephemeral container cloned fresh from git. The **real,
live graded ledger is `history.db`** (`db.py:17`), which is gitignored and lives
only on the production host — it does not exist in this checkout and was not
reachable from here. The only real (non-fabricated) graded-outcome data
available in this environment is the checked-in demo/build seed,
**`clv_seed.db`** (used by `build_static.py` for the static-deploy snapshot,
`analytics.py` search hit only there — it is not the table `db.py`/`backtest.py`/
`model_health.py` query in production).

Everything numeric below is computed directly from `clv_seed.db` with no
fabrication, but it is a **small, single-sport, 13-day slice**, not the live
production ledger. Where the task asks for a number I cannot produce honestly
(no data, or a data field that is genuinely unpopulated), I say so explicitly
rather than approximate or invent — per this repo's own overriding rule.

## 1. Ledger inventory

`clv_seed.db.prop_clv`: 3,446 rows total, 3,312 graded (`actual IS NOT NULL`).
`game_date` spans **2026-06-29 → 2026-07-12** (13 days). `graded_at` spans
through 2026-07-16.

| Sport | Total rows | Graded | Graded in last 28d (from today, 2026-08-28) |
|---|---:|---:|---:|
| MLB | 3,446 | 3,312 | **0** |
| WNBA | 0 | 0 | 0 |
| Tennis | 0 | 0 | 0 |
| NFL | 0 | 0 | 0 |

The "0 in last 28 days" row is not a rounding artifact — this environment's
only accessible real data is ~7 weeks stale relative to today, and no
WNBA/Tennis/NFL rows exist at all in the accessible seed. Separately (see §0),
`clv_seed.db`'s own most-recent-28-days-of-*its-own*-data window is
2026-06-19→2026-07-12, but the *entire dataset* only spans 13 days, so a
"recent 28d vs. full history" split (required by Phase 1 item 3, R5) is not
meaningful — there is no ≥28-day full-history window to split against. This
alone means **R5 blocks tuning against this window** regardless of its row
count (3,312 ≫ 400, but the window is 13 days < 28). Recorded as a decision:
Phase 3 must not treat any pattern in this window as a tuning signal.

**Pre-anchor `m` / anchor weight `t` — is it its own field?** Schema-wise yes:
`db.py:90-92` added `model_raw`, `model_raw_prob`, `trust_weight` specifically
so the pre-anchor model survives being shrunk toward the market line
(`db.py:82-86`, `analytics.py:1213-1226`). **In the actual data**, all three
columns are **100% NULL across all 3,312 graded rows** — not partially, not
mostly, entirely unpopulated.

This is not evidence of a broken write path. Tracing the write path
(`analytics.py:1668-1674`, gated on `_tw_use`, itself derived from
`_mlb_trust()` at `analytics.py:1459-1465`) shows `model_raw`/`trust_weight`
are only stamped onto a line when an anchor was *actually applied*.
`_mlb_trust()`'s own docstring says the per-stat trust map was "**validated
out-of-sample 2026-07-21**" — i.e. MLB per-stat anchoring went live *after*
this seed's last graded date (2026-07-16). So for every row in this dataset,
no anchor was applied, `close_proj` legitimately **is** the raw pre-anchor
model output, and the NULL columns are the expected (not buggy) result for
this specific historical window.

The caveat that matters going forward: as of **now** (this repo's `HEAD`),
MLB *does* anchor per-stat (`analytics.py:1213-1226`, `_mlb_trust`), same as
WNBA (`basketball.board`'s trust/sample_weight blend) and Tennis (fully
market-deferred below `trust<0.2`, `tennis/board.py:159-164`). `db.py:82`'s
inline comment — *"MLB doesn't anchor (close_proj IS the raw model)"* — is
now **stale**; it was true for the window this session can see, is no longer
true for the live code. I did not fix the comment in Phase 1 (diagnosis-only
phase); flagged for a doc pass. The substantive risk this stale comment
signals: **if production's real `history.db` has the same 100%-NULL pattern
on `model_raw`/`trust_weight` for rows graded after 2026-07-21, any edge
regression run against that live ledger is silently mixing anchored and
raw values** — exactly the self-feedback failure mode `db.py:353-354`'s own
comment warns about. I could not check this from here; it is a **for-a-human
follow-up**, not a finding I could confirm or rule out with available data.

**Does each sport anchor at all?**

| Sport | Anchors? | Evidence |
|---|---|---|
| MLB | Yes, per-stat, since ~2026-07-21/08-05 | `analytics.py:1213-1226`, `_mlb_trust()` at `analytics.py:1459-1465` |
| WNBA | Yes | `basketball.board`'s trust/sample_weight blend, cited at `analytics.py:1221-1222` |
| Tennis | Yes, fully below `trust<0.2` | `tennis/board.py:159-164` |
| NFL | Not traced (0 ledger rows; out of scope for this pass) | — |

## 2. sd(m − line_close) per stat type (MLB — the only sport with data)

`m = close_proj` (valid proxy for pre-anchor model output in this window —
see §1). `m_sd` = cross-sectional SD of `m` itself.

| stat_type | n | sd(m−L) | m_sd | ratio = sd(m−L)/m_sd | R1 (NO-SIGNAL, ratio<0.15)? |
|---|---:|---:|---:|---:|---|
| Hits Runs Rbis | 385 | 0.4865 | 0.4671 | 1.041 | No |
| Home Runs | 382 | 0.0922 | 0.0922 | 1.000 | No |
| Hits | 375 | 0.3829 | 0.2260 | 1.694 | No |
| Runs | 361 | 0.2407 | 0.2407 | 1.000 | No |
| Rbis | 360 | 0.2450 | 0.2450 | 1.000 | No |
| Walks | 325 | 0.1976 | 0.1976 | 1.000 | No |
| Fantasy Points | 273 | 3.3365 | 7.2262 | 0.462 | No |
| Total Bases | 225 | 0.4166 | 0.4129 | 1.009 | No |
| Stolen Bases | 201 | 0.1025 | 0.1025 | 1.000 | No |
| Singles | 90 | 0.4613 | 0.4613 | 1.000 | No |
| Batter Strikeouts | 90 | 0.4368 | 0.3055 | 1.429 | No |
| **Doubles** | 90 | **0.0000** | **0.0000** | undefined (0/0) | **Yes, in spirit — see below** |
| Strikeouts (P) | 39 | 0.8025 | 1.2992 | 0.618 | No |
| Hits Allowed | 39 | 0.7766 | 0.9568 | 0.812 | No |
| Runs Allowed | 39 | 0.5844 | 0.3593 | 1.627 | No |
| Walks Allowed | 38 | 0.5532 | 0.5964 | 0.928 | No |

No stat cleanly triggers R1's literal formula (all real ratios are well above
0.15 — the model does disagree with the market by a meaningful multiple of
its own cross-sectional spread on every real stat type). **`Doubles` is the
exception and a genuine finding, not a formula edge case**: `close_line` and
`close_proj` are *both* a single constant value (0.5 line, 0.0 projection)
across all 90 graded rows — the model produces a literally identical
"projection" every time regardless of player, i.e. zero real signal, worse
than the ratio formula even implies (0/0, not a small positive number). This
reads like a stub/fallback path for Doubles rather than a fitted rate. **Flagged
as NO-SIGNAL by the spirit of R1** — descope from Phase 3 tuning and treat as
a placeholder needing a real fit, not evidence the model has *tried and
failed* on Doubles.

The many exact `ratio = 1.000` rows are an artifact of fixed half-integer
lines (Runs/RBIs/Walks/Stolen Bases/Singles all price almost exclusively at
0.5), not a coincidence — `L` is ~constant per stat, so `sd(m−L) = sd(m)`
mechanically. Not a bug, just means the ratio metric is uninformative for
fixed-line stats; §3's gamma is the more meaningful signal there.

## 3. Edge regression + encompassing regression

`(y − L) = α + γ(m − L) + ε`, bootstrap 95% CI (2,000 resamples, seeded).
**Only one window exists** (13 days) — per R5, "last 28 days" cannot be
reported separately from "full history" because there is no full-history
window longer than the recent one. Do not read the numbers below as a tuning
signal; treat them as descriptive only.

**MLB pooled across stat types** (n=3,312): α=0.056, γ=**0.306**, SE=0.044,
bootstrap 95% CI **[−0.120, 0.650]** — crosses zero. This pooled number mixes
wildly different raw scales (Fantasy Points has ~40–80x the raw variance of
Home Runs) and is dominated by the highest-variance stat types; it is
reported because the task asked for it, but the per-stat breakdown below is
the trustworthy read.

Encompassing regression `y = a + b_m·m + b_L·L`: `y = 0.223 + 0.185·m +
0.698·L`. `b_m ≠ 0` but is small relative to `b_L` — consistent with a market
that's still doing most of the work and a model that adds a real but modest
increment, which matches the per-stat picture below better than the noisy
pooled γ does.

**Per stat type** (n≥30):

| stat_type | n | γ | bootstrap 95% CI | excludes 0? |
|---|---:|---:|---|---|
| Hits Runs Rbis | 385 | 0.533 | [0.149, 0.937] | Yes |
| Home Runs | 382 | 0.510 | [0.108, 0.986] | Yes |
| Hits | 375 | 0.629 | [0.385, 0.854] | Yes |
| Runs | 361 | −0.041 | [−0.361, 0.283] | No |
| Rbis | 360 | 0.158 | [−0.250, 0.503] | No |
| Walks | 325 | 0.258 | [−0.177, 0.705] | No |
| Fantasy Points | 273 | 0.266 | [−0.355, 0.794] | No |
| Total Bases | 225 | 0.489 | [−0.074, 1.057] | No |
| Stolen Bases | 201 | 0.223 | [−0.210, 0.757] | No |
| Singles | 90 | 0.183 | [−0.150, 0.506] | No |
| Batter Strikeouts | 90 | 0.291 | [−0.137, 0.731] | No |
| Doubles | 90 | — | degenerate (zero variance in m−L) | — |
| Strikeouts (P) | 39 | −0.599 | [−1.521, 0.482] | No |
| Hits Allowed | 39 | 0.124 | [−0.741, 0.731] | No |
| Runs Allowed | 39 | 1.597 | [0.134, 2.991] | Yes |
| Walks Allowed | 38 | 0.533 | [−0.221, 1.373] | No |

Only Hits, Home Runs, Hits+Runs+RBIs, and Runs Allowed have a CI that
excludes zero. **Important**: this is a 13-day, single-window read forbidden
from tuning use by R5 — a wide CI here means "underpowered," not "no edge."
Do not treat the CI-includes-zero stats (Runs, RBIs, Walks, Fantasy Points,
Total Bases, Stolen Bases, Singles, Batter/Pitcher Strikeouts, Hits/Walks
Allowed) as proven NO-SIGNAL; R1's own test (§2) didn't flag any of them
either. The honest read is "not measurable with confidence from what's
available here," which is different from "measurably zero."

## 4. MLB degradation test (trade-deadline split) — **UNMEASURABLE**

The task asks to split the ledger at the 2026-07-31 trade deadline and
compare bias/Brier/γ pre vs. post, plus a held-out ablation of the counter-bias
(stat-gamma shrink) layer on the recent period, plus a break-out by
team/order-slot change.

**Cannot be run.** The only accessible data ends 2026-07-16 — **15 days
before** the trade deadline. There are zero rows on either side of a
2026-07-31 split; the "recent period" the ablation needs to score against
does not exist in this environment.

**R3 does not fire.** R3 only triggers "if the ablation shows removal
improves recent-period performance" — there is no ablation result to trigger
on, positive or negative. I am explicitly **not** removing the stat-gamma
shrink layer (`analytics.py:1213-1226`) in Phase 3 on the basis of an
un-run test. Note the tension this leaves for a human: Phase 3's own standing
rule says an unvalidatable bias-correction layer should be removed, but
removing a live layer on *zero* evidence (rather than on a demonstrated
regression) is itself an unvalidated change — arguably a bigger risk than
leaving a documented, previously-audited layer in place. I'm treating "we
cannot get the evidence in this sandbox" as a reason to preserve status quo,
not a license to change it either direction, and flagging it as an open
item for whoever has `history.db` access (see `00-summary.md`).

One relevant fact **is** independently recoverable from the code, not from
data I ran: the stat-gamma shrink layer's own introduction was itself
motivated by a documented live audit (`analytics.py:1213-1226`) — MLB
"Pitching Outs" measured γ=0.0 (proven zero edge) but was still recommended
Under 91.4% of the time, hitting 33.8%. That's real prior evidence the
un-shrunk system had exactly the failure mode Phase 2/3 are trying to guard
against — cited here as context for why the layer exists, not verified fresh
by me (no "Pitching Outs" rows exist in the accessible seed).

## 5. WNBA pace units mismatch — traced, **does not leak into the level**

Config's `league_pace = 96.0` (`basketball/config.py:38`) vs. ESPN/balldontlie's
directly *computed* per-game WNBA possessions, which land ~80–85
(`basketball/data/espn.py:429`). Traced both definitions:

- `league_pace` is used as an **internal normalization constant**: it's the
  denominator when converting the per-40-minute positional prior into
  per-possession rates (`basketball/model/priors.py:49-51`,
  `_per40_to_poss: per40/league_pace`), and it's the pace fed back into
  `player_possessions(minutes, game_len, pace)` at simulate time
  (`basketball/model/rates.py:37-39`). Used consistently on both sides, its
  absolute value cancels for the LEVEL of a league-average-pace projection —
  it does not need to equal any literally "true" real-world number to keep
  calibration, only to be self-consistent.
- The *computed* ~80–85 is a real per-game number from box scores
  (`_possessions()`, `basketball/data/espn.py:107`), used only as a **relative**
  matchup adjustment: `pace = league_pace × (matchup_computed /
  league_average_computed)` (`basketball/projections.py:150-163`). A
  league-average matchup lands exactly back on `league_pace=96` by
  construction; only the *deviation* from the computed league average moves
  a projection.
- This relative-only design is explicitly a **fix**, not an oversight: the
  same comment block records that feeding the *raw* computed pace in
  directly was tried and measured worse — **"Using the raw pace here cost
  −2.76 of bias"** (`basketball/projections.py:159`,
  `basketball/data/espn.py:429-434`).

**Conclusion: does not leak into the level.** It only ever moves the
matchup-relative delta, by design, with a measured before/after already on
record. **R4 does not fire** — this is a finding that turns out to already be
fixed, reported as a non-issue per this repo's own audit philosophy, not
forced into a change to justify having looked.

One non-blocking recommendation for Phase 3/docs: `league_pace`'s name
invites a future engineer to "fix" it by pointing it at the real computed
80–85 without also refitting `_WNBA_PER40` priors — which *would* leak into
the level and reintroduce the −2.76 bias this design already paid to avoid.
Worth a renaming/comment pass (e.g. `pace_reference_constant`), not a Phase 3
model change.

## 6. Fraction of props with t=0 or t<0.2 — **unmeasurable from available data**

`trust_weight` is 100% NULL across all 3,312 graded rows in the only
accessible ledger (see §1 — this window predates MLB's anchoring going live,
so the field was never populated for these rows; not a logging bug for this
specific window). **I cannot compute this fraction and did not estimate one.**
Per the task's own instruction, props with `t=0` or `t<0.2` carry no model
signal and must be excluded from every performance stat in this run — since
none of the accessible rows carry a `t` value at all, and since I've
independently confirmed via code trace (§1) that no anchoring was live during
this window, **every row in §2/§3 is effectively `t≈1` (full raw model, no
anchor)** by construction, not by having verified individual `t` values. That
satisfies the exclusion rule's *intent* for this dataset specifically, but the
fraction itself is a genuine unknown for any post-2026-07-21 data, and stays
on the unmeasurable list in `00-summary.md`.

Separately, current code does encode a related but distinct threshold:
Tennis fully defers to the market line whenever `trust < 0.2`
(`tennis/board.py:162`) — a design-time choice, not a measured fraction of
live props that land there.

## Decision rules applied

| Rule | Result |
|---|---|
| R1 (sd(m−L) < 0.15·m_sd → NO-SIGNAL) | Did not fire on any real MLB stat type. Fired **in spirit** on `Doubles` (degenerate 0/0, model produces a literal constant) — descoped from Phase 3 tuning. |
| R2 (<400 graded outcomes → UNMEASURABLE) | **WNBA, Tennis, NFL: UNMEASURABLE** (0 graded rows accessible in this environment). MLB has 3,312 graded rows — not UNMEASURABLE by count, but see R5. |
| R3 (ablation shows removal helps → remove counter-bias layer) | **Did not fire** — ablation could not be run (§4). Layer left unchanged. |
| R4 (WNBA mismatch leaks into level → fix priority one) | **Did not fire** — traced and confirmed it does not leak into the level (§5). No fix needed. |
| R5 (never tune on <28 days or <400 outcomes) | **Fires for the entire MLB window** — 13 days < 28, regardless of n=3,312. No tuning decision in this report is based on this window's point estimates; Phase 3 must not either. |

## What Phase 3 inherits from this report

- **MLB**: in scope, but every number above is descriptive/underpowered, not
  a tuning signal (R5). `Doubles` is descoped (stub-like, zero variance).
  Trade-deadline ablation (§4) is not run — no removal decision made on the
  counter-bias layer.
- **WNBA, Tennis, NFL**: UNMEASURABLE by R2 (zero accessible graded rows).
  **Skip all tuning; do not report performance claims for these sports in
  Phase 3.**
