# College Football (FBS) — inside JUICED

**Status: real Monte Carlo engine (`cfb-1.0.0`), shipping honestly unvalidated.**
`cfb/board.py::attach_cfb` runs a full simulation per player-game and writes
`model_proj`/`model_prob`/`model_edge`/… onto every CFB line it can price. Nothing about its
accuracy has been measured, and cannot be here — see "What's genuinely NOT verified" at the
bottom before reading any number this engine produces as a validated one.

## The engine

Every projection is **`projected team plays × usage share × efficiency`**, never a per-game
average of the stat. Play-count spread across FBS offenses is the widest in the sport (a
huddle-free tempo team and a ball-control team are both ordinary FBS programs), so the same
usage share is worth materially different counting stats at two different offenses, and only
an explicit decomposition can carry that.

| module | what it fits |
| --- | --- |
| `model/priors.py` | Per-position per-play usage and per-attempt efficiency league means, each with an **empirical-Bayes shrinkage `k` estimated by method of moments from the league's own between-player variance** — not a constant. Also the per-attempt yard spreads the simulator's Gamma draw uses, the P4/G5/FCS level means behind the transfer translation, and the recruiting-rating→usage fit behind the freshman prior. |
| `model/pace.py` | Team plays per game: each team's own tempo EB-shrunk toward the league mean, averaged across the matchup, plus a least-squares response to CFBD's market spread/total applied as a delta. |
| `model/opponent.py` | Opponent quality: each defense's PPA allowed EB-shrunk first, then a fitted log-yards response to it plus a **measured FCS coefficient**. A game against a weak opponent has its yards normalised *and* contributes proportionally less evidence. |
| `model/garbage_time.py` | The blowout layer: measured spread of real final margins around the market's expected margin → the probability the game ends by 21+, then a fitted slope of the starter group's share of team opportunities on that probability. Symmetric in the sign of the spread by construction, so a 35-point favourite and a 35-point underdog get the identical usage discount. |
| `model/rates.py` | The three-tier prior, as a real shrinkage chain rather than a lookup (below). |
| `sim/engine.py` | The Monte Carlo run: per-trial team plays → garbage-time multiplier → **two-stage parameter draw** → outcome draw. |
| `projections.py` | The only I/O in the engine: fetches, caches, and hands the pure fitting layer its dataclasses. |

### The three prior tiers — blended, not selected

Roster churn is the other thing that makes CFB unlike a pro league: a large share of any
board's players either changed schools or have never taken a college snap. The tier chooses
what goes into the **prior** of a two-stage `(observed·n + prior·k)/(n+k)` chain, and the
player's current-season production is blended into it:

| `proj_kind` | tier | prior |
| --- | --- | --- |
| `cfb_prior_a` | returning production | own prior-season rates, opponent-adjusted |
| `cfb_prior_b` | transfer | same, **multiplied by the measured ratio of the two competition levels' own league means** for that exact rate — a translation, not a chosen penalty |
| `cfb_prior_c` | true freshman / no college production | the recruiting rating's implied usage, at the *same* prior strength `k` as the flat positional mean (better centred, not more confident) |

How much a prior season counts as evidence is the measured **year-over-year correlation** of
that rate, fitted on two *completed* seasons (measuring it on the partial current season would
read as "last year tells us nothing" precisely when last year is all there is). Where it isn't
measurable it is `0.0` and the prior collapses to the positional mean — wider, more
market-anchored, honest.

### Two-stage uncertainty

Each trial draws its own rate multiplier — `Gamma(eff_n, 1/eff_n)` for a count rate,
`Beta(rate·eff_n, (1−rate)·eff_n)` for the bounded one — *before* the outcome draw, scaled by
`eff_n`, the real shrinkage denominator behind that rate. That is what makes a true freshman
priced off a recruiting rating come out **wider** than a returning starter with the identical
point estimate, and it is the same construction MLB, WNBA, tennis and NFL all use.

### Market anchoring

On the **median**, never the mean — `cfb/board.py`'s docstring explains why. CFB yardage is
Gamma-shaped and right-skewed, so blending the mean would reproduce exactly the bug that had
94% of the live NFL board recommending Under (`provenance.MODEL_CHANGELOG`,
`nfl-1.2.0`). Anytime TD is anchored on the probability instead, because a 0/1 array has no
meaningful median. Trust is computed **per market**, over only the rates that market's
simulation consumes.

### No constant in this engine is a fabricated number

`cfb/model/config.py` contains simulation size, minimum-sample gates, two definitions and one
factual conference list — and *nothing else*, deliberately. Every prior, shrinkage strength,
per-attempt spread, opponent factor, pace coefficient and garbage-time slope is fitted at
runtime from real CFBD rows. Below its gate, each fit reports an explicit `unmeasured` basis
and the engine applies **no adjustment** rather than a plausible-looking one.

## What this package provides

1. **`data/cfbd_client.py`** — a REST adapter for CollegeFootballData
   (`api.collegefootballdata.com`, Patreon Tier 3). Teams (all 134 FBS programs), rosters,
   schedule with the market's own spread/over-under (CFBD's `/lines`, aggregated real
   sportsbooks), per-player box scores, and per-team advanced efficiency (PPA, success
   rate) — the modeling agent's opponent-adjustment and pace inputs. Requires
   `CFBD_API_KEY`; unset, every method returns an honestly-empty result, never a crash.
2. **`data/odds_provider.py`** — a swappable `OddsProvider` interface (same spirit as
   `basketball/data/base.py`'s `GameLogSource`) plus `TheOddsApiAdapter`, the only source of
   CFB **player props** — CFBD does not carry them. Markets: `player_pass_yds`,
   `player_rush_yds`, `player_reception_yds`, `player_receptions`, `player_anytime_td`.
   Requires `ODDS_API_KEY`; unset, returns `[]`.
3. **Canonical player table + id mapping** (`schema.py`, `repositories.py`,
   `player_matching.py`) — `cfb_players` is the single source of truth every other CFB table
   keys off, `cfb_player_ids` maps a source's own id (CFBD athlete id, or an Odds-API player
   display name — that source has no numeric id) to it, and unresolved/low-confidence
   fuzzy matches land in `cfb_unmatched_players` for human review instead of being silently
   mis-mapped. This is the exact same problem `fantasy/`'s Sleeper-id mapping already
   solved — `player_matching.py` imports `fantasy.player_matching`'s pure
   `normalize_name`/`find_best_match` functions directly rather than reimplementing them.
4. **The ledger** — CFB rows use the *same* `prop_clv` table every other sport does
   (`sport='CFB'`), and the schema already carries every pre-anchor field
   (`model_raw`, `model_raw_prob`, `trust_weight`, `model_version`, `model_n`, …) generically
   for any sport — no CFB-specific migration was needed. `cfb/tests/test_ledger.py` proves a
   full-schema CFB row round-trips through `db.log_clv`/`db.init_db` on a fresh temp DB.
5. **`player_status.py`** — a manual, admin-editable status override (`cfb_player_status`,
   `PUT /api/cfb/player-status/{id}`, ADMIN-gated) with a timestamp, since no CFB injury
   report is mandated to exist. `is_stale` flags a projection when the override is missing or
   was confirmed too long before kickoff (`STALE_AFTER_HOURS`).
6. **`lines.py`** — turns `TheOddsApiAdapter`'s player props into board Line-dicts and
   registers as an entry in `books.REGISTRY` (`fetch_cfb`), so both deploy paths pick it up
   through the existing `books.fetch_extra_books()` call with zero further wiring.
7. **`routes_cfb.py`** (`/api/cfb/*`) — teams/players/status are public reads (transparency
   over already-derived data); writing a status override or resolving a fuzzy-match review
   row requires `ADMIN`.
8. **`model/` + `sim/engine.py` + `projections.py` + `board.py`** — the engine described
   above. `projections.py` is the only module in it that does I/O; everything under `model/`
   and `sim/` is a pure function of already-fetched dataclasses, which is what makes each fit
   testable with no network and no database.

## Gating: only what a book actually posted

We sync all 134 FBS teams' rosters internally (`players_sync.py`) independently of odds —
that's the *internal* player universe the modeling agent projects against. `lines.py` never
walks that roster to synthesize a prop; it only ever transforms what
`TheOddsApiAdapter.player_props()` actually returned for an event. Books price props on a
fraction of the slate — a player/market nobody posted never becomes a line.

## License constraints (CFBD, Patreon Tier 3) — enforced in code, non-negotiable

- `CFBD_API_KEY` is **server-side only**. It is read from `os.environ` in
  `cfb/data/cfbd_client.py` and nowhere else touches it — never in a `static/dashboard.html`
  response, never in `build_static.py`'s output JSON.
- **No route returns a raw CFBD response.** Every `routes_cfb.py` endpoint returns our own
  canonical DB rows (`cfb_teams`/`cfb_players`/`cfb_player_status`), a transform, not a
  passthrough. No bulk raw export exists or should be added.
- `"Data provided by CollegeFootballData.com"` must appear in the CFB footer — **frontend
  concern, not yet implemented; flagged here so it isn't lost** before this sport ships
  user-visible props.

## Testing

`python -m pytest cfb -q`. Everything runs offline against `cfb/tests/fakes.py`, a synthetic
40-team FBS league with four *planted* effects (an FCS production inflation, a defensive-PPA
sensitivity, a blowout starter-usage cut, and a per-team tempo spread) that the fits are
asserted to recover. `cfb/tests/test_model.py` also checks closed-form expectations computed
by hand — the empirical-Bayes `k`, each tier's blended prior, the blowout probability and the
usage multiplier at both spread extremes. `projections.set_source` /
`projections.set_positions_override` are the injection points.

## What's genuinely NOT verified

**Everything about this engine's accuracy.** No `CFBD_API_KEY` has existed in any environment
this code has run in, so not one of its fits has ever seen a live CFBD response, and the
ledger contains zero graded CFB rows, so there is nothing to calibrate against. The tests
prove the *mechanism* — that a planted effect comes back out of the fit, that a hand-computed
prior matches, that a thin sample is wider than a deep one at the same point estimate. They
prove nothing about whether the model is right about college football. `model_health` /
`backtest` correctly report `insufficient_data` for CFB and will until real games grade under
`cfb-1.0.0`.

Every JSON shape parsed in `data/cfbd_client.py` and `data/odds_provider.py` is likewise built
against CFBD's/the-odds-api's own published, documented API contracts — **not confirmed
against a live response**. Every parse failure prints a diagnostic (`[cfb.cfbd_client]` /
`[cfb.odds_provider]`, flush=True) rather than silently returning nothing indistinguishable
from "no key configured" — the same verify-on-first-real-deploy contract
`basketball/data/balldontlie.py` already ships under.

Two known gaps to check on the first real deploy:

- **`player_anytime_td` cannot currently reach the board.** `data/odds_provider.py` requires
  every outcome to carry a numeric `point` and a name of `"Over"`/`"Under"`; the-odds-api
  documents anytime-scorer markets as `"Yes"`/`"No"` outcomes with no `point`, so the parser
  skips them. The engine simulates the market and `cfb/board.py` prices it (anchored on the
  probability, not a median), but no line will exist for it until the adapter is confirmed
  against a real response. Left unfixed deliberately rather than guessed at.
- **The box-score feed carries no targets column** (`/games/players` publishes receptions),
  so receiving is modelled `receptions_per_play × yards_per_reception` rather than
  `targets × catch_rate × yards_per_target` like the NFL engine. That is a data limitation,
  not a modelling preference.
