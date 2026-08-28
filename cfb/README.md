# College Football (FBS) — inside JUICED

**Status: data/plumbing layer only (Phase 4, part A of 2). No projection math exists yet.**
`cfb/board.py::attach_cfb` is a no-op stub — every CFB line reaches the board unprojected
until the modeling agent's engine (garbage-time model, pace model, 3-tier prior fallback)
lands on top of what's here. Registered into `analytics.attach_projections` and
`provenance.MODEL_VERSIONS` (`cfb-0.1.0`) now, not later, so that work is a body-only change.

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

## Extension points for the modeling agent

- **Garbage-time / blowout-probability layer** — `data/cfbd_client.py::schedule()` exposes
  the market spread/over-under per game (CFBD's `/lines`); `team_efficiency()` exposes
  offense/defense PPA + success rate per team-game for the opponent-adjustment/pace inputs.
- **Plays-per-game × usage-share × efficiency pace model** — `team_efficiency()`'s `plays`
  field plus `player_game_stats()`'s box lines are the raw usage-share inputs; no rate-fitting
  exists yet (unlike `basketball/model/rates.py` or `nfl/model/usage.py` — that's this phase's
  actual math work).
- **3-tier prior fallback** — no tiers/priors exist yet. `cfb/board.py`'s docstring
  recommends a `proj_kind` convention (e.g. `cfb_prior_a`/`b`/`c`) mirroring NFL's
  `nfl_regular`/`nfl_preseason` split so a calibration query can distinguish which tier
  actually priced a graded row (`db.stat_gammas`'s `proj_kind` scoping already supports this
  with zero schema change).
- **Opponent-adjusted down-weighting vs FCS/bottom-quartile opponents** — `TeamRef` and
  `ScheduleGame` both carry `classification`/`home_classification`/`away_classification`
  (`"fbs"` vs `"fcs"`), the input an opponent-strength downweight needs.
- **Version bump** — `provenance.MODEL_VERSIONS["CFB"]` is `cfb-0.1.0` (data layer only, no
  math). Bump to `cfb-1.0.0` on the first real engine, per CLAUDE.md's "new engine, nothing
  to orphan" precedent (see NFL's `nfl-1.0.0` changelog entry).

## What's genuinely NOT verified

Every JSON shape parsed in `data/cfbd_client.py` and `data/odds_provider.py` is built against
CFBD's/the-odds-api's own published, documented API contracts — **not confirmed against a
live response**. This sandbox's egress proxy has no route to either host, and no
`CFBD_API_KEY`/`ODDS_API_KEY` was available in this environment either way. Every parse
failure prints a diagnostic (`[cfb.cfbd_client]` / `[cfb.odds_provider]`, flush=True) rather
than silently returning nothing indistinguishable from "no key configured" — the same
verify-on-first-real-deploy contract `basketball/data/balldontlie.py` already ships under.
