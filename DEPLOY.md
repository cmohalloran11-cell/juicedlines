# Deploying JUICED

Two ways to put it online. Pick based on whether you want it **free** or **fully live**.

---

## Option A — Free public site, ~5-min updates, no credit card

Serves a **prebuilt snapshot** as static files. A GitHub Action (`refresh.yml`) runs the
model every ~5 min and force-pushes `board.json` to the **`data`** branch, which the page
reads **straight from GitHub raw** (updates instantly, no host redeploy). The page itself
(`static/index.html`) is served by Vercel **straight from `main`** and only redeploys
when the code actually changes. No server, no card.

Why this shape: it dodges two free-tier ceilings — GitHub Actions minutes (unlimited on a
**public** repo) and the host's deploy cap (Vercel ~100/day; we avoid it by not
redeploying on every refresh).

**What you get:** the full board, edges, and parlay with model projections; each prop's
drawer shows the projection card. (Live *recent-games / hit-rate* analytics and true
continuous updates are Option B.)

### Steps
1. **Make the repo public** (Settings → General → Danger Zone → Change visibility). This
   is required: it gives unlimited Actions minutes and lets GitHub raw serve `board.json`.
   No secrets are in the repo — `config.json`/`history.db`/caches are gitignored, and
   `config.example.json` is only a placeholder.
2. Enable write for the Action: **Settings → Actions → General → Workflow permissions →
   Read and write**. Then let the workflow run once (**Actions → refresh board → Run
   workflow**) — it creates the `data` branch.
3. Host it on **Vercel** — Production Branch = `main`, **Root Directory = `static`**:
   - **Settings → Build and Deployment → Root Directory → `static`** → Framework Preset
     **Other**, Build Command empty → Redeploy. Vercel serves `main/static/index.html` as
     static files; the page pulls `board.json` live from the `data` branch via raw, so the
     host never redeploys on a data change.
   - ⚠️ Root Directory **must** be `static`. The page lives in `static/`, not the repo
     root, so leaving Root Directory blank serves the root and **404s** (`NOT_FOUND`).
   - **Cloudflare Pages:** Production branch **`main`**, build output directory **`static`**,
     no build command.
4. Open the URL. The board self-updates every ~5 min from the `data` branch via raw — the
   host never has to redeploy for a data change.

### AI Juice on the static deploy (Vercel Project → Settings → Environment Variables)
The `static/api/ai/*.js` serverless functions are separate from the table below (those are
`main.py`/Render env vars) — set on Vercel directly:
- `GEMINI_API_KEY` (+ optional `AI_MODEL`) — same as the live server.
- `SUPABASE_URL` / `SUPABASE_ANON_KEY` — to verify who's calling (AI Juice requires sign-in;
  see the daily-quota note below).
- `SUPABASE_SERVICE_ROLE_KEY` — **new, server-side only, never sent to the browser.**
  Project Settings → API → `service_role` secret. Lets the serverless function read/write
  another user's row in `ai_usage` (bypassing RLS) to enforce the daily AI limit. Missing ⇒
  AI Juice fails safe with "usage tracking isn't configured", not unmetered access.
- **One-time schema migration required**: the `ai_usage` table (10/day free, 100/day Pro —
  `ai_juice.AI_DAILY_LIMITS`) is created by `store/schema.py`'s `init_schema()`, which only
  runs when `main.py` starts against this project's Postgres (its FastAPI `lifespan` calls
  it automatically). If the live server has never been pointed at production `DATABASE_URL`,
  run it once: `DATABASE_URL=<prod connection string> python -c "import store; store.init_schema(store.get_database())"`.
  Until then, AI Juice on the static site fails safe the same way (usage tracking
  unavailable) rather than allowing unmetered calls.

### Fantasy Draft Assistant on the static deploy

`static/api/fantasy/*.js` ports `fantasy/` + `routes_fantasy.py` (the live-server reference
implementation) to Vercel serverless functions, following the exact `static/api/ai/*.js`
pattern: no raw Postgres driver, all data access through Supabase's PostgREST REST API with
the `service_role` key. No new Vercel env vars beyond AI Juice's above — same three
(`SUPABASE_URL`/`SUPABASE_ANON_KEY`/`SUPABASE_SERVICE_ROLE_KEY`) cover both.

The heavy bulk syncs (Sleeper's ~5MB player dump, nflverse's ~33MB projections CSV) do
**not** run in these functions — a Vercel Hobby function's 10s timeout isn't close to enough.
They run instead as scheduled Python on GitHub's unthrottled runners
(`.github/workflows/fantasy-sync.yml`, reusing `fantasy/players_sync.py` and
`fantasy/projections_sync.py` unmodified), writing straight to the project's Supabase
Postgres. The Vercel functions only ever do fast reads of already-synced data plus small
request-time Sleeper calls (a user, a league, a roster — never the bulk dumps).

**One-time setup**: GitHub → repo → Settings → Secrets and variables → Actions → new
repository secret `FANTASY_DATABASE_URL` (the Supabase project's Postgres connection
string — Project Settings → Database → Connection string → URI; prefer the pooled/pgbouncer
string on port 6543, since this job opens a fresh connection per run). No manual schema step
needed beyond that — `fantasy.init_schema()`/`store.init_schema()` run on every sync
(idempotent), so a fresh Supabase project self-heals on the workflow's first run
(`Actions → fantasy sync → Run workflow`, or wait for the daily schedule).

`BOARD_URL` in `index.html` points at this repo's raw `data/board.json` (update it if you
fork/rename). Locally the page just loads `./board.json`, so the same file works in dev
(live server) and in production (static).

> Freshness is ~5 min (GitHub Actions' scheduling floor; can drift to ~10 under load).
> Fine for pre-game props; live in-game moves still lag a little. Want true continuous
> (~90 s) updates → Option B.

---

## Option B — Full live server (continuous updates + live analytics)

This runs the real FastAPI app: a background loop refreshes every ~90 s, the analytics
drawer queries live game logs, and it keeps a SQLite ledger. That needs a host that
runs a **long-lived process** — **not** Vercel/serverless (per-request, frozen between
calls, ephemeral disk, and the ~40 s cold pull exceeds function timeouts).

- **Render:** New → Blueprint → pick the repo (`render.yaml` configures it). Free tier
  sleeps after 15 min idle; **$7/mo** stays always-on.
- **Railway / Fly.io / any container host:** use the `Dockerfile` (`uvicorn main:app`
  on `$PORT`). ~$5/mo for always-on.

---

## Rollback

CI (`.github/workflows/test.yml` + `deploy-prod.yml`'s `test` job) blocks a broken commit
from reaching production, but it can't catch a runtime-only issue that only shows up under
real traffic or real upstream data. If that happens, roll back — don't try to hotfix live.

### Option A (static/Vercel)
The site itself: **Vercel → Deployments → find the last good deployment → "⋯" → Promote to
Production.** Takes effect immediately, no rebuild.
The board data (`data` branch): it's force-pushed every ~5 min by `refresh.yml`, so a bad
*board* (not a bad *site*) self-heals on the next cycle — nothing to roll back manually.
If a bad board is somehow persisting, disable the workflow (**Actions → refresh board →
"..." → Disable workflow**) to stop the churn while you investigate, then re-enable it.

### Option B (full server — Render/Railway/Fly/Docker)
- **Render:** Dashboard → your service → **Events** tab → find the last good deploy →
  **Rollback to this deploy**. One click, Render redeploys that exact commit.
- **Railway:** Project → Deployments → find the last good one → **⋮ → Redeploy**.
- **Fly.io:** `fly releases` to list, then `fly deploy --image <previous-image-ref>` (or
  `fly releases rollback` if available on your CLI version) to return to it.
- **Any Docker host without a built-in rollback UI:** re-deploy the previous known-good
  commit directly — `git checkout <last-good-sha> && docker build ... && docker push ...`
  (or re-trigger your CI/CD pipeline against that SHA) — then redeploy from that image.

### General playbook (any path)
1. **Identify the last good commit** — `git log --oneline` on `main`, cross-referenced
   against when the issue started (check `/health` uptime and the request logs).
2. **Roll back the deploy** using the platform steps above — don't `git revert` and
   re-deploy under pressure; that's slower and risks a second mistake mid-incident.
3. **Once stable, `git revert` the bad commit(s) on `main`** (not `reset --hard` — keep
   history) so the next normal deploy doesn't reintroduce the same bug, then investigate
   with the pressure off.
4. **The database is untouched by a rollback.** Neither deploy path rolls back `history.db`/
   `juiced.db` schema or data — rolling back code only matters if the bad deploy didn't
   also write bad data. If it did, that's a separate, deliberate data-fix, not a rollback.

---

## Environment variables (Juiced 2.0)

All optional — the app runs with none set. Configure on the host (Render/Railway env vars).

| Variable | Default | Purpose |
|---|---|---|
| `HISTORY_RETENTION_DAYS` | `14` | Rolling window for the line-movement snapshot table. `0` disables pruning. Keeps `history.db` bounded (it reached 1.77 GB unbounded). |
| `JUICE_VERSION` | `1` | Which Juice Score the board serves. `1` = the 2026-08-05 unsigned 0–100 composite (current production). `2` = the rebuilt **signed** score in `[-100, +100]` (positive = over, negative = under, `null` = no model opinion / model-integrity fault). **Do not set to `2` in production yet** — `static/dashboard.html` still assumes an unsigned, never-null score in its colour scale, sort order and "Juice ≥ 80" filters, and the score's decile-monotonicity validation exists for MLB only. Set it on **both** deploy paths or the static and live boards will disagree. See `reports/02-juice.md`. |
| `DATABASE_URL` | *(unset → SQLite)* | Point at Postgres (`postgres://…`) to move the users/watchlists/portfolio store off SQLite. Requires `psycopg` (see requirements.txt). |
| `SUPABASE_URL` | *(unset)* | Supabase project URL (e.g. `https://<ref>.supabase.co`). Enables **ES256/JWKS** verification of Supabase JWTs — the current default. Unset + no secret ⇒ per-user endpoints fail closed (401). |
| `SUPABASE_JWT_SECRET` | *(unset)* | Legacy HS256 symmetric secret (only if not using JWKS). |
| `SUPABASE_JWT_AUD` | `authenticated` | JWT audience claim to verify. |
| `JUICED_AUTH_DEV` | *(off)* | Local-only escape hatch: with auth unconfigured, trust an `X-Dev-User` header. **Never set in production.** |
| `AI_PROVIDER` | `gemini` | `gemini` or `anthropic`. |
| `GEMINI_API_KEY` | *(unset)* | Google Gemini key — enables AI Juice (`/api/ai/*`) when `AI_PROVIDER=gemini`. |
| `ANTHROPIC_API_KEY` / `AI_API_KEY` | *(unset)* | Anthropic key — used when `AI_PROVIDER=anthropic`. |
| `AI_MODEL` | `gemini-2.5-flash` | Model for AI Juice (provider-appropriate default). |
| `RATE_LIMIT_RPM` | `300` | Per-IP requests/minute over `/api/*` (sliding window). `0` disables it entirely. |
| `SNAPSHOT_INTERVAL` | `180` | Seconds between background board refreshes. |
| `SENTRY_DSN` | *(unset)* | Enables real-time error monitoring (unhandled exceptions) via Sentry. Unset ⇒ the app never talks to Sentry at all — safe to leave unset locally and on any deployment that doesn't want it. |
| `SENTRY_ENVIRONMENT` | `production` | Tag shown on Sentry issues (e.g. `staging`, `production`) — only matters if `SENTRY_DSN` is set. |
| `SENTRY_TRACES_SAMPLE_RATE` | `0.0` | Performance-tracing sample rate (`0.0`–`1.0`). Off by default; only matters if `SENTRY_DSN` is set. |
| `NFL_CACHE_DIR` | `data/nfl_cache/` | Where the NFL engine caches the fetched nflverse-data release CSVs (tens of MB). Relocate off a read-only or ephemeral filesystem if needed. |
| `BALLDONTLIE_API_KEY` | *(unset)* | WNBA's data source (`basketball/data/balldontlie.py`) — rosters, game logs, pace, upcoming opponents. Sign up free at [balldontlie.io](https://balldontlie.io) for a key. Unset ⇒ WNBA has no real projections (props still post, but with no `model_proj` — the board hides them from the picks list, same as any unprojected line). Two prior free/keyless sources (ESPN, then stats.wnba.com) were tried and both failed in production — see the module's own docstring for the history. |
| `CFBD_API_KEY` | *(unset)* | College Football (`cfb/data/cfbd_client.py`) — teams, rosters, schedule + market spread/total, per-player box scores, per-team advanced efficiency. Requires a [CollegeFootballData.com](https://collegefootballdata.com) Patreon Tier 3 key. **Server-side only** — never exposed to the browser or written into `build_static.py`'s output JSON (license constraint). Unset ⇒ every CFBD-backed CFB endpoint returns an honestly-empty result; the roster sync loop (`cfb/players_sync.py`) idles as a no-op. **On the static deploy (Option A) this must be a GitHub repo secret, not a Vercel env var** — Vercel only serves the prebuilt `board.json`; `refresh.yml` (GitHub Actions) is what actually runs `cfb/data/cfbd_client.py`, same as `BALLDONTLIE_API_KEY` above. Set it at Settings → Secrets and variables → Actions. |
| `ODDS_API_KEY` | *(unset)* | College Football player props (`cfb/data/odds_provider.py`, [the-odds-api.com](https://the-odds-api.com)) — CFBD carries no player props, so this is the only source of CFB prop lines. Unset ⇒ `cfb.lines.fetch_cfb_props` returns `([], None)`, a clean no-op (same as any other optional book in `books.py`). **Same static-deploy caveat as `CFBD_API_KEY` above** — a GitHub Actions secret, not a Vercel env var, on Option A. |

New endpoints: `/api/version` (model/feature versions), `/api/ai/status`, `/api/ai/explain?id=`, and the authenticated `/api/me`, `/api/watchlists*`, `/api/portfolio*`.

## Notes

- **No secrets/keys required to run** the app or the other three sports. PrizePicks reads
  the cookie-free partner API; MLB/ESPN need no keys; the NFL engine reads the free,
  no-auth nflverse-data GitHub releases. Auth and AI features activate only when their env
  vars are set. WNBA is the one exception: it needs `BALLDONTLIE_API_KEY` for real
  projections (see the table above) — without it the sport still loads, just with no
  model-projected props.
- **CFB needs BOTH keys to be useful, and each does a different job.** `ODDS_API_KEY` is the
  only source of CFB prop *lines* (CFBD carries no player props); `CFBD_API_KEY` is the only
  source of the *data the engine fits on* — the engine estimates every prior, shrinkage
  strength, opponent factor and pace coefficient at runtime from real CFBD rows rather than
  from constants, so with no CFBD key there is nothing to fit and `cfb.projections.league_data`
  returns `None`. Neither key set ⇒ CFB is entirely inert (no teams synced, no props pulled).
  `ODDS_API_KEY` only ⇒ real prop lines post with no `model_proj`, the same visible state as
  WNBA with no `BALLDONTLIE_API_KEY`. Both set ⇒ full projections
  (`proj_kind` = `cfb_prior_a`/`b`/`c`, see `cfb/README.md`). Also run the roster sync
  (`cfb/players_sync.py`, wired into `main.py`'s lifespan): the box-score feed carries no
  position, so an unsynced `cfb_players` table leaves every player in the pooled prior bucket.
- **CFB accuracy is not measurable yet, by construction.** It is a brand-new engine with zero
  graded rows, so `model_health`/`backtest` report `insufficient_data` for it — that is
  correct, not a bug, and it will stay that way until real games grade under `cfb-1.0.0`.
- The `betting_dashboard` clients (Underdog, Kalshi, MLB) are **vendored** here
  (`underdog.py`, `kalshi.py`, `mlb_model.py`) so the repo is self-contained. Local dev
  still prefers the sibling `../betting_dashboard` if present. Re-copy if you update them.
- The stat-projector engine (`../stat-projector`) is optional — absent, MLB projections
  fall back to the built-in empirical model.
