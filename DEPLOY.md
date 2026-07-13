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

## Notes

- **No secrets/keys required.** PrizePicks reads the cookie-free partner API; MLB/ESPN
  need no keys. `config.json` is gitignored and optional.
- The `betting_dashboard` clients (Underdog, Kalshi, MLB) are **vendored** here
  (`underdog.py`, `kalshi.py`, `mlb_model.py`) so the repo is self-contained. Local dev
  still prefers the sibling `../betting_dashboard` if present. Re-copy if you update them.
- The stat-projector engine (`../stat-projector`) is optional — absent, MLB projections
  fall back to the built-in empirical model.
