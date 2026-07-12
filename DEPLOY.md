# Deploying JUICED

Two ways to put it online. Pick based on whether you want it **free** or **fully live**.

---

## Option A — Free public site, no credit card (recommended to start)

Serves a **prebuilt snapshot** as plain static files. A GitHub Action runs the model
every ~20 min and force-pushes `index.html` + `board.json` to the **`deploy`** branch;
a static host serves that. No server, no card.

**What you get:** the full board, edges, and parlay, with model projections. Each
prop's drawer shows the projection card. (The live *recent-games / hit-rate* analytics
and continuous updates are Option B only.)

### Steps
1. The GitHub Action (`.github/workflows/refresh.yml`) is already in the repo. It runs
   on push, every 20 min, and via **Actions → refresh board → Run workflow**. Let it run
   once — it creates the `deploy` branch.
   - If the push step is denied, enable **Settings → Actions → General → Workflow
     permissions → Read and write**.
2. Host the `deploy` branch (both are free, no card, work with a private repo):
   - **Vercel:** Import the repo → Project **Settings → Git → Production Branch =
     `deploy`** → **Framework Preset: Other**, no build command, output = root. Redeploy.
   - **Cloudflare Pages:** Create a project from the repo → **Production branch =
     `deploy`** → no build command → deploy.
3. Open the URL. It auto-updates every ~20 min as the Action republishes `deploy`.

The frontend auto-detects there's no backend and loads `./board.json`, so the **same
`index.html` works locally against the live server and in production as a static site**.

> Freshness is ~20 min (GitHub Actions' floor). Fine for pre-game props; live in-game
> line moves will lag. Want faster/continuous → Option B.

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
