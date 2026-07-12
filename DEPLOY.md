# Deploying JUICED

## Why not Vercel / Netlify (serverless)

JUICED is a **persistent server**, not a serverless app:

- a background loop (`_snapshot_loop`) refreshes the board every ~90 s and keeps an
  **in-memory** cache warm,
- the first full data pull takes ~30–40 s (far over Vercel's function timeout),
- it writes a local **SQLite** ledger (`history.db`).

Serverless functions are invoked per-request and frozen in between, with an ephemeral
filesystem — so the background loop, the cache, and the ledger can't work. A Vercel
deploy of this repo just 404s because there's no serverless entrypoint (and there
shouldn't be). **Use a host that runs a long-lived process.**

## Deploy on Render (easiest, free tier)

1. Push this repo to GitHub (already done: `cmohalloran11-cell/juicedlines`).
2. Render → **New → Blueprint** → pick the repo. `render.yaml` sets everything up
   (build `pip install -r requirements.txt`, start `uvicorn main:app --host 0.0.0.0
   --port $PORT`).
3. Open the service URL.

> Render's **free** web service sleeps after 15 min idle; on wake it cold-starts and
> the board is briefly empty until the first refresh finishes. A paid instance ($7/mo)
> stays always-on so the refresh loop never stops.

## Railway / Fly.io / any container host

There's a `Dockerfile`. Point Railway/Fly at the repo (or `fly launch`) — it runs
`uvicorn main:app` on `$PORT`. No build config needed.

## Notes

- **No secrets required.** PrizePicks now reads the cookie-free partner API, and the
  MLB/ESPN feeds need no keys. `config.json` is gitignored and optional (the app runs
  fine without it; per-source failures degrade gracefully).
- The `betting_dashboard` clients (Underdog, Kalshi, MLB) are **vendored** into this
  repo (`underdog.py`, `kalshi.py`, `mlb_model.py`) so it's self-contained. Local dev
  still prefers the sibling `../betting_dashboard` if present; the deploy uses the
  vendored copies. Re-copy them if you update the originals.
- The advanced stat-projector engine (`../stat-projector`) is optional — when absent,
  MLB projections fall back to the built-in empirical model.
