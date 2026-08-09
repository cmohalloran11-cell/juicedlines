// Shared helper for the Fantasy Draft Assistant's Vercel functions. NOT a route itself
// (leading underscore -- Vercel only treats non-underscore files under /api as endpoints).
//
// Architecture note (2026-08): the live-server FastAPI backend (routes_fantasy.py) is the
// reference implementation and stays the source of truth for anyone running Option B
// (DEPLOY.md). These functions are a SEPARATE port for the Option A (Vercel static) deploy,
// following the exact pattern static/api/ai/_lib.js already established: no raw Postgres
// driver, all data access through Supabase's PostgREST REST API using the service_role key
// (server-side only, never sent to the browser -- same as ai/_lib.js).
//
// The heavy bulk syncs (Sleeper's ~5MB /v1/players/nfl dump, nflverse's ~33MB player_stats
// CSV) do NOT run in these functions -- a Vercel Hobby function has a 10s timeout, nowhere
// near enough. They run as scheduled Python (the EXISTING, already-tested fantasy/players_
// sync.py and fantasy/projections_sync.py, unmodified) via .github/workflows/fantasy-sync.yml,
// writing straight to this same Supabase Postgres. These functions only ever do fast reads
// of already-synced data plus request-time Sleeper calls (small payloads: a user, a league,
// a roster, draft picks -- never the bulk dumps).
//
// Required Vercel env vars beyond SUPABASE_URL/SUPABASE_ANON_KEY (already set for AI Juice):
//   SUPABASE_SERVICE_ROLE_KEY -- already required for ai_usage; reused here.
// Required one-time setup: the fantasy_* tables must exist in this Supabase Postgres --
// run `fantasy.init_schema(store.get_database())` once against its DATABASE_URL (same
// pattern DEPLOY.md documents for ai_usage). See fantasy-sync.yml, which also runs this on
// every scheduled sync (idempotent), so a fresh Supabase project self-heals on the first run.

const SUPABASE_URL = process.env.SUPABASE_URL;
const ANON_KEY = process.env.SUPABASE_ANON_KEY;
const SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

function bearer(req) {
  const h = (req.headers && (req.headers.authorization || req.headers.Authorization)) || "";
  const parts = h.split(" ");
  return parts.length === 2 && parts[0].toLowerCase() === "bearer" ? parts[1] : null;
}

function pgHeaders(extra) {
  return { apikey: SERVICE_KEY, Authorization: `Bearer ${SERVICE_KEY}`,
          "Content-Type": "application/json", ...extra };
}

async function pgSelect(table, qs) {
  const r = await fetch(`${SUPABASE_URL}/rest/v1/${table}?${qs}`, { headers: pgHeaders() });
  if (!r.ok) throw new Error(`pgSelect ${table} failed: ${r.status} ${await r.text()}`);
  return r.json();
}

async function pgInsert(table, rows) {
  const r = await fetch(`${SUPABASE_URL}/rest/v1/${table}`, {
    method: "POST", headers: pgHeaders({ Prefer: "return=representation" }),
    body: JSON.stringify(rows),
  });
  if (!r.ok) throw new Error(`pgInsert ${table} failed: ${r.status} ${await r.text()}`);
  return r.json();
}

async function pgPatch(table, filterQs, patch) {
  const r = await fetch(`${SUPABASE_URL}/rest/v1/${table}?${filterQs}`, {
    method: "PATCH", headers: pgHeaders({ Prefer: "return=representation" }),
    body: JSON.stringify(patch),
  });
  if (!r.ok) throw new Error(`pgPatch ${table} failed: ${r.status} ${await r.text()}`);
  return r.json();
}

// Every PostgREST filter value embedded in a query string MUST go through this -- a bare
// `id=eq.${value}` is a real injection vector (a value containing "&" or "," can append
// extra filter clauses, e.g. league_id from req.query.league_id is fully attacker-controlled).
function eq(value) {
  return `eq.${encodeURIComponent(String(value))}`;
}
function inList(values) {
  return `in.(${values.map(v => encodeURIComponent(String(v))).join(",")})`;
}

function nowIso() {
  return new Date().toISOString().slice(0, 19);   // matches Python's isoformat(timespec="seconds")
}

// Verifies the caller's Supabase session (forwarded to Supabase directly, same as ai/_lib.js
// -- no JWT reimplemented in Node) and upserts/reads the identity from `users`, mirroring
// auth.py's _identity_from_claims exactly: missing user_metadata.plan defaults to the
// 'lifetime' founding-member tier (ELITE), only an explicit 'free' means FREE. Role is never
// set here (defaults to USER on first sight) -- promoting someone to ADMIN is a manual DB
// edit on every deploy path, this one included.
async function verifyUser(req) {
  const token = bearer(req);
  if (!token || !SUPABASE_URL || !ANON_KEY || !SERVICE_KEY) return null;
  let u;
  try {
    const r = await fetch(`${SUPABASE_URL}/auth/v1/user`,
      { headers: { Authorization: `Bearer ${token}`, apikey: ANON_KEY } });
    if (!r.ok) return null;
    u = await r.json();
  } catch (_) { return null; }
  if (!u || !u.id) return null;

  const meta = u.user_metadata || {};
  const plan = meta.plan || "lifetime";
  const tier = String(plan).toLowerCase() === "free" ? "FREE" : "ELITE";
  const now = nowIso();
  try {
    const existing = await pgSelect("users", `id=${eq(u.id)}&select=role,tier`);
    if (existing.length) {
      await pgPatch("users", `id=${eq(u.id)}`, { email: u.email || null, tier, updated_at: now });
      return { id: u.id, email: u.email, role: existing[0].role || "USER", tier };
    }
    await pgInsert("users", [{ id: u.id, email: u.email || null,
                              display_name: meta.full_name || null,
                              role: "USER", tier, created_at: now, updated_at: now }]);
    return { id: u.id, email: u.email, role: "USER", tier };
  } catch (_) {
    // `users` table not migrated yet -- still authenticate the caller (role defaults to USER,
    // which is correct/safe; only the persistence step failed) rather than fail the request.
    return { id: u.id, email: u.email, role: "USER", tier };
  }
}

function sendJson(res, status, body) {
  res.status(status).json(body);
}

const SLEEPER_BASE = "https://api.sleeper.app/v1";
async function sleeperGet(path) {
  const r = await fetch(`${SLEEPER_BASE}${path}`,
    { headers: { "User-Agent": "juicedlines-fantasy-vercel/1.0" } });
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`Sleeper HTTP ${r.status} for ${path}`);
  return r.json();
}

module.exports = {
  SUPABASE_URL, ANON_KEY, SERVICE_KEY,
  bearer, pgSelect, pgInsert, pgPatch, eq, inList, nowIso,
  verifyUser, sendJson, sleeperGet,
};
