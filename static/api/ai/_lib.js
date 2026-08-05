// Shared helper for the AI Juice Vercel functions (explain.js, coach.js, usage.js) — NOT a
// route itself (leading underscore; Vercel only treats non-underscore files under /api as
// endpoints). Verifies the caller's Supabase session and enforces the same per-user daily AI
// quota the live server enforces (ai_juice.AI_DAILY_LIMITS / store.AiUsageRepository) — against
// the SAME Postgres `ai_usage` table (the project's Supabase Postgres), via PostgREST with the
// service-role key, so a user's quota is identical no matter which deploy path serves them.
//
// Requires two Vercel env vars beyond the existing GEMINI_API_KEY/SUPABASE_URL/
// SUPABASE_ANON_KEY: nothing new for verifying the token (uses the anon key, same as the
// browser), but incrementing usage needs SUPABASE_SERVICE_ROLE_KEY (Project Settings → API →
// service_role secret — server-side only, NEVER sent to the browser) so it can bypass RLS to
// write another user's row. Also requires the `ai_usage` table to exist (store/schema.py's
// CREATE TABLE — runs automatically the next time main.py starts against this Postgres; until
// then, checkQuota() below fails safe to "AI Juice is temporarily unavailable").

const AI_DAILY_LIMITS = { FREE: 10, ELITE: 100 };

function dailyLimit(tier) {
  return AI_DAILY_LIMITS[(tier || "FREE").toUpperCase()] || AI_DAILY_LIMITS.FREE;
}

function todayUtc() {
  return new Date().toISOString().slice(0, 10);
}

function bearer(req) {
  const h = (req.headers && (req.headers.authorization || req.headers.Authorization)) || "";
  const parts = h.split(" ");
  return parts.length === 2 && parts[0].toLowerCase() === "bearer" ? parts[1] : null;
}

// Verifies the token against Supabase directly (no JWKS/JWT verification reimplemented in
// Node) — real identity, not trusted from the client body.
async function verifyUser(req) {
  const token = bearer(req);
  const supabaseUrl = process.env.SUPABASE_URL, anonKey = process.env.SUPABASE_ANON_KEY;
  if (!token || !supabaseUrl || !anonKey) return null;
  try {
    const r = await fetch(`${supabaseUrl}/auth/v1/user`, {
      headers: { Authorization: `Bearer ${token}`, apikey: anonKey },
    });
    if (!r.ok) return null;
    const u = await r.json();
    if (!u || !u.id) return null;
    // Same rule as auth.py's _identity_from_claims: missing plan = the 'lifetime' founding-
    // member default (premium), only an explicit 'free' means FREE.
    const plan = (u.user_metadata && u.user_metadata.plan) || "lifetime";
    return { id: u.id, tier: String(plan).toLowerCase() === "free" ? "FREE" : "ELITE" };
  } catch (_) {
    return null;
  }
}

async function getUsage(userId, date) {
  const supabaseUrl = process.env.SUPABASE_URL, serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  const r = await fetch(
    `${supabaseUrl}/rest/v1/ai_usage?user_id=eq.${encodeURIComponent(userId)}&usage_date=eq.${date}&select=count`,
    { headers: { apikey: serviceKey, Authorization: `Bearer ${serviceKey}` } });
  if (!r.ok) return null;   // table/permissions not ready yet — caller must fail safe
  const rows = await r.json();
  return (rows[0] && rows[0].count) || 0;
}

async function incrementUsage(userId, date, currentCount) {
  const supabaseUrl = process.env.SUPABASE_URL, serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  await fetch(`${supabaseUrl}/rest/v1/ai_usage`, {
    method: "POST",
    headers: {
      apikey: serviceKey, Authorization: `Bearer ${serviceKey}`,
      "Content-Type": "application/json", Prefer: "resolution=merge-duplicates",
    },
    body: JSON.stringify({ user_id: userId, usage_date: date, count: currentCount + 1,
                          updated_at: new Date().toISOString() }),
  });
}

// Returns {ok:true, user, used, limit} if the call may proceed (a call is consumed), or
// {ok:false, response} — a ready-to-return {available:false, reason} payload — when it must
// be refused: not signed in, daily limit reached, or the quota table isn't reachable (fails
// SAFE — refuses the call rather than allowing unmetered access).
async function checkQuota(req) {
  if (!process.env.SUPABASE_SERVICE_ROLE_KEY) {
    return { ok: false, response: { available: false,
      reason: "AI Juice usage tracking isn't configured (SUPABASE_SERVICE_ROLE_KEY missing)." } };
  }
  const user = await verifyUser(req);
  if (!user) {
    return { ok: false, response: { available: false, reason: "Sign in to use AI Juice." } };
  }
  const date = todayUtc();
  const limit = dailyLimit(user.tier);
  const used = await getUsage(user.id, date);
  if (used === null) {
    return { ok: false, response: { available: false,
      reason: "AI Juice usage tracking is temporarily unavailable." } };
  }
  if (used >= limit) {
    return { ok: false, response: { available: false,
      reason: `Daily AI Juice limit reached (${used}/${limit}). Resets at midnight UTC.` } };
  }
  await incrementUsage(user.id, date, used);
  return { ok: true, user, used: used + 1, limit };
}

module.exports = { dailyLimit, todayUtc, verifyUser, getUsage, incrementUsage, checkQuota };
