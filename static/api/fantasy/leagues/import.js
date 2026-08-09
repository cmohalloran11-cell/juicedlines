// POST /api/fantasy/leagues/import -- auth required. Port of routes_fantasy.py's
// import_league(): reads the league's real scoring settings + roster shape from Sleeper and
// persists it for this user. Find-then-update-or-insert (not a DB-level upsert) so a
// re-import preserves the existing row's id -- mirrors fantasy.repositories.
// LeagueRepository.upsert exactly.

const { verifyUser, sleeperGet, pgSelect, pgInsert, pgPatch, eq, sendJson, nowIso } = require("../_lib");
const crypto = require("node:crypto");

module.exports = async function handler(req, res) {
  if (req.method !== "POST") return sendJson(res, 405, { detail: "Method not allowed" });
  const user = await verifyUser(req);
  if (!user) return sendJson(res, 401, { detail: "Missing bearer token." });

  let body;
  try { body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {}); }
  catch (_) { body = {}; }
  const sleeperLeagueId = body.sleeper_league_id;
  if (!sleeperLeagueId) return sendJson(res, 422, { detail: "sleeper_league_id is required" });

  let league;
  try { league = await sleeperGet(`/league/${encodeURIComponent(sleeperLeagueId)}`); }
  catch (e) { return sendJson(res, 502, { detail: "Sleeper request failed" }); }
  if (!league) return sendJson(res, 404, { detail: "Sleeper league not found" });

  const now = nowIso();
  const fields = {
    name: league.name || null, season: league.season || null,
    league_size: league.total_rosters || null,
    scoring_settings: JSON.stringify(league.scoring_settings || {}),
    roster_positions: JSON.stringify(league.roster_positions || []),
    updated_at: now,
  };
  try {
    const existing = await pgSelect("fantasy_leagues",
      `user_id=${eq(user.id)}&sleeper_league_id=${eq(sleeperLeagueId)}&select=id`);
    let out;
    if (existing.length) {
      const updated = await pgPatch("fantasy_leagues", `id=${eq(existing[0].id)}`, fields);
      out = updated[0];
    } else {
      const created = await pgInsert("fantasy_leagues", [{
        id: crypto.randomBytes(16).toString("hex"), user_id: user.id, sleeper_league_id: sleeperLeagueId,
        ...fields, imported_at: now,
      }]);
      out = created[0];
    }
    sendJson(res, 200, { league: { ...out, scoring_settings: JSON.parse(out.scoring_settings),
                                   roster_positions: JSON.parse(out.roster_positions) } });
  } catch (e) {
    sendJson(res, 500, { detail: "Could not save league" });
  }
};
