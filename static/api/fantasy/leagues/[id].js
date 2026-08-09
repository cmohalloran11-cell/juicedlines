// GET /api/fantasy/leagues/:id -- public, no auth. Port of routes_fantasy.py's
// league_detail(). `id` here is the SLEEPER league id (passthrough), not our internal
// fantasy_leagues.id -- matches the Python route's own naming exactly.

const { sleeperGet, sendJson } = require("../_lib");

module.exports = async function handler(req, res) {
  const { id } = req.query;
  let league;
  try { league = await sleeperGet(`/league/${encodeURIComponent(id)}`); }
  catch (e) { return sendJson(res, 502, { detail: "Sleeper request failed" }); }
  if (!league) return sendJson(res, 404, { detail: "Sleeper league not found" });
  sendJson(res, 200, league);
};
