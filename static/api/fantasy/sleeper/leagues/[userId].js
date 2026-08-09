// GET /api/fantasy/sleeper/leagues/:userId?season= -- public, no auth. Port of
// routes_fantasy.py's sleeper_leagues(). No 5-min TTL cache here (see _lib.js's
// architecture note -- Vercel functions don't share in-memory state across invocations);
// this is a small, cheap Sleeper call either way.

const { sleeperGet, sendJson } = require("../../_lib");

module.exports = async function handler(req, res) {
  const { userId } = req.query;
  const season = req.query.season || "";
  let leagues;
  try { leagues = await sleeperGet(`/user/${encodeURIComponent(userId)}/leagues/nfl/${encodeURIComponent(season)}`); }
  catch (e) { return sendJson(res, 502, { detail: "Sleeper request failed" }); }
  sendJson(res, 200, { leagues: leagues || [] });
};
