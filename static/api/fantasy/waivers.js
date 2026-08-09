// GET /api/fantasy/waivers?league_id=&roster_id=&top_n= -- auth required. Port of
// routes_fantasy.py's waivers().

const { verifyUser, sendJson } = require("./_lib");
const { getLeagueForUser, scoredBoard } = require("./_board");
const { removeDrafted, recommend } = require("./_draftState");
const { leagueRosters, resolveSleeperIds } = require("./_roster");

module.exports = async function handler(req, res) {
  const user = await verifyUser(req);
  if (!user) return sendJson(res, 401, { detail: "Missing bearer token." });
  const leagueId = req.query.league_id, rosterId = req.query.roster_id;
  const topN = req.query.top_n ? Math.max(1, Math.min(50, parseInt(req.query.top_n, 10))) : 15;
  if (!leagueId || !rosterId) return sendJson(res, 422, { detail: "league_id and roster_id are required" });

  const league = await getLeagueForUser(user.id, leagueId);
  if (!league) return sendJson(res, 404, { detail: "league not found or not imported" });

  let rosters;
  try { rosters = await leagueRosters(league.sleeper_league_id); }
  catch (e) { return sendJson(res, 502, { detail: "Sleeper request failed" }); }
  const myRoster = rosters.find(r => String(r.roster_id) === String(rosterId));
  if (!myRoster) return sendJson(res, 404, { detail: "roster not found in this league" });

  let board;
  try { board = await scoredBoard(league); }
  catch (e) { return sendJson(res, 500, { detail: "Could not build the draft board" }); }

  const rosteredSleeperIds = rosters.flatMap(r => r.players || []);
  const rosteredPlayerIds = await resolveSleeperIds(rosteredSleeperIds);
  const available = removeDrafted(board, rosteredPlayerIds.map(pid => ({ player_id: pid })));

  const myPlayerIds = new Set(await resolveSleeperIds(myRoster.players || []));
  const boardByPlayer = new Map(board.map(p => [p.player_id, p]));
  const myCurrent = [...myPlayerIds].map(pid => boardByPlayer.get(pid)).filter(Boolean);

  const recs = recommend(available, league.roster_positions, myCurrent, topN);
  sendJson(res, 200, { league_id: leagueId, roster_id: rosterId, available_count: available.length, recommendations: recs });
};
