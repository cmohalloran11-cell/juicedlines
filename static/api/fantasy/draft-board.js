// GET /api/fantasy/draft-board?league_id= -- auth required. Port of routes_fantasy.py's
// draft_board().

const { verifyUser, sendJson } = require("./_lib");
const { getLeagueForUser, scoredBoard } = require("./_board");

module.exports = async function handler(req, res) {
  const user = await verifyUser(req);
  if (!user) return sendJson(res, 401, { detail: "Missing bearer token." });
  const leagueId = req.query.league_id;
  if (!leagueId) return sendJson(res, 422, { detail: "league_id is required" });

  const league = await getLeagueForUser(user.id, leagueId);
  if (!league) return sendJson(res, 404, { detail: "league not found or not imported" });
  try {
    const board = await scoredBoard(league);
    sendJson(res, 200, { league_id: leagueId, board });
  } catch (e) {
    sendJson(res, 500, { detail: "Could not build the draft board" });
  }
};
