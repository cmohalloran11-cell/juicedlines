// POST /api/fantasy/trade/analyze {league_id, side_a, side_b} -- auth required. Port of
// routes_fantasy.py's trade_analyze().

const { verifyUser, sendJson } = require("../_lib");
const { getLeagueForUser, scoredBoard } = require("../_board");
const { evaluateTrade } = require("../_trade");

module.exports = async function handler(req, res) {
  if (req.method !== "POST") return sendJson(res, 405, { detail: "Method not allowed" });
  const user = await verifyUser(req);
  if (!user) return sendJson(res, 401, { detail: "Missing bearer token." });

  let body;
  try { body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {}); }
  catch (_) { body = {}; }
  const { league_id: leagueId, side_a: sideA, side_b: sideB } = body;
  if (!leagueId || !Array.isArray(sideA) || !Array.isArray(sideB) || !sideA.length || !sideB.length) {
    return sendJson(res, 422, { detail: "both side_a and side_b need at least one player" });
  }

  const league = await getLeagueForUser(user.id, leagueId);
  if (!league) return sendJson(res, 404, { detail: "league not found or not imported" });

  let board;
  try { board = await scoredBoard(league); }
  catch (e) { return sendJson(res, 500, { detail: "Could not build the draft board" }); }

  const boardByPlayer = Object.fromEntries(board.map(p => [p.player_id, p]));
  sendJson(res, 200, evaluateTrade(boardByPlayer, sideA, sideB));
};
