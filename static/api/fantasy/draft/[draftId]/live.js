// GET /api/fantasy/draft/:draftId/live?league_id=&roster_id= -- auth required. Port of
// routes_fantasy.py's live_draft(). No 10s TTL cache here (see _lib.js's architecture
// note); the frontend already polls this at a fixed 10s cadence, so a serverless function
// with no shared cache is an acceptable cost, not a correctness issue.

const { verifyUser, sleeperGet, sendJson } = require("../../_lib");
const { getLeagueForUser, scoredBoard } = require("../../_board");
const { removeDrafted, positionalRuns, recommend } = require("../../_draftState");

module.exports = async function handler(req, res) {
  const user = await verifyUser(req);
  if (!user) return sendJson(res, 401, { detail: "Missing bearer token." });
  const { draftId } = req.query;
  const leagueId = req.query.league_id;
  const rosterId = req.query.roster_id;
  if (!leagueId) return sendJson(res, 422, { detail: "league_id is required" });

  const league = await getLeagueForUser(user.id, leagueId);
  if (!league) return sendJson(res, 404, { detail: "league not found or not imported" });

  let picks;
  try { picks = (await sleeperGet(`/draft/${encodeURIComponent(draftId)}/picks`)) || []; }
  catch (e) { return sendJson(res, 502, { detail: "Sleeper request failed" }); }

  let board;
  try { board = await scoredBoard(league); }
  catch (e) { return sendJson(res, 500, { detail: "Could not build the draft board" }); }

  const boardByPlayer = new Map(board.map(p => [p.player_id, p]));
  for (const pick of picks) {
    if (!pick.position) {
      const meta = pick.metadata || {};
      const hit = boardByPlayer.get(pick.player_id);
      pick.position = (hit && hit.position) || meta.position || null;
    }
  }

  const available = removeDrafted(board, picks);
  const runs = positionalRuns(picks);
  const draftedByUser = rosterId != null ? picks.filter(p => String(p.roster_id) === String(rosterId)) : [];
  const recs = recommend(available, league.roster_positions, draftedByUser);

  sendJson(res, 200, {
    draft_id: draftId, picks_made: picks.length, available,
    positional_runs: runs, recommendations: recs,
  });
};
