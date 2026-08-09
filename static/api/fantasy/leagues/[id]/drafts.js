// GET /api/fantasy/leagues/:id/drafts -- public, no auth. Port of
// routes_fantasy.py's league_drafts().

const { sleeperGet, sendJson } = require("../../_lib");

module.exports = async function handler(req, res) {
  const { id } = req.query;
  let drafts;
  try { drafts = await sleeperGet(`/league/${encodeURIComponent(id)}/drafts`); }
  catch (e) { return sendJson(res, 502, { detail: "Sleeper request failed" }); }
  sendJson(res, 200, { drafts: drafts || [] });
};
