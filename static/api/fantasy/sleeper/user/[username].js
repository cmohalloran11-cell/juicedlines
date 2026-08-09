// GET /api/fantasy/sleeper/user/:username -- public, no auth. Port of
// routes_fantasy.py's sleeper_user().

const { sleeperGet, sendJson } = require("../../_lib");

module.exports = async function handler(req, res) {
  const { username } = req.query;
  let user;
  try { user = await sleeperGet(`/user/${encodeURIComponent(username)}`); }
  catch (e) { return sendJson(res, 502, { detail: "Sleeper request failed" }); }
  if (!user) return sendJson(res, 404, { detail: "Sleeper username not found" });
  sendJson(res, 200, user);
};
