// GET /api/fantasy/unmatched-players -- admin only. Port of routes_fantasy.py's
// unmatched_players().

const { verifyUser, pgSelect, sendJson } = require("./_lib");

module.exports = async function handler(req, res) {
  const user = await verifyUser(req);
  if (!user) return sendJson(res, 401, { detail: "Missing bearer token." });
  if (user.role !== "ADMIN" && user.role !== "SUPER_ADMIN") {
    return sendJson(res, 403, { detail: "admin only" });
  }
  try {
    const rows = await pgSelect("fantasy_unmatched_players", "status=eq.pending&order=created_at.desc&select=*");
    sendJson(res, 200, { unmatched: rows });
  } catch (e) {
    sendJson(res, 500, { detail: "Could not load unmatched players" });
  }
};
