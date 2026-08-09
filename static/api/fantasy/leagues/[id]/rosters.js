// GET /api/fantasy/leagues/:id/rosters -- public, no auth. Port of routes_fantasy.py's
// league_rosters_route(): roster_id + owner display name for every team, so the UI can
// offer "which team is yours?" instead of a raw numeric Sleeper roster_id.

const { sleeperGet, sendJson } = require("../../_lib");

module.exports = async function handler(req, res) {
  const { id } = req.query;
  let rosters, users;
  try {
    [rosters, users] = await Promise.all([
      sleeperGet(`/league/${encodeURIComponent(id)}/rosters`),
      sleeperGet(`/league/${encodeURIComponent(id)}/users`),
    ]);
  } catch (e) { return sendJson(res, 502, { detail: "Sleeper request failed" }); }

  const usersById = new Map((users || []).map(u => [u.user_id, u]));
  const out = (rosters || []).map(r => {
    const owner = usersById.get(r.owner_id) || {};
    return {
      roster_id: r.roster_id,
      owner_display_name: owner.display_name || owner.username || "Unknown",
      team_name: (owner.metadata || {}).team_name || null,
    };
  });
  sendJson(res, 200, { rosters: out });
};
