// GET /api/fantasy/leagues -- auth required. Port of routes_fantasy.py's my_leagues().
// Vercel maps this file to /api/fantasy/leagues (index.js is the directory's own route).

const { verifyUser, pgSelect, eq, sendJson } = require("../_lib");

module.exports = async function handler(req, res) {
  const user = await verifyUser(req);
  if (!user) return sendJson(res, 401, { detail: "Missing bearer token." });
  try {
    const rows = await pgSelect("fantasy_leagues", `user_id=${eq(user.id)}&order=imported_at.desc&select=*`);
    const leagues = rows.map(r => ({ ...r, scoring_settings: JSON.parse(r.scoring_settings || "{}"),
                                     roster_positions: JSON.parse(r.roster_positions || "[]") }));
    sendJson(res, 200, { leagues });
  } catch (e) {
    sendJson(res, 500, { detail: "Could not load leagues" });
  }
};
