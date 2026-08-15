// Consolidated Fantasy Draft Assistant router -- ONE Vercel Serverless Function handling
// every /api/fantasy/* route, instead of one function per route.
//
// Why: Vercel's Hobby plan caps a deployment at 12 Serverless Functions. This project's
// static/api/ directory had grown to 17 real functions (4 AI + 13 fantasy, undercounting
// the underscore-prefixed helper modules which Vercel correctly excludes from the count) --
// found live 2026-08 when a deploy started failing with "No more than 12 Serverless
// Functions can be added to a Deployment on the Hobby plan," silently freezing production
// on whatever the last successful deploy was. Consolidating the 13 fantasy route files
// (leagues/*, draft-board, draft/*/live, lineup, waivers, trade/analyze, sleeper/*,
// unmatched-players) into this single catch-all brings the total to 5, with real headroom.
//
// Every handler function below is the UNCHANGED body of its former standalone file (see git
// history for static/api/fantasy/leagues/[id].js etc.) -- only how the path parameter (id/
// draftId/userId/username) is obtained changed, from a named dynamic-route segment to an
// index into Vercel's `path` catch-all array. External URLs are identical to before; this
// is a routing consolidation, not an API change.

const { verifyUser, sleeperGet, pgSelect, pgInsert, pgPatch, eq, inList, sendJson, nowIso } = require("./_lib");
const { getLeagueForUser, scoredBoard } = require("./_board");
const { removeDrafted, positionalRuns, recommend } = require("./_draftState");
const { scorePlayers } = require("./_scoring");
const { optimalLineup, lineupDelta } = require("./_lineup");
const { findRoster, resolveSleeperIds, leagueRosters } = require("./_roster");
const { evaluateTrade } = require("./_trade");
const crypto = require("node:crypto");

const PROJECTED_GAMES = 17;

// GET /api/fantasy/leagues -- auth required. Port of routes_fantasy.py's my_leagues().
async function myLeagues(req, res) {
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
}

// POST /api/fantasy/leagues/import -- auth required. Port of routes_fantasy.py's import_league().
async function importLeague(req, res) {
  if (req.method !== "POST") return sendJson(res, 405, { detail: "Method not allowed" });
  const user = await verifyUser(req);
  if (!user) return sendJson(res, 401, { detail: "Missing bearer token." });

  let body;
  try { body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {}); }
  catch (_) { body = {}; }
  const sleeperLeagueId = body.sleeper_league_id;
  if (!sleeperLeagueId) return sendJson(res, 422, { detail: "sleeper_league_id is required" });

  let league;
  try { league = await sleeperGet(`/league/${encodeURIComponent(sleeperLeagueId)}`); }
  catch (e) { return sendJson(res, 502, { detail: "Sleeper request failed" }); }
  if (!league) return sendJson(res, 404, { detail: "Sleeper league not found" });

  const now = nowIso();
  const fields = {
    name: league.name || null, season: league.season || null,
    league_size: league.total_rosters || null,
    scoring_settings: JSON.stringify(league.scoring_settings || {}),
    roster_positions: JSON.stringify(league.roster_positions || []),
    updated_at: now,
  };
  try {
    const existing = await pgSelect("fantasy_leagues",
      `user_id=${eq(user.id)}&sleeper_league_id=${eq(sleeperLeagueId)}&select=id`);
    let out;
    if (existing.length) {
      const updated = await pgPatch("fantasy_leagues", `id=${eq(existing[0].id)}`, fields);
      out = updated[0];
    } else {
      const created = await pgInsert("fantasy_leagues", [{
        id: crypto.randomBytes(16).toString("hex"), user_id: user.id, sleeper_league_id: sleeperLeagueId,
        ...fields, imported_at: now,
      }]);
      out = created[0];
    }
    sendJson(res, 200, { league: { ...out, scoring_settings: JSON.parse(out.scoring_settings),
                                   roster_positions: JSON.parse(out.roster_positions) } });
  } catch (e) {
    sendJson(res, 500, { detail: "Could not save league" });
  }
}

// GET /api/fantasy/leagues/:id -- public, no auth. Port of routes_fantasy.py's league_detail().
async function leagueDetail(req, res, id) {
  let league;
  try { league = await sleeperGet(`/league/${encodeURIComponent(id)}`); }
  catch (e) { return sendJson(res, 502, { detail: "Sleeper request failed" }); }
  if (!league) return sendJson(res, 404, { detail: "Sleeper league not found" });
  sendJson(res, 200, league);
}

// GET /api/fantasy/leagues/:id/drafts -- public, no auth. Port of routes_fantasy.py's league_drafts().
async function leagueDrafts(req, res, id) {
  let drafts;
  try { drafts = await sleeperGet(`/league/${encodeURIComponent(id)}/drafts`); }
  catch (e) { return sendJson(res, 502, { detail: "Sleeper request failed" }); }
  sendJson(res, 200, { drafts: drafts || [] });
}

// GET /api/fantasy/leagues/:id/rosters -- public, no auth. Port of routes_fantasy.py's league_rosters_route().
async function leagueRostersRoute(req, res, id) {
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
}

// GET /api/fantasy/draft-board?league_id= -- auth required. Port of routes_fantasy.py's draft_board().
async function draftBoard(req, res) {
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
}

// GET /api/fantasy/draft/:draftId/live?league_id=&roster_id= -- auth required. Port of
// routes_fantasy.py's live_draft().
async function draftLive(req, res, draftId) {
  const user = await verifyUser(req);
  if (!user) return sendJson(res, 401, { detail: "Missing bearer token." });
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
}

// GET /api/fantasy/lineup?league_id=&roster_id=&week= -- auth required. Port of
// routes_fantasy.py's lineup_route().
async function lineup(req, res) {
  const user = await verifyUser(req);
  if (!user) return sendJson(res, 401, { detail: "Missing bearer token." });
  const leagueId = req.query.league_id, rosterId = req.query.roster_id;
  if (!leagueId || !rosterId) return sendJson(res, 422, { detail: "league_id and roster_id are required" });

  const league = await getLeagueForUser(user.id, leagueId);
  if (!league) return sendJson(res, 404, { detail: "league not found or not imported" });

  let week = req.query.week ? parseInt(req.query.week, 10) : null;
  if (!week) {
    try { const state = await sleeperGet("/state/nfl"); week = (state && state.week) || 1; }
    catch (e) { week = 1; }
  }

  let roster;
  try { roster = await findRoster(league.sleeper_league_id, rosterId); }
  catch (e) { return sendJson(res, 502, { detail: "Sleeper request failed" }); }
  if (!roster) return sendJson(res, 404, { detail: "roster not found in this league" });

  const rosterPlayerIds = await resolveSleeperIds(roster.players || []);
  const currentStarterIds = await resolveSleeperIds(roster.starters || []);

  const emptyResponse = () => sendJson(res, 200, {
    league_id: leagueId, roster_id: rosterId, week, starters: [], bench: [],
    projected_points: 0.0, delta: null,
    note: "No projections available yet for this roster's players.",
  });
  if (!rosterPlayerIds.length) return emptyResponse();

  const season = league.season ? parseInt(league.season, 10) : new Date().getUTCFullYear();
  let projections, players;
  try {
    projections = await pgSelect("fantasy_projections",
      `provider=eq.nflverse&season=${eq(season)}&week=is.null&player_id=${inList(rosterPlayerIds)}&select=player_id,stats`);
    const playerIds = projections.map(p => p.player_id);
    players = playerIds.length
      ? await pgSelect("fantasy_players", `id=${inList(playerIds)}&select=id,full_name,position,team`)
      : [];
  } catch (e) {
    return sendJson(res, 500, { detail: "Could not load projections" });
  }
  const playerById = new Map(players.map(p => [p.id, p]));

  const enriched = [];
  for (const proj of projections) {
    const player = playerById.get(proj.player_id);
    if (!player) continue;
    const seasonStats = typeof proj.stats === "string" ? JSON.parse(proj.stats) : proj.stats;
    const weekStats = {};
    for (const [k, v] of Object.entries(seasonStats)) weekStats[k] = Math.round((v / PROJECTED_GAMES) * 100) / 100;
    enriched.push({ player_id: proj.player_id, name: player.full_name, position: player.position,
                    team: player.team, stats: weekStats });
  }
  if (!enriched.length) return emptyResponse();

  const rosterBoard = scorePlayers(enriched, league.scoring_settings);
  const optimal = optimalLineup(rosterBoard, league.roster_positions);
  const scoredById = new Map(rosterBoard.map(p => [p.player_id, p]));
  const delta = lineupDelta(optimal, currentStarterIds, scoredById);

  sendJson(res, 200, { league_id: leagueId, roster_id: rosterId, week, ...optimal, delta });
}

// GET /api/fantasy/waivers?league_id=&roster_id=&top_n= -- auth required. Port of
// routes_fantasy.py's waivers().
async function waivers(req, res) {
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
}

// POST /api/fantasy/trade/analyze {league_id, side_a, side_b} -- auth required. Port of
// routes_fantasy.py's trade_analyze().
async function tradeAnalyze(req, res) {
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
}

// GET /api/fantasy/unmatched-players -- admin only. Port of routes_fantasy.py's unmatched_players().
async function unmatchedPlayers(req, res) {
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
}

// GET /api/fantasy/sleeper/leagues/:userId?season= -- public, no auth. Port of
// routes_fantasy.py's sleeper_leagues().
async function sleeperLeagues(req, res, userId) {
  const season = req.query.season || "";
  let leagues;
  try { leagues = await sleeperGet(`/user/${encodeURIComponent(userId)}/leagues/nfl/${encodeURIComponent(season)}`); }
  catch (e) { return sendJson(res, 502, { detail: "Sleeper request failed" }); }
  sendJson(res, 200, { leagues: leagues || [] });
}

// GET /api/fantasy/sleeper/user/:username -- public, no auth. Port of routes_fantasy.py's sleeper_user().
async function sleeperUser(req, res, username) {
  let user;
  try { user = await sleeperGet(`/user/${encodeURIComponent(username)}`); }
  catch (e) { return sendJson(res, 502, { detail: "Sleeper request failed" }); }
  if (!user) return sendJson(res, 404, { detail: "Sleeper username not found" });
  sendJson(res, 200, user);
}

module.exports = async function handler(req, res) {
  const raw = req.query.path;
  const path = Array.isArray(raw) ? raw : (raw ? [raw] : []);
  const [seg0, seg1, seg2] = path;

  if (seg0 === "leagues") {
    if (path.length === 1) return myLeagues(req, res);
    if (seg1 === "import") return importLeague(req, res);
    if (path.length === 2) return leagueDetail(req, res, seg1);
    if (path.length === 3 && seg2 === "drafts") return leagueDrafts(req, res, seg1);
    if (path.length === 3 && seg2 === "rosters") return leagueRostersRoute(req, res, seg1);
  } else if (seg0 === "draft-board" && path.length === 1) {
    return draftBoard(req, res);
  } else if (seg0 === "draft" && path.length === 3 && seg2 === "live") {
    return draftLive(req, res, seg1);
  } else if (seg0 === "lineup" && path.length === 1) {
    return lineup(req, res);
  } else if (seg0 === "waivers" && path.length === 1) {
    return waivers(req, res);
  } else if (seg0 === "trade" && path.length === 2 && seg1 === "analyze") {
    return tradeAnalyze(req, res);
  } else if (seg0 === "unmatched-players" && path.length === 1) {
    return unmatchedPlayers(req, res);
  } else if (seg0 === "sleeper" && seg1 === "leagues" && path.length === 3) {
    return sleeperLeagues(req, res, seg2);
  } else if (seg0 === "sleeper" && seg1 === "user" && path.length === 3) {
    return sleeperUser(req, res, seg2);
  }

  return sendJson(res, 404, { detail: "Not found" });
};
