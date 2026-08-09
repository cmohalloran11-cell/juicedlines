// GET /api/fantasy/lineup?league_id=&roster_id=&week= -- auth required. Port of
// routes_fantasy.py's lineup_route().
//
// Weekly value: each RAW STAT is divided by PROJECTED_GAMES independently, THEN scored --
// not "season fantasy_points / 17". That matches the Python nflverse adapter's week=<int>
// path exactly (projected_games=1 vs PROJECTED_GAMES=17 applied per-stat -- see
// fantasy/projections/nflverse_adapter.py's get_projections docstring), NOT a separately-
// synced weekly dataset. This adapter has no per-week matchup/injury signal to vary it week
// to week, so there is nothing a second sync would add; computing it here avoids needing
// fantasy-sync.yml to run a second job. (Dividing the already-rounded season total instead
// would give a very slightly different number due to rounding order -- not what the Python
// reference does.)

const { verifyUser, sleeperGet, pgSelect, inList, eq, sendJson } = require("./_lib");
const { getLeagueForUser } = require("./_board");
const { scorePlayers } = require("./_scoring");
const { optimalLineup, lineupDelta } = require("./_lineup");
const { findRoster, resolveSleeperIds } = require("./_roster");

const PROJECTED_GAMES = 17;

module.exports = async function handler(req, res) {
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
};
