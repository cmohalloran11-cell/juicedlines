// Shared board-building helper used by draft-board.js, draft/[draftId]/live.js, waivers.js,
// and trade/analyze.js -- port of routes_fantasy.py's _league_or_404 / _scored_board.

const { pgSelect, eq, inList } = require("./_lib");
const { scorePlayers } = require("./_scoring");
const { computeVor, assignTiers } = require("./_vor");

async function getLeagueForUser(userId, leagueId) {
  const rows = await pgSelect("fantasy_leagues", `id=${eq(leagueId)}&user_id=${eq(userId)}&select=*`);
  if (!rows.length) return null;
  const row = rows[0];
  return {
    ...row,
    scoring_settings: JSON.parse(row.scoring_settings || "{}"),
    roster_positions: JSON.parse(row.roster_positions || "[]"),
  };
}

// Season-long (week=NULL) VOR-ranked, tiered board for one league's actual scoring rules +
// roster shape. Projections are pre-synced by fantasy-sync.yml (Python, unmodified) --
// this function only ever reads, never fetches nflverse/Sleeper's bulk data itself.
async function scoredBoard(league) {
  const season = league.season ? parseInt(league.season, 10) : new Date().getUTCFullYear();
  const projections = await pgSelect("fantasy_projections",
    `provider=eq.nflverse&season=${eq(season)}&week=is.null&select=player_id,stats`);
  if (!projections.length) return [];

  const playerIds = projections.map(p => p.player_id);
  const players = await pgSelect("fantasy_players", `id=${inList(playerIds)}&select=id,full_name,position,team`);
  const playerById = new Map(players.map(p => [p.id, p]));

  const enriched = [];
  for (const proj of projections) {
    const player = playerById.get(proj.player_id);
    if (!player) continue;
    enriched.push({
      player_id: proj.player_id, name: player.full_name, position: player.position, team: player.team,
      stats: typeof proj.stats === "string" ? JSON.parse(proj.stats) : proj.stats,
    });
  }
  const scored = scorePlayers(enriched, league.scoring_settings);
  const ranked = computeVor(scored, league.roster_positions, league.league_size || 12);
  return assignTiers(ranked);
}

module.exports = { getLeagueForUser, scoredBoard };
