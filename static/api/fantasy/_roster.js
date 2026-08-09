// Roster resolution shared by lineup.js and waivers.js -- port of routes_fantasy.py's
// _league_rosters / _find_roster / _resolve_sleeper_ids.

const { pgSelect, inList, sleeperGet } = require("./_lib");

async function leagueRosters(sleeperLeagueId) {
  return (await sleeperGet(`/league/${encodeURIComponent(sleeperLeagueId)}/rosters`)) || [];
}

async function findRoster(sleeperLeagueId, rosterId) {
  const rosters = await leagueRosters(sleeperLeagueId);
  return rosters.find(r => String(r.roster_id) === String(rosterId)) || null;
}

// Sleeper player ids -> canonical player_id, silently skipping anything unmapped -- same
// "omit rather than mis-map" policy as everywhere else in the fantasy data layer.
async function resolveSleeperIds(sleeperIds) {
  if (!sleeperIds || !sleeperIds.length) return [];
  const rows = await pgSelect("fantasy_player_ids",
    `source=eq.sleeper&source_id=${inList(sleeperIds)}&select=source_id,player_id`);
  const byId = new Map(rows.map(r => [r.source_id, r.player_id]));
  return sleeperIds.map(sid => byId.get(sid)).filter(Boolean);
}

module.exports = { leagueRosters, findRoster, resolveSleeperIds };
