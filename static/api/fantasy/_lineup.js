// Port of fantasy/lineup.py -- pure optimal-starting-lineup assignment. Keep in sync.

const { FLEX_ELIGIBLE, BENCH_SLOTS } = require("./_vor");

function eligiblePositions(slot) {
  return FLEX_ELIGIBLE[slot] || new Set([slot]);
}

function optimalLineup(scoredPlayers, rosterPositions) {
  const startingSlots = rosterPositions.map(s => s.toUpperCase()).filter(s => !BENCH_SLOTS.has(s));
  const order = startingSlots.map((_, i) => i).sort(
    (a, b) => eligiblePositions(startingSlots[a]).size - eligiblePositions(startingSlots[b]).size);

  const remaining = new Map(scoredPlayers.map(p => [p.player_id, p]));
  const starters = new Array(startingSlots.length).fill(null);
  for (const idx of order) {
    const slot = startingSlots[idx];
    const eligible = eligiblePositions(slot);
    let best = null;
    for (const p of remaining.values()) {
      if (eligible.has(p.position) && (best === null || p.fantasy_points > best.fantasy_points)) best = p;
    }
    if (!best) { starters[idx] = { slot, player: null }; continue; }
    starters[idx] = { slot, player: best };
    remaining.delete(best.player_id);
  }
  const projectedPoints = Math.round(
    starters.reduce((s, x) => s + (x.player ? x.player.fantasy_points : 0), 0) * 100) / 100;
  const bench = Array.from(remaining.values()).sort((a, b) => b.fantasy_points - a.fantasy_points);
  return { starters, bench, projected_points: projectedPoints };
}

function lineupDelta(optimal, currentStarterIds, scoredById) {
  const currentSet = new Set(currentStarterIds);
  let currentPoints = 0;
  for (const pid of currentStarterIds) if (scoredById.has(pid)) currentPoints += scoredById.get(pid).fantasy_points;
  currentPoints = Math.round(currentPoints * 100) / 100;
  const optimalIds = new Set(optimal.starters.filter(s => s.player).map(s => s.player.player_id));
  const start = [...optimalIds].filter(pid => !currentSet.has(pid));
  const sit = currentStarterIds.filter(pid => !optimalIds.has(pid));
  return {
    current_points: currentPoints,
    optimal_points: optimal.projected_points,
    points_gained: Math.round((optimal.projected_points - currentPoints) * 100) / 100,
    start, sit,
  };
}

module.exports = { optimalLineup, lineupDelta };
