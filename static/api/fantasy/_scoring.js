// Port of fantasy/scoring.py -- pure scoring engine, no hardcoded PPR/half-PPR anywhere.
// See fantasy/scoring.py for the full rationale; keep this in sync with that file.

function scoreStats(stats, scoringSettings) {
  let total = 0;
  for (const [k, v] of Object.entries(scoringSettings || {})) {
    if (Object.prototype.hasOwnProperty.call(stats || {}, k)) total += Number(stats[k] || 0) * Number(v);
  }
  return Math.round(total * 100) / 100;
}

function scorePlayers(projections, scoringSettings) {
  return projections.map(p => ({ ...p, fantasy_points: scoreStats(p.stats || {}, scoringSettings) }));
}

module.exports = { scoreStats, scorePlayers };
