// Port of fantasy/vor.py -- pure VOR + tiering. Keep in sync with that file.

const FLEX_ELIGIBLE = {
  FLEX: new Set(["RB", "WR", "TE"]),
  SUPER_FLEX: new Set(["QB", "RB", "WR", "TE"]),
  REC_FLEX: new Set(["WR", "TE"]),
  WRRB_FLEX: new Set(["RB", "WR"]),
};
const BENCH_SLOTS = new Set(["BN", "IR", "TAXI"]);

function starterCounts(rosterPositions, leagueSize) {
  const counts = {};
  for (let slot of rosterPositions) {
    slot = slot.toUpperCase();
    if (BENCH_SLOTS.has(slot)) continue;
    const eligible = FLEX_ELIGIBLE[slot];
    if (eligible) {
      for (const pos of eligible) counts[pos] = (counts[pos] || 0) + 1 / eligible.size;
    } else {
      counts[slot] = (counts[slot] || 0) + 1;
    }
  }
  const out = {};
  for (const [pos, n] of Object.entries(counts)) out[pos] = n * leagueSize;
  return out;
}

function replacementLevels(scoredPlayers, rosterPositions, leagueSize) {
  const counts = starterCounts(rosterPositions, leagueSize);
  const byPos = {};
  for (const p of scoredPlayers) (byPos[p.position] = byPos[p.position] || []).push(p.fantasy_points);
  const levels = {};
  for (const [pos, n] of Object.entries(counts)) {
    const pts = (byPos[pos] || []).slice().sort((a, b) => b - a);
    if (!pts.length) continue;
    const idx = Math.max(Math.min(Math.round(n) - 1, pts.length - 1), 0);
    levels[pos] = pts[idx];
  }
  return levels;
}

function computeVor(scoredPlayers, rosterPositions, leagueSize) {
  const levels = replacementLevels(scoredPlayers, rosterPositions, leagueSize);
  const out = scoredPlayers.map(p => {
    const level = levels[p.position];
    const vor = level !== undefined ? Math.round((p.fantasy_points - level) * 100) / 100 : null;
    return { ...p, vor };
  });
  out.sort((a, b) => {
    const aNull = a.vor === null, bNull = b.vor === null;
    if (aNull !== bNull) return aNull ? 1 : -1;
    return (b.vor || 0) - (a.vor || 0);
  });
  return out;
}

function assignTiers(vorRanked, gapThreshold = 0.75) {
  const ranked = vorRanked.filter(p => p.vor !== null);
  const unranked = vorRanked.filter(p => p.vor === null);
  if (ranked.length < 2) {
    ranked.forEach(p => { p.tier = 1; });
    unranked.forEach(p => { p.tier = ranked.length ? 2 : 1; });
    return ranked.concat(unranked);
  }
  const gaps = [];
  for (let i = 0; i < ranked.length - 1; i++) gaps.push(ranked[i].vor - ranked[i + 1].vor);
  const meanGap = gaps.reduce((a, b) => a + b, 0) / gaps.length;
  const variance = gaps.reduce((a, g) => a + (g - meanGap) ** 2, 0) / gaps.length;
  const cutoff = meanGap + gapThreshold * Math.sqrt(variance);
  let tier = 1;
  ranked[0].tier = tier;
  for (let i = 0; i < gaps.length; i++) {
    if (gaps[i] > cutoff && gaps[i] > 0) tier += 1;
    ranked[i + 1].tier = tier;
  }
  unranked.forEach(p => { p.tier = tier + 1; });
  return ranked.concat(unranked);
}

module.exports = { FLEX_ELIGIBLE, BENCH_SLOTS, starterCounts, replacementLevels, computeVor, assignTiers };
