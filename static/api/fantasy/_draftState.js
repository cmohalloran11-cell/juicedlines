// Port of fantasy/draft_state.py -- pure live-draft logic. Keep in sync with that file.

const { FLEX_ELIGIBLE, BENCH_SLOTS } = require("./_vor");

function removeDrafted(tieredBoard, picks) {
  const drafted = new Set(picks.filter(p => p.player_id).map(p => p.player_id));
  return tieredBoard.filter(p => !drafted.has(p.player_id));
}

function positionalRuns(picks, window = 5, runThreshold = 3) {
  const recent = picks.slice(-window);
  const counts = {};
  for (const p of recent) if (p.position) counts[p.position] = (counts[p.position] || 0) + 1;
  const out = {};
  for (const [pos, n] of Object.entries(counts)) if (n >= runThreshold) out[pos] = n;
  return out;
}

function remainingNeeds(rosterPositions, draftedByUser) {
  const needed = {};
  for (let slot of rosterPositions) {
    slot = slot.toUpperCase();
    if (BENCH_SLOTS.has(slot)) continue;
    needed[slot] = (needed[slot] || 0) + 1;
  }
  const filled = {};
  for (const p of draftedByUser) if (p.position) filled[p.position] = (filled[p.position] || 0) + 1;
  const remaining = {};
  for (const [slot, n] of Object.entries(needed)) {
    if (FLEX_ELIGIBLE[slot]) { remaining[slot] = n; continue; }
    remaining[slot] = Math.max(0, n - (filled[slot] || 0));
  }
  return remaining;
}

function needWeight(position, needs) {
  if ((needs[position] || 0) > 0) return 1.15;
  for (const [slot, n] of Object.entries(needs)) {
    if (n > 0 && FLEX_ELIGIBLE[slot] && FLEX_ELIGIBLE[slot].has(position)) return 1.08;
  }
  return 1.0;
}

function recommend(board, rosterPositions, draftedByUser, topN = 10) {
  const needs = remainingNeeds(rosterPositions, draftedByUser);
  const weighted = board.map(p => {
    const nw = needWeight(p.position || "", needs);
    return { ...p, need_weight: nw, recommendation_score: Math.round((p.vor || 0) * nw * 100) / 100 };
  });
  weighted.sort((a, b) => b.recommendation_score - a.recommendation_score);
  return weighted.slice(0, topN);
}

module.exports = { removeDrafted, positionalRuns, remainingNeeds, needWeight, recommend };
