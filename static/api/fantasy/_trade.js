// Port of fantasy/trade.py -- pure trade evaluation on VOR. Keep in sync.

const EVEN_THRESHOLD = 1.0;

function side(boardByPlayer, ids) {
  const players = [], missing = [];
  let total = 0;
  for (const pid of ids) {
    const p = boardByPlayer[pid];
    if (!p) { missing.push(pid); continue; }
    players.push(p);
    total += p.vor || 0;
  }
  return { players, missing_player_ids: missing, total_vor: Math.round(total * 100) / 100 };
}

function evaluateTrade(boardByPlayer, sideAIds, sideBIds) {
  const a = side(boardByPlayer, sideAIds);
  const b = side(boardByPlayer, sideBIds);
  const diff = Math.round((a.total_vor - b.total_vor) * 100) / 100;
  let verdict;
  if (Math.abs(diff) < EVEN_THRESHOLD) verdict = "Roughly even trade";
  else if (diff > 0) verdict = `Side A gains ${diff} VOR`;
  else verdict = `Side B gains ${-diff} VOR`;
  return { side_a: a, side_b: b, vor_diff: diff, verdict };
}

module.exports = { evaluateTrade };
