"""
Date-strict match-win backtest with calibration by surface and tour.

For each test season: fit rates + Elo ONLY on prior seasons (no post-match info for
the match being predicted), then for every test match assign the two players by a
neutral key (player id order — never by outcome) and predict P(player1 wins) with the
closed-form model. Bucket predictions vs realized win rate; report ECE / Brier /
accuracy overall and split by surface. (Closing-line bias needs a historical odds
feed — flagged, not faked.)
"""

from __future__ import annotations

from collections import defaultdict

from ..data import get_history
from ..model import rates as R
from ..model.matchup import match_win_analytic


def run(tour: str, fit_years: list[int], test_years: list[int]) -> dict:
    hist = get_history()
    fit_matches = hist.player_matches(tour, fit_years)
    base, by_id = R.fit(fit_matches, tour)
    test = [m for m in hist.player_matches(tour, test_years) if m.won]  # one row per match

    def rates(pid, name):
        return by_id.get(pid) or R.PlayerRates(pid, name, tour, base.spw_avg, base.rpw_avg,
                                               base.ace_rate_avg, base.df_rate_avg,
                                               base.pts_per_svgame_avg)

    pairs = []                       # (pred_p1, outcome_p1_won, surface)
    for m in test:
        if not (m.player_id and m.opp_id):
            continue
        # neutral assignment: player1 = smaller id (independent of who won)
        if m.player_id <= m.opp_id:
            p1, p1n, p2, p2n, p1_won = m.player_id, m.player, m.opp_id, m.opp, True
        else:
            p1, p1n, p2, p2n, p1_won = m.opp_id, m.opp, m.player_id, m.player, False
        bo = m.best_of or 3
        pred = match_win_analytic(rates(p1, p1n), rates(p2, p2n), m.surface or "Hard", base, bo)
        pairs.append((pred, 1.0 if p1_won else 0.0, m.surface or "Hard"))

    return _report(pairs, tour, fit_years, test_years)


def _metrics(pairs):
    n = len(pairs)
    if not n:
        return None
    brier = sum((p - o) ** 2 for p, o, _ in pairs) / n
    acc = sum(1 for p, o, _ in pairs if (p >= 0.5) == (o == 1.0)) / n
    ece = 0.0
    buckets = defaultdict(lambda: [0, 0.0, 0.0])   # bin -> [n, sum_pred, sum_out]
    for p, o, _ in pairs:
        b = min(9, int(p * 10))
        buckets[b][0] += 1; buckets[b][1] += p; buckets[b][2] += o
    for b, (cnt, sp, so) in buckets.items():
        ece += abs(sp / cnt - so / cnt) * cnt / n
    return {"n": n, "brier": round(brier, 4), "acc": round(acc, 4),
            "ece": round(ece, 4), "buckets": dict(sorted(buckets.items()))}


def _report(pairs, tour, fit_years, test_years) -> dict:
    overall = _metrics(pairs)
    print(f"\n{tour} match-win backtest — fit {fit_years[0]}-{fit_years[-1]}, "
          f"test {test_years[0]}-{test_years[-1]}  ({overall['n']} matches)")
    print(f"  OVERALL  acc={overall['acc']:.3f}  Brier={overall['brier']:.4f}  ECE={overall['ece']:.4f}")
    print("  calibration (predicted P(p1 wins) vs realized):")
    for b, (cnt, sp, so) in overall["buckets"].items():
        if cnt >= 20:
            print(f"    {b/10:.1f}-{b/10+0.1:.1f}: n={cnt:>4} pred={sp/cnt:.3f} realized={so/cnt:.3f}")
    by_surface = {}
    for surf in ("Hard", "Clay", "Grass"):
        mt = _metrics([p for p in pairs if p[2] == surf])
        if mt:
            by_surface[surf] = {k: mt[k] for k in ("n", "acc", "brier", "ece")}
            print(f"  {surf:6}: n={mt['n']:>4} acc={mt['acc']:.3f} Brier={mt['brier']:.4f} ECE={mt['ece']:.4f}")
    return {"overall": {k: overall[k] for k in ("n", "acc", "brier", "ece")},
            "by_surface": by_surface}
