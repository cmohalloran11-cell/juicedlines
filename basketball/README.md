# Basketball Projection System (WNBA) — inside JUICED

One shared projection **core** — per-possession rates × projected minutes ×
projected pace, opponent-adjusted, simulated to a **distribution per stat** — driven by
a league config. WNBA is a re-baselining job on stable data (tight intervals,
validated). Attaches projections to the live PrizePicks/Underdog board.

## The shared core

1. **Rates per possession** (`model/rates.py`) — for each base stat (pts, reb, ast,
   stl, blk, 3pm, to) a per-possession rate, recency-weighted, then **regressed toward
   a league prior** with pseudo-possessions (shrinkage). The prior + shrinkage strength
   are the league hooks.
2. **Prior** (`model/priors.py`) — positional per-40 averages.
3. **Minutes** (`model/minutes.py`) — modelled as its **own** component (news-injectable),
   because it swings counting stats most.
4. **Pace** (`model/pace.py`) — possessions/48-equiv for the matchup (league baseline on
   the board; team pace for backtest).
5. **Opponent** (`model/opponent.py`) — per-stat defensive multiplier (neutral hook v1).
6. **Simulate** (`sim/engine.py`) — draw minutes + pace once per sim, then each stat as an
   overdispersed count (Negative-Binomial). Combos (PRA, stocks…) are summed **within** a
   sim so they stay correlated. Yields a distribution → O/U probability + variance.
7. Every projection carries a **confidence** (high/medium/low) from effective sample size
   + shrinkage, so the app can gate which markets to surface.

## Config A — WNBA (validated)

Re-baselined, not re-architected: WNBA pace baseline, tighter 10–11-deep rotations,
positional priors, healthy samples → light shrinkage → tight intervals.

**Backtest** (`backtest/calibration.py`, date-strict walk-forward, 518 held-out games):

| Stat | Bias | MAE | ECE |
|---|---|---|---|
| Points | +0.05 | 4.18 | 0.013 |
| Rebounds | +0.04 | 1.71 | 0.018 |
| Assists | +0.06 | 1.26 | 0.022 |

Essentially unbiased and well-calibrated (predicted P(over) ≈ realized) — this is the
league that validates the shared core.

## Data — swappable adapters (`data/`)

The model only sees `data/base.py` dataclasses. **ESPN** (`data/espn.py`, free, no key) is
the live source for both leagues: rosters, per-game logs (WNBA gamelog endpoint; SL from
box scores), and team pace from box-score possession components.

**Summer-League background** (`data/background.py`) — draft slot + pre-NBA league +
translated per-40 rates, keyed per player. A local `sl_background.json` seed is the
reliable injection point; Bart Torvik (college, `getadvstats.php` confirmed reachable)
and RealGM (international) are the wire-up targets. Seed schema:

```json
[{"player": "Full Name", "draft_pick": 3, "pre_league": "NCAA",
  "archetype": "ball-handler", "rates40": {"pts": 22, "reb": 5, "ast": 6,
  "stl": 1.4, "blk": 0.3, "3pm": 2.5, "to": 3.0}}]
```

Anything unseeded falls back to the generic SL positional prior (wide, low confidence).

## Board integration

`board.py::attach_basketball(lines)` groups live WNBA/SL prop lines by (league, player),
projects once per player, and writes `model_proj` / `model_prob` / `model_edge` /
`bball_confidence` onto every modelled market (points, rebounds, assists, threes, steals,
blocks, turnovers, PRA/PR/PA/RA, stocks, fantasy). Period/quarter markets are skipped.

## Assumptions & limitations

- **Opponent defense is a neutral hook on the board** (positional rate-allowed needs a
  richer feed than free ESPN); pace uses the league baseline on the board (team/matchup
  pace is used by the backtest). Both are documented refinement hooks, not silent guesses.
- No live in-game / injury modelling; pre-game only.
