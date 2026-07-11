# Tennis Projection System (ATP + WTA) — inside JUICED

A serve/return point model wrapped in a Monte-Carlo match simulator, with a
surface-weighted Elo backbone. 1v1 — **players and matches, no teams**. Produces
distributions for every prop market and attaches projections to the **live
PrizePicks/Underdog tennis lines** on the JUICED board.

## Pipeline

1. **Serve/return rates** (`model/rates.py`) — per player, `spw` (serve points won)
   and `rpw` (return points won), overall + per surface, regressed toward the tour
   baseline via pseudo-counts (thin samples shrink harder). Surface rates fall back
   to `overall + surface_shift`. Ace/DF rates and points-per-service-game too. **ATP
   and WTA are fit separately — baselines are never shared.**
2. **Matchup** (`model/matchup.py`) — `p_serve(A vs B) = spw_avg + (spw_A − spw_avg)
   − (rpw_B − rpw_avg)`, clamped 0.50–0.85. Closed-form game/tiebreak/set/match
   probabilities validate the simulator and feed match-win prob.
3. **Elo backbone** (`model/elo.py`) — overall + surface Elo, blended, updated per
   match. Independent match-win estimate blended with the point model; also a tier
   prior for thin players.
4. **Simulation** (`sim/engine.py`) — vectorized game-level Monte-Carlo (N≥10k,
   ~50ms/match): holds via closed-form hold prob, tiebreaks via tiebreak prob;
   serve points → aces/DFs as Binomials. Every count prop falls out of the
   distributions. Format (best-of-3/5, final-set rule) is a per-match parameter.
5. **Projections** (`projections.py`) — fits/caches per tour, projects a match,
   reads any market. Each carries a **confidence** (`high`/`medium`/`low`) from
   effective sample size + shrinkage, so thin-sample / qualifier matches surface
   with wide intervals and can be gated.
6. **Board** (`board.py`) — groups live tennis prop lines into matches and attaches
   `model_proj` / `model_prob` / confidence to the markets we model (games played,
   games won, sets played/won, aces, double faults, breakpoints).

## Data — swappable adapters (`data/`)

The model only sees the dataclasses in `data/base.py`. Select sources in `config.py`.

| Source | Status | Notes |
|---|---|---|
| `sackmann` (mirror) | **built** | Real serve-stat CSVs, historical (~2016–22). ⚠ CC **non-commercial** — build/validate only. |
| `espn` | stub | Free live scoreboard (surface, today's slate, current form). Wire to fill the freshness gap. |
| `licensed` | stub | Production feed (Matchstat/Tennis-API, Sportradar). Implement `data/*` against it, set `sources` to `licensed`. |

**To go production:** implement the licensed adapter (map its JSON into `base.py`
dataclasses), set the key (`TENNIS_FEED_KEY`), flip `sources`. Nothing else changes.

## Validation

`python -c "from tennis.backtest import calibration as c; c.run('ATP',[2015,2016,2017],[2018])"`
— date-strict (no leakage), reports accuracy / Brier / ECE overall and **by surface**.
2018 result: 61.6% match accuracy; the pure point model is **overconfident at the
extremes** (predicts 84% → realized 69%), worse on clay — which is why the Elo blend
tempers it. Tune `match_prob_blend` against this.

## Assumptions & limitations (read before trusting output)

- **Mirror data is stale (~2016–22) and non-commercial.** Current players (Sinner,
  Alcaraz) are thin → correctly flagged **low confidence**. Real freshness needs the
  ESPN live layer or a licensed feed.
- **Surface defaults to Hard on the board** until ESPN surface detection is wired —
  it only shifts serve rates ~±0.03, but on grass/clay slates expect a systematic
  bias (the market check flags it). **Top refinement.**
- **No live in-match / injury / retirement modelling.** Pre-match only.
- **Per-set (period_1_*) and tie-breaks-played markets aren't modelled yet** — those
  lines are skipped rather than guessed.
- **Point model assumes iid points** → overconfident on lopsided matches; the Elo
  blend and shrinkage compress this but don't eliminate it.
