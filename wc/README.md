# WC Prop Model (inside JUICED)

A World Cup 2026 **player-prop projection engine** for single-stat markets —
anytime goalscorer, shots on target, cards/bookings, GK saves — with a value
finder that flags mismatches against sportsbook odds. It lives **inside JUICED**
(the sports-edge FastAPI app), surfaced through the **WC Model** tab and
`/api/wc/*` endpoints — not a separate tool.

## Pipeline

1. **Team goal expectancy** (`model/`) — Dixon-Coles: each side's expected goals =
   `league_avg × attack × opponent_defense`, attack/defense blended from goals + xG
   and normalized to the tournament baseline; a home-advantage bump applies only
   when a host nation actually plays at home (WC is otherwise neutral). A score
   matrix carries the Dixon-Coles low-score correction ρ.
2. **Match simulation** (`sim/`) — 10,000 Monte-Carlo sims per fixture, sampling
   exact scorelines from the DC matrix, plus a per-sim tempo (total goals vs
   reference) for volume scaling.
3. **Player allocation** —
   - *Goals*: `Binomial(team_goals, xg_share × minutes)` — share of xG, not raw minutes.
   - *Shots on target*: `Poisson(sot90 × minutes × tempo)`.
   - *Cards*: `Poisson((yellow+red)/90 × minutes × intensity)`, intensity bumped for
     knockout and rivalry games (configurable).
   - *GK saves*: shots-faced `Poisson(opp_shots × on-target-rate × tempo)`, then
     `Binomial(shots_faced, save%)`.
4. **Minutes / rotation** — projections scale by a minutes-fraction from each
   player's start probability, and every row carries a **confidence** flag
   (`confirmed` / `probable` / `rotation risk`).
5. **Value finder** (`value/`) — American odds → implied probability; rank props by
   `edge = model − implied` and `EV = model × decimal − 1`.

## Data sources — swappable adapters

Everything the model sees is a dataclass in `data/base.py`, produced by an adapter.
Pick each source in **`config.yaml`** (`sources:` → `sample` | `csv` | `api`):

| Source | Sample | CSV (`data/files/*.csv`) | API (stub — you implement) |
|---|---|---|---|
| Fixtures + lineups | ✅ built-in | `fixtures.csv` | API-Football (`keys.api_football`) |
| Team strength / xG | ✅ | `strength.csv` | Understat / FBref / API-Football |
| Player history | ✅ | `players.csv` | API-Football player stats |
| Odds | ✅ | `odds.csv` | The Odds API (`keys.odds_api`) |

**To wire real data:** implement the four classes in `data/api.py` (map your
provider's JSON into the `base.py` dataclasses), set your keys via env
(`API_FOOTBALL_KEY`, `ODDS_API_KEY`) or `config.yaml`, and flip the matching
`sources:` entry to `api`. Nothing else changes. For quick offline testing, drop
CSVs in `data/files/` (headers documented in `data/csv.py`) and use `csv`.

## Tuning

Model weights live in **`config.yaml`** (no code changes): `n_sims`,
`league_avg_goals`, `home_advantage`, `recency_decay`, `dc_rho`,
`tempo_ref_goals`, and the `intensity` (knockout/rivalry) + `confidence`
thresholds.

## Run it

```bash
# in JUICED: open the "WC Model" tab, or hit the API
GET /api/wc/matches                 # fixtures
GET /api/wc?match=wc-qf1            # projections + value (all fixtures if no match)
GET /api/wc/export?fmt=csv|json     # export

# standalone
python -c "from wc import projections; print(projections.to_csv(projections.run()))"

# tests
python wc/tests/test_sim.py && python wc/tests/test_value.py    # or: pytest wc/tests
```

## Assumptions & limitations (read before betting)

- **Small player samples.** Per-90 rates over a 12-month + group-stage window are
  noisy for players with few minutes; treat low-minute players' props with care
  (the confidence flag helps, but doesn't fix the variance).
- **No live in-match adjustment.** Projections are pre-match only.
- **No injury/news feed.** Rotation is modeled only via a start-probability input;
  a late scratch or a surprise XI isn't captured until the lineup source updates.
- **Goals allocation ≈ xG share**, which rewards volume shooters/penalty takers but
  can't know set-piece roles or in-game tactical shifts.
- **GK saves are modeled independently of the simulated goals conceded** (per the
  spec's save-% approach), so saves + conceded aren't forced to reconcile.
- **The value finder only prices lines the model outputs** (SoT/saves at the modeled
  half-point lines, anytime goal/card). Odds on other lines are skipped rather than
  guessed.
- **Sample data is illustrative**, not real form — the numbers are only as good as
  the sources you plug in.
