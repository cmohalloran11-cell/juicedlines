"""
Game environment — how big the game is and how it will be played.

Every number here comes from the `schedules` release's own closing spread_line/total_line
rather than from a model of implied totals. That is deliberate: the market's total already
aggregates everything an implied-total model would try to reconstruct (and more), so building
one would add error, not information. This module's only job is turning those two numbers into
expected team plays and an expected pass/rush split, using coefficients fitted to real games.

Fitted on 2,174 team-games (2022-2025 REG) joined to their schedule row:
    team_snaps     = 60.0921 + 0.17825 * team_spread + 0.11875 * total_line
    scrim_plays    = 58.045  + 0.1800  * team_spread + 0.0983  * total_line
    dropback_share = 0.4628  - 0.002641 * team_spread + 0.002373 * total_line
`team_spread` is positive when THIS team is favoured, so both signs are the expected game
script: a favourite runs more plays and throws less. The r values (0.12 and -0.15) are small,
which is why the effect is applied as a shift on a wide distribution rather than as a
confident point move — the simulator's snaps_sd_frac carries the residual spread.

`expected_team_snaps` and `expected_scrimmage_plays` are DIFFERENT quantities and the
difference matters: offensive snaps (65.3 a game) include penalties, kneels, spikes and
two-point tries that scrimmage plays (62.4) don't. Every per-snap opportunity rate in this
engine is fit against the snap_counts feed, so the simulator must use SNAPS; scrimmage plays
are carried alongside only because they're the more familiar "plays" number to display.

Weather comes from the same schedules row (temp/wind/roof), not from a separate weather API:
those fields are already the game's recorded conditions, and for a market-priced game the
spread/total have priced them in anyway. Measured on 1,190 outdoor games, only high wind is
large enough to act on (dropback share 0.5402 at >=15mph vs 0.5655 below it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..config import cfg
from ..data.base import ScheduleGame


@dataclass
class GameEnvironment:
    team: str
    opponent: Optional[str]
    season_type: str                    # "regular" | "preseason"
    game_id: Optional[str]
    spread: Optional[float]             # positive = THIS team favoured
    game_total: Optional[float]
    team_total: Optional[float]
    expected_team_snaps: float           # offensive snaps — what per-snap rates multiply
    expected_scrimmage_plays: float      # pass attempts + sacks + carries — for display
    expected_dropback_share: float
    roof: Optional[str] = None
    surface: Optional[str] = None
    temp: Optional[float] = None
    wind: Optional[float] = None
    weather: Optional[str] = None       # plain-language, None when genuinely unrecorded
    basis: str = "league_baseline"      # market_priced | league_baseline


def _weather_text(roof: Optional[str], temp: Optional[float], wind: Optional[float]) -> Optional[str]:
    if roof in ("dome", "closed"):
        return "indoors"
    bits = []
    if temp is not None:
        bits.append(f"{temp:.0f}F")
    if wind is not None:
        bits.append(f"{wind:.0f}mph wind")
    if bits:
        return ", ".join(bits)
    # A future game has a known roof but no recorded conditions yet — say which is known
    # rather than dropping the one real fact along with the missing ones.
    return roof or None


def game_environment(game: Optional[ScheduleGame], team: str,
                     season_type: str = "regular") -> GameEnvironment:
    """One team's view of one game. A game with no schedule row at all (every preseason game —
    the schedules release publishes none) or with an unpriced spread/total falls back to the
    measured league baseline and says so via `basis`, rather than pretending to a total."""
    e = cfg("environment", default={})
    snaps = float(e.get("league_snaps_mean", 65.3))
    plays = float(e.get("league_plays_mean", 62.4))
    dropback = float(e.get("league_dropback_share", 0.567))

    if game is None:
        return GameEnvironment(team=team, opponent=None, season_type=season_type, game_id=None,
                               spread=None, game_total=None, team_total=None,
                               expected_team_snaps=round(snaps, 2),
                               expected_scrimmage_plays=round(plays, 2),
                               expected_dropback_share=round(dropback, 4))

    opponent = game.opponent_of(team)
    is_home = game.home_team == team
    spread = None
    if game.spread_line is not None:
        spread = float(game.spread_line) if is_home else -float(game.spread_line)
    total = float(game.total_line) if game.total_line is not None else None
    team_total = round(total / 2.0 + spread / 2.0, 2) if (total is not None and spread is not None) else None

    basis = "league_baseline"
    if total is not None and spread is not None:
        snaps = (float(e["snaps_intercept"]) + float(e["snaps_per_spread"]) * spread
                 + float(e["snaps_per_total"]) * total)
        plays = (float(e["plays_intercept"]) + float(e["plays_per_spread"]) * spread
                 + float(e["plays_per_total"]) * total)
        dropback = (float(e["dropback_intercept"]) + float(e["dropback_per_spread"]) * spread
                    + float(e["dropback_per_total"]) * total)
        basis = "market_priced"

    if game.wind is not None and game.roof not in ("dome", "closed"):
        if float(game.wind) >= float(e.get("high_wind_mph", 15.0)):
            # Shift by the MEASURED high-wind gap rather than replacing the fitted value, so a
            # windy game keeps its spread/total-driven game script.
            dropback += float(e["high_wind_dropback_share"]) - float(e["normal_dropback_share"])

    return GameEnvironment(
        team=team, opponent=opponent, season_type=season_type, game_id=game.game_id,
        spread=spread, game_total=total, team_total=team_total,
        expected_team_snaps=round(max(38.0, min(95.0, snaps)), 2),
        expected_scrimmage_plays=round(max(35.0, min(90.0, plays)), 2),
        expected_dropback_share=round(max(0.25, min(0.85, dropback)), 4),
        roof=game.roof, surface=game.surface, temp=game.temp, wind=game.wind,
        weather=_weather_text(game.roof, game.temp, game.wind), basis=basis)


def measure_environment(weeks, snaps, schedule) -> dict:
    """Refit config.environment's coefficients from real data — the pipeline that produced
    them, kept runnable. A team's offensive snaps for a game are the MAX offense_snaps across
    its players (an offensive lineman is on the field for essentially all of them), which is
    the only way the per-player snap feed exposes the team total. Returns {} when there aren't
    enough priced team-games to fit (fewer than 200), rather than emitting coefficients off a
    handful of games."""
    import numpy as np
    from collections import defaultdict

    sched = {g.game_id: g for g in schedule}
    agg: dict[tuple, dict] = defaultdict(lambda: defaultdict(float))
    for w in weeks:
        d = agg[(w.game_id, w.team)]
        d["att"] += w.pass_attempts
        d["sack"] += w.sacks
        d["car"] += w.carries
    team_snaps: dict[tuple, float] = defaultdict(float)
    for s in snaps:
        key = (s.game_id, s.team)
        team_snaps[key] = max(team_snaps[key], float(s.offense_snaps or 0.0))

    X, snap_totals, plays, drop = [], [], [], []
    for (gid, team), d in agg.items():
        g = sched.get(gid)
        if g is None or g.spread_line is None or g.total_line is None:
            continue
        p = d["att"] + d["sack"] + d["car"]
        sn = team_snaps.get((gid, team), 0.0)
        if p < 30 or sn < 30:
            continue
        spread = float(g.spread_line) if g.home_team == team else -float(g.spread_line)
        X.append([1.0, spread, float(g.total_line)])
        snap_totals.append(sn)
        plays.append(p)
        drop.append((d["att"] + d["sack"]) / p)
    if len(X) < 200:
        return {}
    A = np.array(X)
    cs = np.linalg.lstsq(A, np.array(snap_totals), rcond=None)[0]
    cp = np.linalg.lstsq(A, np.array(plays), rcond=None)[0]
    cd = np.linalg.lstsq(A, np.array(drop), rcond=None)[0]
    resid_s = np.array(snap_totals) - A @ cs
    resid_d = np.array(drop) - A @ cd
    return {
        "n_team_games": len(X),
        "snaps_intercept": round(float(cs[0]), 4), "snaps_per_spread": round(float(cs[1]), 5),
        "snaps_per_total": round(float(cs[2]), 5),
        "plays_intercept": round(float(cp[0]), 4), "plays_per_spread": round(float(cp[1]), 5),
        "plays_per_total": round(float(cp[2]), 5),
        "dropback_intercept": round(float(cd[0]), 4),
        "dropback_per_spread": round(float(cd[1]), 6),
        "dropback_per_total": round(float(cd[2]), 6),
        "league_snaps_mean": round(float(np.mean(snap_totals)), 3),
        "league_plays_mean": round(float(np.mean(plays)), 3),
        "snaps_per_scrimmage_play": round(float(np.mean(snap_totals) / np.mean(plays)), 4),
        "snaps_sd_frac": round(float(resid_s.std() / np.mean(snap_totals)), 4),
        "dropback_sd": round(float(resid_d.std()), 4),
    }
