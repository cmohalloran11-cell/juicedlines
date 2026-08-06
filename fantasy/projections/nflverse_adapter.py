"""
fantasy.projections.nflverse_adapter — the one ProjectionsProvider implemented in phase 1.

Methodology (deliberately simple and fully disclosed -- see JUICED's no-fabricated-numbers
rule): a player's projection is their empirical-Bayes-shrunk per-game rate from their most
recent completed regular season, projected across a flat assumed PROJECTED_GAMES-game season.
This is a historical-performance baseline, NOT a predictive model -- there is no injury/depth-
chart/scheme adjustment. `methodology` on every returned row says so explicitly; nothing in
this codebase may strip that field before it reaches a user. It exists so draft-board/VOR/
scoring work end-to-end before a licensed commercial provider (still TBD) replaces it -- see
the ProjectionsProvider interface in base.py.

Shrinkage: (observed_per_game*n + positional_prior_per_game*k) / (n+k), the same pattern used
in projector.regress_to_prior / basketball.fit_rates / tennis._shrink (see CLAUDE.md). The
prior for a stat+position is the mean per-game rate across every player at that position with
at least one game in the lookback window -- not a hardcoded constant.

Data source: the nflverse-data GitHub release ("player_stats" tag), confirmed live 2026-08 --
weekly per-player stats back to 1999, ~33MB, no auth. Fetched only by the daily sync job, never
on a request path (same constraint as Sleeper's players/nfl dump).
"""
from __future__ import annotations

import csv
import logging
import os
from collections import defaultdict
from typing import Iterable, Optional

import requests

from .base import ProjectionsProvider, CANONICAL_STAT_KEYS

log = logging.getLogger(__name__)

PLAYER_STATS_URL = os.getenv(
    "FANTASY_NFLVERSE_STATS_URL",
    "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats.csv",
)

LOOKBACK_SEASONS = 2       # seasons of history considered for the rate + the shrinkage prior
SHRINKAGE_K = 6.0          # games of "prior" weight -- roughly half weight by ~6 games played
PROJECTED_GAMES = 17       # flat assumed season length (a documented assumption, not a model)

# nflverse raw column -> our canonical key (fantasy.projections.CANONICAL_STAT_KEYS). Fumbles
# lost is split across three raw columns (sack/rushing/receiving) and summed; everything else
# is a 1:1 rename.
_RAW_TO_CANONICAL: dict[str, str] = {
    "passing_yards": "pass_yd", "passing_tds": "pass_td", "interceptions": "pass_int",
    "completions": "pass_cmp", "attempts": "pass_att", "passing_2pt_conversions": "pass_2pt",
    "rushing_yards": "rush_yd", "rushing_tds": "rush_td", "carries": "rush_att",
    "rushing_2pt_conversions": "rush_2pt",
    "receptions": "rec", "receiving_yards": "rec_yd", "receiving_tds": "rec_td",
    "receiving_2pt_conversions": "rec_2pt", "targets": "rec_tgt",
}
_FUMBLE_LOST_COLS = ("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost")
_FUMBLE_COLS = ("sack_fumbles", "rushing_fumbles", "receiving_fumbles")

OFFENSE_POSITIONS = {"QB", "RB", "WR", "TE"}


def _to_float(v) -> float:
    try:
        return float(v) if v not in (None, "") else 0.0
    except (ValueError, TypeError):
        return 0.0


def fetch_rows(url: str = PLAYER_STATS_URL, timeout: float = 120.0) -> Iterable[dict]:
    """Stream the nflverse player_stats release CSV and yield raw rows as dicts. Isolated from
    everything below so tests can feed synthetic rows without any HTTP."""
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        reader = csv.DictReader(line for line in resp.iter_lines(decode_unicode=True) if line)
        yield from reader


def _row_to_canonical_stats(row: dict) -> dict[str, float]:
    stats = {canon: _to_float(row.get(raw, 0)) for raw, canon in _RAW_TO_CANONICAL.items()}
    stats["fum_lost"] = sum(_to_float(row.get(c, 0)) for c in _FUMBLE_LOST_COLS)
    stats["fum"] = sum(_to_float(row.get(c, 0)) for c in _FUMBLE_COLS)
    return stats


def aggregate_seasons(rows: Iterable[dict], seasons: set[int]) -> dict[tuple[str, int], dict]:
    """Sum each offense player's raw stats + games-played within `seasons`, keyed by
    (gsis_id, season). Pure function of already-fetched rows -- the unit-testable core."""
    out: dict[tuple[str, int], dict] = {}
    for row in rows:
        if row.get("season_type") != "REG":
            continue
        try:
            season = int(row["season"])
        except (KeyError, ValueError, TypeError):
            continue
        if season not in seasons:
            continue
        position = (row.get("position") or "").upper()
        if position not in OFFENSE_POSITIONS:
            continue
        gsis_id = row.get("player_id")
        if not gsis_id:
            continue
        key = (gsis_id, season)
        agg = out.setdefault(key, {
            "gsis_id": gsis_id, "season": season, "position": position,
            "name": row.get("player_display_name") or row.get("player_name") or "",
            "team": row.get("recent_team"), "games": 0,
            "totals": defaultdict(float),
        })
        agg["games"] += 1
        agg["position"] = position  # last-seen position wins (a season's primary snap position)
        for k, v in _row_to_canonical_stats(row).items():
            agg["totals"][k] += v
    return out


def _positional_priors(per_player_season: dict[tuple[str, int], dict]) -> dict[tuple[str, str], float]:
    """Mean per-game rate for each (position, stat), across every player-season with >=1 game
    in the lookback window -- the shrinkage prior. Recomputed from real data every run, never
    hardcoded."""
    sums: dict[tuple[str, str], float] = defaultdict(float)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for agg in per_player_season.values():
        games = agg["games"]
        if games <= 0:
            continue
        for stat, total in agg["totals"].items():
            key = (agg["position"], stat)
            sums[key] += total / games
            counts[key] += 1
    return {k: sums[k] / counts[k] for k in sums if counts[k] > 0}


def project_players(per_player_season: dict[tuple[str, int], dict], target_season: int,
                    projected_games: int = PROJECTED_GAMES,
                    shrinkage_k: float = SHRINKAGE_K) -> list[dict]:
    """Shrink each player's most recent season rate toward the positional prior and project
    across `projected_games`. Pure function -- the actual methodology, independent of how the
    aggregates were produced, so it's unit-testable with synthetic aggregates."""
    priors = _positional_priors(per_player_season)
    latest_by_player: dict[str, tuple[str, int]] = {}
    for (gsis_id, season) in per_player_season:
        if season >= target_season:
            continue
        cur = latest_by_player.get(gsis_id)
        if cur is None or season > cur[1]:
            latest_by_player[gsis_id] = (gsis_id, season)

    out: list[dict] = []
    for gsis_id, key in latest_by_player.items():
        agg = per_player_season[key]
        n = agg["games"]
        position = agg["position"]
        stats: dict[str, float] = {}
        for stat_key in CANONICAL_STAT_KEYS:
            observed_rate = (agg["totals"].get(stat_key, 0.0) / n) if n > 0 else 0.0
            prior_rate = priors.get((position, stat_key), 0.0)
            shrunk = (observed_rate * n + prior_rate * shrinkage_k) / (n + shrinkage_k)
            stats[stat_key] = round(shrunk * projected_games, 2)
        out.append({
            "gsis_id": gsis_id,
            "name": agg["name"],
            "position": position,
            "team": agg["team"],
            "stats": stats,
            "provider": "nflverse",
            "methodology": (
                f"Empirical-Bayes-shrunk per-game rate from the {key[1]} season ({n} games) "
                f"x {projected_games} projected games. Historical-performance baseline, not a "
                f"predictive model."
            ),
        })
    return out


class NflverseProjectionsProvider(ProjectionsProvider):
    name = "nflverse"

    def get_projections(self, season: int, week: Optional[int] = None) -> list[dict]:
        """Season-long (week=None): the shrunk per-game rate x PROJECTED_GAMES, as documented
        on the module. Weekly (week=<int>): the SAME shrunk per-game rate, unmultiplied --
        this adapter has no matchup/injury/depth-chart signal to differentiate one week from
        another, so every week in a season gets an identical flat number. That's disclosed in
        `methodology`, not presented as week-granular modeling this adapter doesn't have."""
        seasons = {season - i for i in range(1, LOOKBACK_SEASONS + 1)}
        per_player_season = aggregate_seasons(fetch_rows(), seasons)
        games = 1 if week is not None else PROJECTED_GAMES
        players = project_players(per_player_season, target_season=season, projected_games=games)
        if week is not None:
            for p in players:
                p["methodology"] += (
                    " Applied flat to every week in the season -- no per-week matchup/injury "
                    "signal in this adapter.")

        # Resolve nflverse GSIS ids to our canonical player_id via the mapping layer. Players
        # with no mapping yet are skipped (not silently zero-scored) -- they'll surface once
        # fantasy.player_matching resolves them on the next sync, per the no-fabricated-
        # numbers rule: omit rather than mis-map.
        import store
        from fantasy.repositories import PlayerMappingRepository
        mapper = PlayerMappingRepository(store.get_database())
        resolved = []
        for p in players:
            player_id = mapper.resolve("nflverse_gsis", p["gsis_id"])
            if player_id is None:
                continue
            resolved.append({
                "player_id": player_id,
                "position": p["position"],
                "stats": p["stats"],
                "provider": p["provider"],
                "methodology": p["methodology"],
            })
        return resolved
