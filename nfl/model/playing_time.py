"""
Playing-Time Engine — a SAMPLED DISTRIBUTION, not a number.

Everything downstream is opportunity-per-snap times snaps, so the snap share is the single
largest source of uncertainty in an NFL projection and the one place a point estimate does
the most damage: a back splitting a backfield 55/45 and a bell-cow can share a projected
snap share while having completely different floors.

Three levels of evidence, each with an honestly lower confidence than the one above:
  1. the player's OWN recent snap shares, recency-weighted and shrunk toward the positional
     prior (half-life and k both measured by holdout — see config.snap_share),
  2. no usable snap history but a known depth-chart rank -> the measured snap-share
     distribution for that (position, rank) from config.depth_rank_snap_share,
  3. neither -> the flat positional prior at its widest, role "unknown".

The distribution is Beta, because a snap share is a bounded proportion, with its
CONCENTRATION scaled by how much real evidence exists. That concentration is the two-stage
uncertainty for playing time: the simulator draws a snap share per trial from it before
drawing any outcome, so a player we barely know carries a genuinely wider band at the same
point estimate rather than the same band as an established starter.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import cfg


@dataclass
class PlayingTime:
    expected_snap_share: float
    expected_snaps: Optional[float]          # None when team plays are unknown
    concentration: float                     # Beta a+b — the evidence behind the share
    n_games: int
    basis: str                               # own_snap_history | depth_chart | positional_prior
    role: str                                # starter | rotation | backup | fringe | unknown
    depth_rank: Optional[int]
    playing_time_probability: Optional[float]
    confidence: str                          # high | medium | low

    def beta_params(self) -> tuple[float, float]:
        m = min(max(self.expected_snap_share, 1e-3), 1 - 1e-3)
        c = max(self.concentration, 0.2)
        return m * c, (1.0 - m) * c

    def snap_share_range(self, lo: float = 10.0, hi: float = 90.0) -> tuple[float, float]:
        """The 80% interval of the sampled snap share, from the Beta itself rather than from a
        simulation — deterministic, so the shipped range never wobbles run to run."""
        from scipy.stats import beta as _beta
        a, b = self.beta_params()
        return float(_beta.ppf(lo / 100.0, a, b)), float(_beta.ppf(hi / 100.0, a, b))


def _role_from_share(share: float, depth_rank: Optional[int]) -> str:
    """Role label from the projected snap share, with the depth chart breaking ties. The
    thresholds are the midpoints between the MEASURED rank-1/rank-2/rank-3 snap-share means
    in config.depth_rank_snap_share (pooled across positions: 0.67 / 0.34 / 0.21), i.e. they
    describe where a real rank-1 workload stops looking like a rank-2 one."""
    if share >= 0.60:
        return "starter"
    if share >= 0.35:
        return "rotation"
    if share >= 0.12:
        return "backup"
    if depth_rank is None and share <= 0.0:
        return "unknown"
    return "fringe"


def depth_rank_prior(position: str, depth_rank: Optional[int]) -> Optional[tuple[float, float, int]]:
    """(mean, sd, n) measured snap share for a (position, depth rank). Ranks 3+ are pooled —
    see the config comment for why a raw rank-3 mean is survivorship, not depth."""
    table = cfg("depth_rank_snap_share", default={}).get((position or "").upper())
    if not table or depth_rank is None:
        return None
    return table.get(min(int(depth_rank), 3))


def _concentration_from_moments(mean: float, sd: float) -> float:
    """Method-of-moments Beta concentration for a measured (mean, sd)."""
    if not (0.0 < mean < 1.0) or sd <= 0:
        return 1.0
    return max(0.2, mean * (1.0 - mean) / (sd * sd) - 1.0)


def project_playing_time(snap_sample: list[float], position: str,
                         depth_rank: Optional[int] = None,
                         expected_team_snaps: Optional[float] = None,
                         active_games: Optional[int] = None,
                         team_games: Optional[int] = None) -> PlayingTime:
    """`snap_sample` is the player's offense_pct per game, MOST-RECENT-FIRST.

    `active_games`/`team_games` drive playing_time_probability: the share of the team's recent
    games in which the player actually recorded an offensive snap. Both None (no snap feed at
    all) leaves it None rather than assuming the player is available — that unknown is what
    the confidence flag and the widened band are for.
    """
    pos = (position or "").upper()
    prior = cfg("snap_share", "prior", default={}).get(pos, 0.5)
    hl = cfg("snap_share", "halflife", default={}).get(pos, 1.5)
    k = cfg("snap_share", "shrink_games", default={}).get(pos, 0.5)
    full_conc = cfg("snap_share", "concentration", default={}).get(pos, 7.0)
    full_games = float(cfg("snap_share", "concentration_full_games", default=8))

    dr = depth_rank_prior(pos, depth_rank)

    if snap_sample:
        wsum = wval = 0.0
        for i, pct in enumerate(snap_sample):
            w = 0.5 ** (i / hl) if hl > 0 else 1.0
            wsum += w
            wval += w * pct
        # A known depth-chart rank is a strictly better shrinkage target than the flat
        # positional prior, and it's measured off the same joined data.
        target = dr[0] if dr else prior
        share = (wval + k * target) / (wsum + k) if (wsum + k) > 0 else target
        n = len(snap_sample)
        # Evidence ramps from the tier/positional prior's own measured spread up to the
        # player's measured within-season concentration once they have a real sample.
        base_conc = _concentration_from_moments(*dr[:2]) if dr else 1.0
        w_own = min(1.0, n / full_games)
        conc = (1.0 - w_own) * base_conc + w_own * full_conc
        basis, confidence = "own_snap_history", ("high" if n >= 6 else "medium" if n >= 3 else "low")
    elif dr:
        share, conc, n = dr[0], _concentration_from_moments(dr[0], dr[1]), 0
        basis, confidence = "depth_chart", "low"
    else:
        share, conc, n = prior, 1.0, 0
        basis, confidence = "positional_prior", "low"

    prob = None
    if team_games:
        # Shrunk the same way every other rate here is; k=2 games of prior weight keeps one
        # missed game from reading as 0.5. Capped at 1 because the two counts come from
        # different joins (a player row can survive a team game the snap feed dropped).
        played = min(float(active_games or 0), float(team_games))
        prob = round((played + 2.0 * 0.9) / (float(team_games) + 2.0), 3)

    role = _role_from_share(share, depth_rank) if basis != "positional_prior" else "unknown"
    expected_snaps = round(share * expected_team_snaps, 1) if expected_team_snaps else None
    return PlayingTime(
        expected_snap_share=round(float(share), 4), expected_snaps=expected_snaps,
        concentration=round(float(conc), 3), n_games=n, basis=basis, role=role,
        depth_rank=depth_rank, playing_time_probability=prob, confidence=confidence)


def sample_snap_share(pt: PlayingTime, n: int, rng: np.random.Generator) -> np.ndarray:
    """Per-trial snap share — the parameter-uncertainty draw playing time contributes to the
    two-stage simulation. Mean is E[Beta] = the point estimate regardless of concentration, so
    widening the band for a thin sample never moves the projection."""
    a, b = pt.beta_params()
    return rng.beta(a, b, n)


def measure_snap_prior(snaps, positions=("QB", "RB", "WR", "TE")) -> dict[str, tuple[float, int]]:
    """{position: (mean offense_pct, n)} straight off real snap rows — the pipeline that
    produced config.snap_share.prior, kept runnable so the table can be refreshed rather than
    trusted forever."""
    acc: dict[str, list] = {p: [] for p in positions}
    for s in snaps:
        p = (s.position or "").upper()
        if p in acc and s.offense_pct is not None:
            acc[p].append(float(s.offense_pct))
    return {p: (round(float(np.mean(v)), 4), len(v)) for p, v in acc.items() if v}


def measure_depth_rank_shares(snap_rows, depth_entries, season_type: str = "REG",
                              min_n: int = 30) -> dict[str, dict[int, tuple]]:
    """{position: {rank: (mean, sd, n)}} measured off real snap rows joined to the depth
    charts — the pipeline that produced config.depth_rank_snap_share, kept runnable so that
    table can be refreshed rather than trusted forever. Ranks 3+ are pooled into rank 3, the
    same way depth_rank_prior reads them (see the config block's own comment on why a raw
    rank-3 mean is survivorship, not depth). A (position, rank) bucket with fewer than
    `min_n` rows is omitted rather than published at a sample too thin to mean anything."""
    rank: dict[tuple, int] = {}
    for d in depth_entries:
        if d.rank is None:
            continue
        key = (d.team, d.player.strip().lower())
        if key not in rank or d.rank < rank[key]:
            rank[key] = d.rank
    buckets: dict[tuple, list] = defaultdict(list)
    for s in snap_rows:
        if s.season_type != season_type or s.offense_pct is None:
            continue
        r = rank.get((s.team, s.player.strip().lower()))
        if r is None:
            continue
        buckets[((s.position or "").upper(), min(int(r), 3))].append(float(s.offense_pct))
    out: dict[str, dict[int, tuple]] = defaultdict(dict)
    for (pos, r), v in buckets.items():
        if len(v) >= min_n:
            out[pos][r] = (round(float(np.mean(v)), 4), round(float(np.std(v)), 4), len(v))
    return dict(out)
