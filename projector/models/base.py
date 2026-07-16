"""
base.py — shared projection data types and distribution summarisation.

A projection is always a *distribution* (from Monte Carlo samples), summarised
into mean/median/quartiles/floor/ceiling plus over-threshold probabilities and
the human-readable factors that drove it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np


@dataclass
class Projection:
    stat: str
    mean: float
    median: float
    p25: float
    p75: float
    floor: float            # 10th pct
    ceiling: float          # 90th pct
    std: float
    thresholds: dict[float, float] = field(default_factory=dict)  # line -> P(>= line is cleared)
    drivers: list[str] = field(default_factory=list)
    samples: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self, with_samples: bool = False) -> dict[str, Any]:
        d = asdict(self)
        d.pop("samples", None)
        if with_samples and self.samples is not None:
            d["samples"] = self.samples.tolist()
        return d

    def prob_over(self, line: float) -> float:
        """P(stat > line) — half-point lines have no push."""
        if self.samples is None:
            return float("nan")
        return float(np.mean(self.samples > line))


def summarize(samples: np.ndarray, stat: str, drivers: list[str] | None = None,
              lines: list[float] | None = None) -> Projection:
    """Turn Monte-Carlo samples into a Projection with standard percentiles."""
    s = np.asarray(samples, dtype=float)
    s = s[~np.isnan(s)]
    if s.size == 0:
        s = np.zeros(1)
    p = np.percentile(s, [10, 25, 50, 75, 90])
    proj = Projection(
        stat=stat,
        mean=round(float(s.mean()), 3),
        median=round(float(p[2]), 3),
        p25=round(float(p[1]), 3),
        p75=round(float(p[3]), 3),
        floor=round(float(p[0]), 3),
        ceiling=round(float(p[4]), 3),
        std=round(float(s.std()), 3),
        drivers=drivers or [],
        samples=s,
    )
    # default over/under thresholds around the mean (e.g. common prop lines)
    if lines is None:
        base = max(0.5, round(proj.mean * 2) / 2)
        lines = sorted({round(base + d, 1) for d in (-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5)
                        if base + d > 0})
    proj.thresholds = {float(l): round(float(np.mean(s > l)), 3) for l in lines}
    return proj


def blend_rate(recent: float, season: float, career: float, w: dict) -> float:
    """Recency-weighted blend of three rate estimates."""
    return (w["recent_weight"] * _f(recent)
            + w["season_weight"] * _f(season)
            + w["career_weight"] * _f(career))


def regress_to_prior(observed: float, prior: float, n: float, pseudo: float) -> float:
    """
    Bayesian shrinkage toward a predictive-metric prior (e.g. xwOBA-implied rate).
    observed: empirical rate; prior: model expectation; n: sample (PA / 90s);
    pseudo: strength of the prior (pseudo-observations).
    """
    n = max(0.0, _f(n))
    return (n * _f(observed) + pseudo * _f(prior)) / (n + pseudo) if (n + pseudo) else _f(prior)


def _f(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default
