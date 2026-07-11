"""
Surface-weighted Elo backbone.

Processes matches chronologically, updating an overall Elo and a per-surface Elo
after each result. Used two ways: (1) an independent match-win estimate to blend
with / sanity-check the point model, and (2) a rating-tier prior to shrink thin-sample
players toward. ATP and WTA are rated separately.
"""

from __future__ import annotations

from collections import defaultdict

from ..config import cfg


class EloModel:
    def __init__(self, tour: str):
        self.tour = tour
        self.start = cfg("model", "elo_start")
        self.k = cfg("model", "elo_k")
        self.w = cfg("model", "elo_surface_weight")
        self.overall: dict[str, float] = defaultdict(lambda: self.start)
        self.surface: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(lambda: self.start))
        self.n: dict[str, int] = defaultdict(int)

    def fit(self, matches) -> "EloModel":
        wins = sorted((m for m in matches if m.won), key=lambda m: str(m.date))  # one row per match
        for m in wins:
            w, l, s = m.player_id, m.opp_id, m.surface
            if not w or not l:
                continue
            ew = 1.0 / (1 + 10 ** ((self.overall[l] - self.overall[w]) / 400))
            self.overall[w] += self.k * (1 - ew); self.overall[l] -= self.k * (1 - ew)
            if s:
                sw, sl = self.surface[s][w], self.surface[s][l]
                es = 1.0 / (1 + 10 ** ((sl - sw) / 400))
                self.surface[s][w] += self.k * (1 - es); self.surface[s][l] -= self.k * (1 - es)
            self.n[w] += 1; self.n[l] += 1
        return self

    def rating(self, pid: str, surface: str | None = None) -> float:
        ov = self.overall.get(pid, self.start)
        if surface and surface in self.surface and pid in self.surface[surface]:
            return self.w * self.surface[surface][pid] + (1 - self.w) * ov
        return ov

    def win_prob(self, a: str, b: str, surface: str | None = None) -> float:
        ra, rb = self.rating(a, surface), self.rating(b, surface)
        return 1.0 / (1 + 10 ** ((rb - ra) / 400))
