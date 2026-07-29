"""
Surface-weighted Elo backbone.

Processes matches chronologically, updating an overall Elo and a per-surface Elo
after each result. Used two ways: (1) an independent match-win estimate to blend
with / sanity-check the point model, and (2) a rating-tier prior to shrink thin-sample
players toward. ATP and WTA are rated separately.

Keyed by NORMALIZED PLAYER NAME, not the historical source's player_id — the Sackmann
mirror (frozen at 2022) and the live ESPN results (`data/espn.py`) use unrelated id
schemes, and a name is the only link between them. This lets `fit()` be called twice —
once on the historical mirror, once on recent ESPN results — and have a returning
player's live results actually extend the SAME rating rather than starting a disconnected
one. See projections.py's `_model()` for the two-stage fit + rank-prior seeding.
"""

from __future__ import annotations

import math
import unicodedata
from collections import defaultdict

from ..config import cfg


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace(".", "").replace("'", "").split())


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
        """Apply match results chronologically. `matches` need only `.won`, `.player`,
        `.opp`, `.surface`, `.date` (one winner-perspective row per match is enough —
        `.opp` carries the loser). Composable: call again with a later window of
        results (e.g. live ESPN matches) to extend ratings already fit on history."""
        wins = sorted((m for m in matches if m.won), key=lambda m: str(m.date))
        for m in wins:
            w, l, s = _norm(m.player), _norm(m.opp), m.surface
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

    def seed_from_rank(self, ranks: dict[str, int]) -> "EloModel":
        """Cold-start prior for a player with NO fitted rating yet (never appeared in the
        historical mirror or in live results) but a real current tour ranking — e.g. the
        Sackmann mirror predates Sinner/Alcaraz's careers entirely, so without this they'd
        sit at a flat, wrong "average tour player" 1500 until enough live results accrue.
        Rating ~ 2200 - 130*log10(rank), roughly matching real ATP/WTA Elo-vs-rank spread
        (rank 1 ≈ 2200, rank 10 ≈ 1970, rank 100 ≈ 1740). Never overwrites an existing
        (historical or live-updated) rating — this is a PRIOR, not a correction."""
        for name, rank in ranks.items():
            if name in self.overall or not rank or rank < 1:
                continue
            self.overall[name] = 2200.0 - 130.0 * math.log10(max(1, rank))

    def rating(self, pid: str, surface: str | None = None) -> float:
        pid = _norm(pid)
        ov = self.overall.get(pid, self.start)
        if surface and surface in self.surface and pid in self.surface[surface]:
            return self.w * self.surface[surface][pid] + (1 - self.w) * ov
        return ov

    def win_prob(self, a: str, b: str, surface: str | None = None) -> float:
        ra, rb = self.rating(a, surface), self.rating(b, surface)
        return 1.0 / (1 + 10 ** ((rb - ra) / 400))

    def eff_matches(self, name: str) -> int:
        return self.n.get(_norm(name), 0)
