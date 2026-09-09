"""
Team-level reconciliation — does the board's set of per-player projections still add up?

usage.py fits and shrinks every player independently and sim/engine.py simulates every player
independently, so nothing anywhere in the pipeline knows that the players on one team compete
for the SAME finite pool of pass attempts and carries that environment.py has already sized.
Five receivers can each be projected for a perfectly defensible 8 targets on a team whose
environment supports ~33 pass attempts, and no individual projection looks wrong from the
inside.

This module MEASURES and REPORTS that gap. It deliberately does not redistribute, clamp or
rewrite anything: which player should lose volume is a judgement no data currently in this
engine supports, and quietly editing a projection to satisfy an accounting identity would hide
the integrity failure rather than surface it.

The team side of the comparison is environment.py's own output, converted from plays to
attempts:
    dropbacks     = expected_scrimmage_plays * expected_dropback_share
    pass attempts = dropbacks * (1 - league sack rate)
    rush attempts = expected_scrimmage_plays * (1 - expected_dropback_share)
The sack rate is config.defense.league.sack_rate — MEASURED 0.0690 over 128 team-seasons — and
is needed because dropback_share is a share of SCRIMMAGE plays, which counts sacks, while a
target and a pass attempt do not.

The check is ONE-SIDED. A board only carries the players a book happened to post a prop for, so
the summed projections normally sit well below the team total; an undershoot carries no
information about anything. Only an overshoot is evidence.

The tolerance is the engine's OWN stated uncertainty about team volume rather than a chosen
number: config.sim.snaps_sd_frac (0.1342) and config.sim.dropback_sd (0.1047) are the measured
residual spreads the simulator already draws team plays and the pass/rush split from, so one
standard deviation of a team's attempt pool is
    pass attempts  sqrt(0.1342^2 + (0.1047/0.5673)^2) = 22.8%
    rush attempts  sqrt(0.1342^2 + (0.1047/0.4327)^2) = 27.7%
A violation is therefore "the summed player projections exceed the team's expected volume even
after giving the team a full standard deviation of its own measured game-to-game spread".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

from ..config import cfg

# per-player quantity -> (team attempt pool it is drawn from, projection field it is read off)
_QUANTITIES = {
    "targets": ("pass_attempts", "expected_targets"),
    "pass_attempts": ("pass_attempts", "expected_pass_attempts"),
    "carries": ("rush_attempts", "expected_carries"),
}

# Standard deviations of the team pool the tolerance is expressed in. One, for the reason in
# the module docstring; raising it would only make the check blinder to the failure it exists
# to catch.
TOLERANCE_SDS = 1.0


@dataclass
class VolumeViolation:
    """One team-game whose summed player projections outrun the team volume that produced
    them. Reported, never corrected — see the module docstring."""
    team: str
    game_id: Optional[str]
    quantity: str                       # targets | pass_attempts | carries
    projected_total: float
    team_expected: float
    tolerance_high: float
    n_players: int
    contributors: list = field(default_factory=list)   # (player, position, value), largest first
    environment_basis: str = "league_baseline"

    @property
    def overshoot(self) -> float:
        return self.projected_total - self.team_expected

    def message(self) -> str:
        top = ", ".join(f"{p} ({pos}) {v:.1f}" for p, pos, v in self.contributors[:5])
        where = f" {self.game_id}" if self.game_id else ""
        return (f"team volume violation: {self.team}{where} — {self.n_players} projected "
                f"players sum to {self.projected_total:.1f} {self.quantity} against a team "
                f"expectation of {self.team_expected:.1f} (flags above {self.tolerance_high:.1f}, "
                f"environment {self.environment_basis}); largest: {top}")


def team_attempt_pool(environment) -> dict[str, float]:
    """The team's expected attempts for one game, from its own GameEnvironment."""
    plays = float(environment.expected_scrimmage_plays)
    dropback_share = float(environment.expected_dropback_share)
    sack_rate = float(cfg("defense", "league", "sack_rate", default=0.0690))
    dropbacks = plays * dropback_share
    return {
        "dropbacks": dropbacks,
        "pass_attempts": dropbacks * (1.0 - sack_rate),
        "rush_attempts": plays * (1.0 - dropback_share),
    }


def pool_relative_sd(pool: str) -> float:
    """One standard deviation of a team attempt pool, as a fraction of its mean, propagated
    from the two measured residual spreads the simulator itself draws team volume from."""
    plays_cv = float(cfg("sim", "snaps_sd_frac", default=0.1342))
    dropback_sd = float(cfg("sim", "dropback_sd", default=0.1047))
    league_share = float(cfg("environment", "league_dropback_share", default=0.5673))
    share = league_share if pool == "pass_attempts" else 1.0 - league_share
    return math.sqrt(plays_cv ** 2 + (dropback_sd / share) ** 2)


def reconcile_team(projections: list[dict], environment) -> list[VolumeViolation]:
    """Violations for one team in one game. `projections` are project_player() outputs for
    every player of that team currently on the board; `environment` is that team's own
    GameEnvironment (any of the projections' `environment` will do — they are the same object
    for the same team-game)."""
    if not projections:
        return []
    pool = team_attempt_pool(environment)
    team = projections[0].get("team") or ""
    game_id = getattr(environment, "game_id", None)

    out: list[VolumeViolation] = []
    for quantity, (pool_key, proj_field) in _QUANTITIES.items():
        contributors = sorted(
            ((p.get("player") or "", p.get("position") or "", float(p.get(proj_field) or 0.0))
             for p in projections),
            key=lambda c: c[2], reverse=True)
        total = sum(c[2] for c in contributors)
        expected = pool[pool_key]
        high = expected * (1.0 + TOLERANCE_SDS * pool_relative_sd(pool_key))
        if total > high:
            out.append(VolumeViolation(
                team=team, game_id=game_id, quantity=quantity,
                projected_total=round(total, 2), team_expected=round(expected, 2),
                tolerance_high=round(high, 2), n_players=len(projections),
                contributors=[c for c in contributors if c[2] > 0.05],
                environment_basis=getattr(environment, "basis", "league_baseline")))
    return out


def reconcile_board(projections: Iterable[dict]) -> list[VolumeViolation]:
    """Every violation across a whole board's worth of projections, grouped by team-game. A
    player resolved twice (the same name reaching the board under two team spellings) is
    counted once, so a duplicate can never manufacture a violation on its own."""
    groups: dict[tuple, list[dict]] = {}
    seen: set[tuple] = set()
    for p in projections:
        if not p or not p.get("team"):
            continue
        env = p.get("environment")
        if env is None:
            continue
        key = (p["team"], getattr(env, "game_id", None))
        dedupe = key + ((p.get("player") or "").strip().lower(),)
        if dedupe in seen:
            continue
        seen.add(dedupe)
        groups.setdefault(key, []).append(p)

    out: list[VolumeViolation] = []
    for members in groups.values():
        out.extend(reconcile_team(members, members[0]["environment"]))
    return out
