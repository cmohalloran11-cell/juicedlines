"""
Usage model — OPPORTUNITY and EFFICIENCY fitted separately, never blended.

A receiving-yards projection is `expected_targets x yards_per_target`, not an average of
past receiving yards, because those two components move for completely different reasons: a
target share is a role (it survives a bad game), a yards-per-target is a skill plus a lot of
noise (it does not). Averaging the product throws away the fact that we know the role far
better than the efficiency, and it makes a playing-time change impossible to apply.

Opportunity is expressed PER OFFENSIVE SNAP so playing_time.py can scale it without the two
models having to agree about anything else.

Every rate is shrunk the codebase-standard way — (observed*n + prior*k)/(n+k), the same form
as projector.regress_to_prior / basketball.fit_rates / tennis._shrink — where n is the rate's
REAL denominator (snaps for a per-snap rate, carries for yards-per-carry, targets for
catch-rate). `eff_n` records that denominator plus k for each rate: it is the amount of real
evidence behind the estimate, and it is what the simulator's per-trial parameter-uncertainty
draw is scaled by, so a rookie and a ten-year starter with identical point estimates do not
get identical spreads.

Both the priors and k are measured, not chosen — see config.py's provenance notes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from ..config import cfg, position_cfg
from ..data.base import PlayerWeek, SnapWeek

OPPORTUNITY_RATES = ("carries_per_snap", "targets_per_snap", "pass_att_per_snap")
EFFICIENCY_RATES = ("yards_per_carry", "yards_per_target", "catch_rate",
                    "completion_rate", "yards_per_attempt")

# rate -> (numerator field, denominator field) on the accumulated totals below.
_RATE_TERMS = {
    "carries_per_snap": ("carries", "snaps"),
    "targets_per_snap": ("targets", "snaps"),
    "pass_att_per_snap": ("pass_attempts", "snaps"),
    "yards_per_carry": ("rush_yards", "carries"),
    "yards_per_target": ("rec_yards", "targets"),
    "catch_rate": ("receptions", "targets"),
    "completion_rate": ("completions", "pass_attempts"),
    "yards_per_attempt": ("pass_yards", "pass_attempts"),
}


@dataclass
class UsageRates:
    player: str
    player_id: str
    position: str
    team: str
    opportunity: dict = field(default_factory=dict)     # rate -> per-snap rate
    efficiency: dict = field(default_factory=dict)      # rate -> per-attempt rate
    # Real evidence behind each estimate: its shrinkage denominator (observed denominator +
    # k). Drives the simulator's parameter-uncertainty stage.
    eff_n: dict = field(default_factory=dict)
    n_games: int = 0
    eff_games: float = 0.0                              # recency-weighted
    # Fraction of the estimate coming from the player's own sample rather than the prior,
    # averaged over the rates that actually matter for their position. Board trust reads this.
    sample_weight: float = 0.0
    snap_sample: list = field(default_factory=list)     # offense_pct, most-recent-first
    snaps_sample: list = field(default_factory=list)    # offense_snaps, most-recent-first


def snap_index(snaps: list[SnapWeek]) -> dict[tuple, SnapWeek]:
    """(game_id, team, lowercased player name) -> snap row. nflverse's stat and snap releases
    share no player id (snap_counts carries a PFR id, stats_player a GSIS id), so the join is
    on name within a game — measured at a 95.6% match rate over 2022-2025."""
    return {(s.game_id, s.team, s.player.strip().lower()): s for s in snaps}


def _totals(weighted) -> dict[str, float]:
    """`weighted` is an iterable of (recency weight, PlayerWeek, SnapWeek|None)."""
    t: dict[str, float] = defaultdict(float)
    for wt, w, s in weighted:
        t["snaps"] += wt * float((s.offense_snaps if s else None) or 0.0)
        t["carries"] += wt * w.carries
        t["targets"] += wt * w.targets
        t["pass_attempts"] += wt * w.pass_attempts
        t["rush_yards"] += wt * w.rush_yards
        t["rec_yards"] += wt * w.rec_yards
        t["receptions"] += wt * w.receptions
        t["completions"] += wt * w.completions
        t["pass_yards"] += wt * w.pass_yards
    return t


def positional_priors(weeks: list[PlayerWeek], snaps: list[SnapWeek]) -> dict[str, dict]:
    """League-wide per-position rates recomputed from real data — the shrinkage targets.

    Returns {} when the joined sample is too thin to be a prior in its own right (an
    early-season board where only a week or two exists); callers fall back to config's
    measured multi-season priors, which is the right answer rather than letting week 1 of a
    season define the league's yards-per-carry.
    """
    idx = snap_index(snaps)
    acc: dict[str, list] = defaultdict(list)
    for w in weeks:
        s = idx.get((w.game_id, w.team, w.player.strip().lower()))
        if s is None or not s.offense_snaps:
            continue
        acc[w.position].append((w, s))
    out: dict[str, dict] = {}
    for pos, pairs in acc.items():
        if len(pairs) < 200:
            continue
        t = _totals((1.0, w, s) for w, s in pairs)
        rates = {}
        for rate, (num, den) in _RATE_TERMS.items():
            if t[den] > 0:
                rates[rate] = t[num] / t[den]
        out[pos] = rates
    return out


def prior_for(position: str, measured: dict[str, dict] | None = None) -> dict[str, float]:
    """Config's measured multi-season priors, overridden per-rate by a live league measurement
    when this season already carries enough joined data to produce one."""
    base = dict(position_cfg("positional_priors", position, default={}) or {})
    live = (measured or {}).get((position or "").upper())
    if live:
        base.update(live)
    return base


def fit_usage(weeks: list[PlayerWeek], snaps_by_key: dict[tuple, SnapWeek],
              position: str, priors: dict[str, float],
              halflife: float | None = None) -> UsageRates:
    """`weeks` most-recent-first. Games with no matching snap row are kept for the efficiency
    rates (which don't need snaps) but excluded from the per-snap opportunity denominator, so
    a missing snap row never silently inflates a rate by shrinking its denominator."""
    hl = halflife if halflife is not None else float(cfg("model", "recency_halflife", default=6))
    ks = position_cfg("shrink_k", position, default={}) or {}

    weighted = []
    eff_games = 0.0
    snap_sample, snaps_sample = [], []
    for i, w in enumerate(weeks):
        s = snaps_by_key.get((w.game_id, w.team, w.player.strip().lower()))
        wt = 0.5 ** (i / hl) if hl > 0 else 1.0
        eff_games += wt
        if s is not None:
            if s.offense_pct is not None:
                snap_sample.append(float(s.offense_pct))
            if s.offense_snaps is not None:
                snaps_sample.append(float(s.offense_snaps))
        weighted.append((wt, w, s))

    t_all = _totals(weighted)
    t_snap = _totals(x for x in weighted if x[2] is not None and x[2].offense_snaps)

    opportunity, efficiency, eff_n = {}, {}, {}
    weights = []
    for rate, (num, den) in _RATE_TERMS.items():
        prior = priors.get(rate)
        if prior is None:
            continue
        k = float(ks.get(rate, 100))
        t = t_snap if den == "snaps" else t_all
        observed_den, observed_num = t[den], t[num]
        value = (observed_num + k * prior) / (observed_den + k) if (observed_den + k) > 0 else prior
        (opportunity if rate in OPPORTUNITY_RATES else efficiency)[rate] = value
        eff_n[rate] = observed_den + k
        if observed_den + k > 0:
            weights.append(observed_den / (observed_den + k))

    return UsageRates(
        player=weeks[0].player if weeks else "", player_id=weeks[0].player_id if weeks else "",
        position=(position or "").upper(), team=weeks[0].team if weeks else "",
        opportunity=opportunity, efficiency=efficiency, eff_n=eff_n,
        n_games=len(weeks), eff_games=round(eff_games, 2),
        sample_weight=round(sum(weights) / len(weights), 3) if weights else 0.0,
        snap_sample=snap_sample, snaps_sample=snaps_sample,
    )


def prior_only_usage(position: str, priors: dict[str, float]) -> UsageRates:
    """A player with no usable games: the prior IS the estimate, and every rate's whole
    effective sample is the prior pseudo-count — maximum parameter uncertainty."""
    ks = position_cfg("shrink_k", position, default={}) or {}
    opportunity = {r: priors[r] for r in OPPORTUNITY_RATES if r in priors}
    efficiency = {r: priors[r] for r in EFFICIENCY_RATES if r in priors}
    eff_n = {r: float(ks.get(r, 100)) for r in list(opportunity) + list(efficiency)}
    return UsageRates(player="", player_id="", position=(position or "").upper(), team="",
                      opportunity=opportunity, efficiency=efficiency, eff_n=eff_n,
                      n_games=0, eff_games=0.0, sample_weight=0.0)


def red_zone_opportunities(usage: UsageRates) -> Optional[float]:
    """None, always, today. Red-zone opportunity is only derivable from the play-by-play
    release, which the nflverse adapter deliberately does not pull (400+ columns, ~50MB a
    season, and nothing currently supported needs it — the touchdown props it would serve are
    out of scope). Returning None rather than a share-of-touches estimate keeps it honestly
    unknown; wire pbp and this becomes a real number."""
    return None


def expected_pass_play_snaps(expected_snaps: Optional[float],
                             dropback_share: Optional[float]) -> Optional[float]:
    """Snaps on which the player was on the field for a pass play — the closest real proxy
    this data supports for routes run. nflverse publishes no routes-run column in any free
    release (it is an NGS/PFF field), so this is an UPPER BOUND on routes, not a measurement
    of them: a receiver is on the field for every one of these but does not run a route on all
    of them (max-protect, screens away from them). Callers surface it as `expected_routes`
    with `expected_routes_basis` saying exactly this."""
    if expected_snaps is None or dropback_share is None:
        return None
    return expected_snaps * dropback_share
