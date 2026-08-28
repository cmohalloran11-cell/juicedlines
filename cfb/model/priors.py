"""
League priors + the empirical-Bayes shrinkage strengths behind them -- all fitted from real
CFBD box scores at runtime, never written down as constants (see config.py's provenance rule).

DECOMPOSITION. Every projection in this engine is
`team plays x usage share x efficiency`, never a per-game average of the stat itself, because
those three move for different reasons and CFB's play-count spread across offenses is far
wider than any pro league's (an 85-play tempo offense and a 58-play ball-control offense are
both ordinary FBS teams). Usage share is therefore expressed PER TEAM OFFENSIVE PLAY so
pace.py can scale it independently, exactly the way nfl/model/usage.py expresses opportunity
per offensive snap.

WHAT THE BOX SCORE ACTUALLY CARRIES. CFBD's `/games/players` publishes receptions, not
targets (see cfb/data/cfbd_client.py::_apply_stat -- `targets` is filled from `receptions`
because the feed has no separate targets column). So receiving is modelled as
`receptions_per_play x yards_per_reception`, not `targets x catch_rate x yards_per_target`
like the NFL engine, which has a real targets column to fit against. That is a data
limitation stated honestly rather than a modelling preference, and it is why no CFB catch
rate exists anywhere in this package.

SHRINKAGE. Each rate is shrunk the codebase-standard way -- (observed*n + prior*k)/(n+k), the
same form as projector.regress_to_prior / basketball.fit_rates / nfl.fit_usage / tennis._shrink
-- where n is the rate's REAL denominator (team plays for a per-play usage share, carries for
yards-per-carry, attempts for completion rate). k is not chosen: it is estimated by method of
moments from the league's own between-player variance, which is the textbook empirical-Bayes
estimator and the only way to get a k here without a holdout sweep (NFL fitted its k by
holdout on four seasons of nflverse data; this engine has never had a live CFBD response to
sweep against, so it estimates k from the same rows it estimates the mean from).

    observed spread of player rates  =  sampling noise  +  real between-player spread
    sigma^2_between = Var_w(r_i) - E_w[v / n_i]        v = the rate's per-unit sampling variance
    k               = v / sigma^2_between              (minus 1 for a bounded rate)

When sigma^2_between comes out <= 0 the data is saying every apparent difference between
players is sampling noise, and the honest response is maximum shrinkage (the league mean),
not a small k that pretends otherwise -- the same conclusion NFL's holdout reached
independently for QB completion rate (k=1000, essentially prior-only).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np

from fantasy.player_matching import normalize_name

from ..data.base import PlayerGameStats
from .config import (MIN_PLAYER_GAMES_FOR_PRIORS, MIN_PLAYERS_FOR_SHRINKAGE_FIT,
                     MIN_RECRUITS_FOR_RATING_FIT)

# rate -> (numerator field, denominator field) on an accumulated totals dict.
RATE_TERMS: dict[str, tuple[str, str]] = {
    "rush_att_per_play": ("rush_attempts", "plays"),
    "rec_per_play": ("receptions", "plays"),
    "pass_att_per_play": ("pass_attempts", "plays"),
    "yards_per_carry": ("rush_yards", "rush_attempts"),
    "yards_per_reception": ("rec_yards", "receptions"),
    "yards_per_attempt": ("pass_yards", "pass_attempts"),
    "completion_rate": ("pass_completions", "pass_attempts"),
    "rush_td_per_carry": ("rush_tds", "rush_attempts"),
    "rec_td_per_reception": ("rec_tds", "receptions"),
    "pass_td_per_completion": ("pass_tds", "pass_completions"),
}

OPPORTUNITY_RATES = ("rush_att_per_play", "rec_per_play", "pass_att_per_play")
EFFICIENCY_RATES = tuple(r for r in RATE_TERMS if r not in OPPORTUNITY_RATES)

# Which sampling model each rate's shrinkage k is derived under -- see the module docstring's
# sigma^2_between formula. "count" is Gamma-Poisson (v = mu), "bounded" is Beta-Binomial
# (v = mu(1-mu), k carries the -1), "continuous" is Normal with a measured per-unit variance.
_RATE_KIND = {
    "rush_att_per_play": "count", "rec_per_play": "count", "pass_att_per_play": "count",
    "rush_td_per_carry": "count", "rec_td_per_reception": "count",
    "pass_td_per_completion": "count",
    "completion_rate": "bounded",
    "yards_per_carry": "continuous", "yards_per_reception": "continuous",
    "yards_per_attempt": "continuous",
}

# Yardage rate -> the box-score (yards, count) pair its per-attempt spread is measured from.
_YARD_TERMS = {
    "yards_per_carry": ("rush_yards", "rush_attempts"),
    "yards_per_reception": ("rec_yards", "receptions"),
    "yards_per_attempt": ("pass_yards", "pass_attempts"),
}

_TOTAL_FIELDS = ("plays", "rush_attempts", "receptions", "pass_attempts", "pass_completions",
                 "rush_yards", "rec_yards", "pass_yards", "rush_tds", "rec_tds", "pass_tds")

# Only these are re-scaled by the opponent production factor. The factor is a per-play
# yards/points-added ratio (opponent.py fits it off CFBD's defensive PPA), so yards is the
# quantity it actually describes; applying it to attempts or touchdowns as well would be
# reading an effect into it that its own measurement doesn't contain.
_OPPONENT_SCALED = ("rush_yards", "rec_yards", "pass_yards")


@dataclass
class PlayerTotals:
    """One player's opponent-adjusted, evidence-weighted season totals. `weight` is the sum of
    the per-game evidence weights actually applied (see accumulate_totals) -- games against an
    opponent whose weakness inflated the production count for proportionally less real
    evidence, so a stat line built entirely on FCS/bottom-quartile opponents carries a smaller
    effective sample than the same line against league-average defenses."""
    player_id: str
    player: str = ""
    team: str = ""
    games: int = 0
    weight: float = 0.0
    totals: dict = field(default_factory=lambda: {f: 0.0 for f in _TOTAL_FIELDS})

    def rate(self, key: str) -> Optional[float]:
        num, den = RATE_TERMS[key]
        d = self.totals.get(den, 0.0)
        return (self.totals.get(num, 0.0) / d) if d > 0 else None


@dataclass
class PositionPriors:
    """Shrinkage targets for one position, plus the evidence behind each of them."""
    position: str
    mean: dict = field(default_factory=dict)
    k: dict = field(default_factory=dict)
    unit_sd: dict = field(default_factory=dict)      # per-attempt yard sd, yardage rates only
    n_players: dict = field(default_factory=dict)
    n_player_games: int = 0
    k_basis: dict = field(default_factory=dict)      # rate -> "measured" | "prior_only"


def accumulate_totals(rows: Iterable[PlayerGameStats], team_plays: dict,
                      production_factor: Optional[Callable[[PlayerGameStats], float]] = None
                      ) -> dict[str, PlayerTotals]:
    """Per-player season totals from raw box lines.

    `team_plays` maps (game_id, team) -> that team's offensive plays in that game; a game
    with no entry still contributes to the efficiency rates (which don't need plays) but not
    to the per-play usage denominator, so a missing plays row can never silently inflate a
    usage share by shrinking its own denominator -- the same guard nfl.fit_usage applies to a
    missing snap row.

    `production_factor(row)` returns how much the opponent inflated that game's yardage
    (>1 = a weaker-than-average defense). Yards are divided by it, normalising the game to a
    league-average opponent, and the whole game's evidence weight is min(1, 1/factor), which
    converts the game's exposure into league-average-opponent-equivalent exposure rather than
    applying a discount nobody measured.
    """
    out: dict[str, PlayerTotals] = {}
    for r in rows:
        f = float(production_factor(r)) if production_factor else 1.0
        if f <= 0:
            f = 1.0
        w = min(1.0, 1.0 / f)
        t = out.get(r.player_id)
        if t is None:
            t = out[r.player_id] = PlayerTotals(player_id=r.player_id, player=r.player,
                                                team=r.team)
        t.games += 1
        t.weight += w
        plays = team_plays.get((r.game_id, r.team))
        if plays:
            t.totals["plays"] += w * float(plays)
        for fld in _TOTAL_FIELDS:
            if fld == "plays":
                continue
            v = float(getattr(r, fld))
            t.totals[fld] += (w / f) * v if fld in _OPPONENT_SCALED else w * v
    return out


def _mom_shrinkage_k(mu: float, pairs: list, unit_var: float, bounded: bool) -> Optional[float]:
    """Method-of-moments empirical-Bayes k, in DENOMINATOR units. None when the league sample
    is too thin to estimate a between-player variance, or when the observed spread of player
    rates is no wider than pure sampling noise -- both mean "shrink all the way to the league
    mean", which the caller expresses as prior-only rather than as some small k."""
    usable = [(num, den) for num, den in pairs if den > 0]
    total_den = sum(den for _, den in usable)
    if len(usable) < MIN_PLAYERS_FOR_SHRINKAGE_FIT or total_den <= 0 or mu <= 0 or unit_var <= 0:
        return None
    observed = sum(den * (num / den - mu) ** 2 for num, den in usable) / total_den
    sampling = len(usable) * unit_var / total_den
    between = observed - sampling
    if between <= 0:
        return None
    k = unit_var / between
    if bounded:
        k -= 1.0
    return k if k > 0 else None


def _unit_variance(rows: Iterable[PlayerGameStats], rate: str, mu: float) -> Optional[float]:
    """Per-attempt variance of a yardage rate, measured off real game lines: the residual of
    (game yards - attempts * league mean) pooled over attempts. This is both the sampling
    variance the shrinkage k needs and the spread the simulator's Gamma yards draw uses, so
    they are measured once, from the same rows, and can't drift apart."""
    yards_f, count_f = _YARD_TERMS[rate]
    num = den = 0.0
    for r in rows:
        c = getattr(r, count_f)
        if c <= 0:
            continue
        num += (getattr(r, yards_f) - c * mu) ** 2
        den += c
    return (num / den) if den > 0 else None


def fit_positional_priors(totals_by_player: dict, rows: list[PlayerGameStats],
                          positions: dict) -> dict[str, PositionPriors]:
    """League per-play/per-attempt rates by position, with their empirical-Bayes shrinkage k
    and (for yardage) the measured per-attempt spread.

    `positions` maps a CFBD athlete id to a position string; an id it doesn't carry lands in
    the pooled "" bucket, which is a real measurement over every player rather than a guess at
    what position they play. A position whose own sample is below the gate is not emitted at
    all and its players fall back to the pooled bucket -- the same "don't let a thin slice
    define a league prior" rule nfl.positional_priors applies.
    """
    by_pos: dict[str, list] = defaultdict(list)
    rows_by_pos: dict[str, list] = defaultdict(list)
    for pid, t in totals_by_player.items():
        pos = (positions.get(pid) or "").upper()
        by_pos[pos].append(t)
        if pos:
            by_pos[""].append(t)
    for r in rows:
        pos = (positions.get(r.player_id) or "").upper()
        rows_by_pos[pos].append(r)
        if pos:
            rows_by_pos[""].append(r)

    out: dict[str, PositionPriors] = {}
    for pos, players in by_pos.items():
        prs = rows_by_pos.get(pos, [])
        if len(prs) < MIN_PLAYER_GAMES_FOR_PRIORS:
            continue
        p = PositionPriors(position=pos, n_player_games=len(prs))
        for rate, (num_f, den_f) in RATE_TERMS.items():
            pairs = [(t.totals[num_f], t.totals[den_f]) for t in players
                     if t.totals[den_f] > 0]
            total_den = sum(d for _, d in pairs)
            if not pairs or total_den <= 0:
                continue
            mu = sum(n for n, _ in pairs) / total_den
            kind = _RATE_KIND[rate]
            if kind == "continuous":
                uv = _unit_variance(prs, rate, mu)
                if not uv or uv <= 0:
                    continue
                p.unit_sd[rate] = float(np.sqrt(uv))
            else:
                uv = mu * (1.0 - mu) if kind == "bounded" else mu
            p.mean[rate] = mu
            p.n_players[rate] = len(pairs)
            k = _mom_shrinkage_k(mu, pairs, uv, bounded=(kind == "bounded"))
            if k is None:
                p.k[rate] = total_den
                p.k_basis[rate] = "prior_only"
            else:
                p.k[rate] = k
                p.k_basis[rate] = "measured"
        out[pos] = p
    return out


def priors_for(position: Optional[str], priors: dict) -> Optional[PositionPriors]:
    """This position's priors, or the pooled bucket when the position is unknown or its own
    sample was too thin to emit. None when not even the pooled bucket could be fitted, which
    the caller must treat as "this league has no usable prior yet", not as zero."""
    p = priors.get((position or "").upper())
    return p if p is not None else priors.get("")


# ── competition level (the tier-B transfer translation) ──────────────────────────────────

def team_tier(conference: Optional[str], classification: Optional[str],
              power_conferences) -> Optional[str]:
    """"power" | "group" | "fcs", or None when CFBD carries neither field for the team. Read
    from whatever conference/classification CFBD returns, so a realigned team re-tiers itself
    on the next roster sync rather than tracking a hardcoded membership list."""
    cls = (classification or "").lower()
    if cls and cls != "fbs":
        return "fcs" if cls == "fcs" else None
    if conference and conference in power_conferences:
        return "power"
    if conference or cls == "fbs":
        return "group"
    return None


def fit_level_factors(totals_by_player: dict, tier_of_player: dict) -> dict[str, dict[str, float]]:
    """{rate: {tier: league mean rate at that competition level}}.

    This is what makes a transfer's prior a real translation instead of an asserted discount:
    the step up from G5 or FCS to a power conference is measured as the ratio of the two
    levels' own league means for that exact rate, on the same rows everything else is fitted
    from. A tier with too few players for a rate is simply absent, and level_factor() then
    returns 1.0 (no translation) rather than a guessed penalty.
    """
    acc: dict[str, dict[str, list]] = {r: defaultdict(list) for r in RATE_TERMS}
    for pid, t in totals_by_player.items():
        tier = tier_of_player.get(pid)
        if not tier:
            continue
        for rate, (num_f, den_f) in RATE_TERMS.items():
            if t.totals[den_f] > 0:
                acc[rate][tier].append((t.totals[num_f], t.totals[den_f]))
    out: dict[str, dict[str, float]] = {}
    for rate, by_tier in acc.items():
        means = {}
        for tier, pairs in by_tier.items():
            if len(pairs) < MIN_PLAYERS_FOR_SHRINKAGE_FIT:
                continue
            den = sum(d for _, d in pairs)
            if den > 0:
                means[tier] = sum(n for n, _ in pairs) / den
        if means:
            out[rate] = means
    return out


def level_factor(factors: dict, rate: str, origin_tier: Optional[str],
                 destination_tier: Optional[str]) -> float:
    """Multiplier translating a rate observed at `origin_tier` to `destination_tier`. Exactly
    1.0 -- no translation -- whenever either level's mean wasn't measurable, which is the
    honest answer for an FCS transfer in a season where no FCS player accumulated enough
    FBS-visible production to measure a level from."""
    if not origin_tier or not destination_tier or origin_tier == destination_tier:
        return 1.0
    means = factors.get(rate) or {}
    o, d = means.get(origin_tier), means.get(destination_tier)
    if not o or not d or o <= 0:
        return 1.0
    return d / o


# ── recruiting rating (the tier-C true-freshman prior) ───────────────────────────────────

@dataclass
class RecruitingPrior:
    """rate -> (intercept, slope) of a least-squares fit of first-season usage on recruiting
    rating, per position. Only USAGE is fitted: a rating is a scouting judgement about role
    and athleticism, and there is no measurement here that it predicts yards per carry, so
    tier-C efficiency stays at the positional prior rather than being nudged by a signal
    nothing showed to be relevant."""
    position: str
    coef: dict = field(default_factory=dict)
    n: int = 0


def fit_recruiting_prior(ratings: list, totals_by_player: dict, name_index: dict,
                         positions: dict) -> dict[str, RecruitingPrior]:
    """Fit rating -> per-play usage on real first-season production.

    `ratings` are CFBD recruiting rows for a signing class; `name_index` maps a normalized
    player name to the athlete id the box scores use (the recruiting feed and the box-score
    feed share no id). Returns {} when fewer than the gate's worth of recruits actually
    produced a usable stat line -- no rating map is emitted off a handful of players.
    """
    by_pos: dict[str, list] = defaultdict(list)
    for rec in ratings:
        if rec.rating is None:
            continue
        pid = name_index.get(normalize_name(rec.name))
        t = totals_by_player.get(pid) if pid else None
        if t is None or t.totals["plays"] <= 0:
            continue
        pos = (positions.get(pid) or rec.position or "").upper()
        by_pos[pos].append((float(rec.rating), t))
        if pos:
            by_pos[""].append((float(rec.rating), t))

    out: dict[str, RecruitingPrior] = {}
    for pos, pairs in by_pos.items():
        if len(pairs) < MIN_RECRUITS_FOR_RATING_FIT:
            continue
        rp = RecruitingPrior(position=pos, n=len(pairs))
        x = np.array([[1.0, r] for r, _ in pairs])
        for rate in OPPORTUNITY_RATES:
            num_f, den_f = RATE_TERMS[rate]
            y = np.array([t.totals[num_f] / t.totals[den_f] for _, t in pairs])
            coef = np.linalg.lstsq(x, y, rcond=None)[0]
            rp.coef[rate] = (float(coef[0]), float(coef[1]))
        out[pos] = rp
    return out


def recruiting_rate(prior: Optional[RecruitingPrior], rate: str,
                    rating: Optional[float]) -> Optional[float]:
    """Rating-implied usage rate, or None when no fit exists for this position/rate or the
    player has no rating -- the caller then falls back to the flat positional prior."""
    if prior is None or rating is None or rate not in prior.coef:
        return None
    a, b = prior.coef[rate]
    v = a + b * float(rating)
    return v if v > 0 else None
