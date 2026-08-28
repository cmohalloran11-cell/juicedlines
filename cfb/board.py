"""
cfb.board -- attach the CFB engine's projections to live board lines.

Same contract as tennis/board.py::attach_tennis, basketball/board.py::attach_basketball and
nfl/board.py::attach_nfl: take the board's line dicts, filter to this sport, group by
(player, market, game), project each player once, and write
model_proj / model_prob / model_edge / model_floor / model_ceiling / model_median / model_n /
proj_kind plus the pre-anchor ledger fields onto every line this sport can price. Called from
analytics.attach_projections exactly like the other sports, so both deploy paths (main.py's
live loop, build_static.py) pick it up with no further wiring.

MARKET ANCHORING is the identical mechanism to NFL/WNBA/Tennis, on the MEDIAN, never the mean.
A market line IS the book's implied 50/50 threshold by definition, so the model's own median is
the apples-to-apples quantity to blend it with; blending the mean and recentering the array
pins the MEAN to the line and leaves the MEDIAN -- which model_prob is computed from -- below
it by the distribution's own skew, which is exactly the bug that had 94% of the live NFL
preseason board recommending Under in 2026-08 (see nfl/board.py's own comment and
provenance.MODEL_CHANGELOG's nfl-1.2.0 / wnba-1.2.0 / tennis-1.3.0 entries). CFB yardage is
Gamma-shaped and right-skewed, so it would reproduce that bug exactly if it blended the mean.

ANYTIME TD is anchored differently and deliberately: its simulated array is 0/1, so a median
shift is meaningless (the median of a Bernoulli array is 0 or 1). The model probability is
blended toward the book's own implied probability instead, which is the same anchoring idea
applied to the quantity a binary market actually prices.

PROJ_KIND carries which of the three prior tiers actually priced the row --
cfb_prior_a (returning production), cfb_prior_b (transfer, level-translated), cfb_prior_c
(recruiting rating / no college production). Every calibration query in db.py already scopes
by proj_kind, so the three tiers can be scored separately the day CFB has graded rows, which
matters more here than for any other sport: they are three genuinely different models, and
the tier-C one is the weakest by construction.

PLAYER STATUS: no CFB injury report is mandated to exist, so cfb/player_status.py's manual
override is the availability signal. A projection whose status is unknown or was last
confirmed too long before kickoff is FLAGGED (`cfb_status_stale`) rather than suppressed or
silently trust-penalised -- flagging is what the data supports; a numeric penalty would be a
constant nobody measured.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

from . import projections as P
from .model.config import BOARD_FULL_TRUST_AT, BOARD_MIN_TRUST
from .player_status import is_stale
from .sim.engine import MARKET_KEYS

_SPORTS = ("CFB",)
_BINARY_MARKETS = ("Anytime TD",)

# Every CFB-specific field this module writes onto a line, for the integration layer.
CFB_FIELDS = (
    "proj_kind", "prior_tier_reason", "projected_plays", "pace_basis", "usage_share",
    "blowout_probability", "expected_margin", "blowout_basis", "garbage_time_multiplier",
    "starterness", "opponent", "opponent_factor", "opponent_is_bottom_quartile",
    "level_factor", "cfb_status_stale", "trust_weight", "model_raw", "model_raw_prob",
    "model_pre_mean", "model_pre_median", "model_pre_sd", "model_pre_prob", "model_anchor_t",
    "p25", "p75", "model_std_dev",
)


def _status_stale(line: dict) -> Optional[bool]:
    """None when the line carries no canonical player id at all (an unresolved Odds-API name --
    see cfb/player_matching.py), because "we never identified this player" is a different
    statement from "we identified him and have no status"."""
    player_id = (line.get("meta") or {}).get("cfb_player_id")
    if not player_id:
        return None
    try:
        import store
        from .player_status import get_status
        row = get_status(store.get_database(), player_id)
    except Exception:
        return None
    return is_stale(row, line.get("start_time"))


def attach_cfb(lines: list[dict]) -> int:
    """Attach projections to live CFB lines. Returns the number of lines projected."""
    clines = [l for l in lines if l.get("sport") in _SPORTS
              and l.get("player") and l.get("line") is not None
              and l.get("stat_type") in MARKET_KEYS]
    if not clines:
        return 0

    try:
        data = P.league_data()
    except Exception as exc:
        print(f"[cfb.board] league_data failed: {type(exc).__name__}: {exc}", flush=True)
        return 0
    if data is None:
        return 0

    groups: dict = defaultdict(list)
    for l in clines:
        game = P.find_game(data, l.get("team"), l.get("start_time"))
        groups[(P.normalize_name(l["player"]), l.get("team"), l["stat_type"],
                game.id if game else None)].append((l, game))

    proj_cache: dict = {}
    done = 0
    for (_nname, team, stat_type, _gid), members in groups.items():
        first, game = members[0]
        ck = (_nname, team, first.get("position"), _gid)
        if ck not in proj_cache:
            try:
                proj_cache[ck] = P.project_player(first["player"], team=team,
                                                  position=first.get("position"), game=game)
            except Exception as exc:
                print(f"[cfb.board] project_player({first['player']!r}) failed: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                proj_cache[ck] = None
        proj = proj_cache[ck]
        if not proj:
            continue
        arr = P.market_dist(proj, stat_type)
        if arr is None:
            continue

        glines = [l for l, _ in members]
        trust = min(1.0, P.market_sample_weight(proj, stat_type) / BOARD_FULL_TRUST_AT)
        anchored = trust >= BOARD_MIN_TRUST
        stale = _status_stale(first)

        if stat_type in _BINARY_MARKETS:
            _write_binary(glines, proj, arr, trust, anchored, stale)
        else:
            _write_continuous(glines, proj, arr, trust, anchored, stale)
        done += len(glines)
    return done


def _common_fields(l: dict, proj: dict, arr: np.ndarray, trust: float, anchored: bool,
                   stale: Optional[bool]) -> None:
    l["proj_kind"] = proj["proj_kind"]
    l["prior_tier_reason"] = proj["tier_reason"]
    l["model_n"] = proj["n_games"]
    l["trust_weight"] = round(float(trust), 3)
    l["model_anchor_t"] = round(float(trust), 3) if anchored else 0.0
    l["model_pre_sd"] = round(float(arr.std()), 4)
    l["projected_plays"] = proj["projected_plays"]
    l["pace_basis"] = proj["pace_basis"]
    l["usage_share"] = proj["usage_share"]
    l["blowout_probability"] = proj["blowout_probability"]
    l["expected_margin"] = proj["expected_margin"]
    l["blowout_basis"] = proj["blowout_basis"]
    l["garbage_time_multiplier"] = proj["garbage_time_multiplier"]
    l["starterness"] = proj["starterness"]
    l["opponent"] = proj["opponent"]
    l["opponent_factor"] = proj["opponent_factor"]
    l["opponent_is_bottom_quartile"] = proj["opponent_is_bottom_quartile"]
    l["level_factor"] = proj["level_factor"]
    l["cfb_status_stale"] = stale


def _write_continuous(glines: list, proj: dict, arr: np.ndarray, trust: float,
                      anchored: bool, stale: Optional[bool]) -> None:
    model_mean = float(arr.mean())
    model_median = float(np.median(arr))
    std = [float(l["line"]) for l in glines if (l.get("odds_type") or "standard") == "standard"]
    anchor = float(np.median(std)) if std else float(np.median([float(l["line"]) for l in glines]))

    blended_median = trust * model_median + (1.0 - trust) * anchor if anchored else anchor
    shifted = arr + (blended_median - model_median)
    q = np.percentile(shifted, [10, 25, 50, 75, 90])
    center = float(shifted.mean())

    for l in glines:
        line_val = float(l["line"])
        l["model_proj"] = round(center, 1)
        l["model_prob"] = round(float((shifted > line_val).mean()), 4)
        l["model_median"] = round(float(q[2]), 1)
        l["model_floor"] = round(max(0.0, float(q[0])), 1)
        l["model_ceiling"] = round(max(float(q[4]), float(q[0])), 1)
        l["model_edge"] = round(center - line_val, 1)
        l["p25"] = round(float(q[1]), 1)
        l["p75"] = round(float(q[3]), 1)
        l["model_std_dev"] = round(float(shifted.std()), 2)
        l["model_raw"] = round(model_mean, 2)
        l["model_raw_prob"] = round(float((arr > line_val).mean()), 4)
        l["model_pre_mean"] = round(model_mean, 2)
        l["model_pre_median"] = round(model_median, 2)
        l["model_pre_prob"] = l["model_raw_prob"]
        _common_fields(l, proj, arr, trust, anchored, stale)


def _write_binary(glines: list, proj: dict, arr: np.ndarray, trust: float,
                  anchored: bool, stale: Optional[bool]) -> None:
    model_p = float(arr.mean())
    for l in glines:
        line_val = float(l["line"])
        market_p = l.get("over_implied")
        blended = model_p
        if market_p is not None:
            blended = (trust * model_p + (1.0 - trust) * float(market_p) if anchored
                       else float(market_p))
        l["model_proj"] = round(blended, 3)
        l["model_prob"] = round(blended, 4)
        l["model_median"] = round(blended, 3)
        l["model_floor"] = 0.0
        l["model_ceiling"] = 1.0
        l["model_edge"] = round(blended - (float(market_p) if market_p is not None else line_val), 3)
        l["model_std_dev"] = round(float(arr.std()), 4)
        l["model_raw"] = round(model_p, 4)
        l["model_raw_prob"] = round(model_p, 4)
        l["model_pre_mean"] = round(model_p, 4)
        l["model_pre_median"] = round(model_p, 4)
        l["model_pre_prob"] = round(model_p, 4)
        _common_fields(l, proj, arr, trust, anchored, stale)
