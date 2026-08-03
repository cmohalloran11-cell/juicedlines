"""
mlb_model.py — single-game MLB projection engine.

Pipeline
--------
player skill rates
        ↓
game context adjustment
        ↓
PA/BF workload simulation
        ↓
correlated event simulation
        ↓
derived stat generation
        ↓
ensemble residual calibration
        ↓
Projection objects

Drop-in replacement for the JUICED MLB projection pipeline.
"""

from __future__ import annotations

from typing import Any
import math

from . import montecarlo as mc
from . import ensemble
from .base import Projection, summarize, _f


# ─────────────────────────────────────────────
# League baselines
# ─────────────────────────────────────────────

LG_RUNS = 4.30
LG_XFIP = 4.10

LG_HIT_RATE = 0.245
LG_HR_RATE = 0.035
LG_BB_RATE = 0.085
LG_K_RATE = 0.225


BATTER_STATS = [
    "plate_appearances",
    "hits",
    "home_runs",
    "rbis",
    "runs",
    "stolen_bases",
    "total_bases",
    "walks",
    "strikeouts",
]


PITCHER_STATS = [
    "strikeouts",
    "earned_runs",
    "runs_allowed",
    "innings_pitched",
    "hits_allowed",
    "walks_allowed",
    "outs_recorded",
]


XGB_CACHE: dict[tuple, Any] = {}


# ─────────────────────────────────────────────
# Generic helpers
# ─────────────────────────────────────────────

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _rate(value: Any, default: float) -> float:
    return _clamp(
        _f(value, default),
        0.0001,
        0.999,
    )


# ─────────────────────────────────────────────
# Context adjustments
# ─────────────────────────────────────────────


def _park_mult(ctx: dict) -> float:
    return _clamp(
        _f(ctx.get("park_factor"), 1.0),
        0.85,
        1.20,
    )


def _weather_hr_mult(ctx: dict) -> float:
    """
    Weather impacts HR carry much more than BB/K.
    """

    if not ctx.get("outdoor"):
        return 1.0

    temp = _f(ctx.get("temp_f"), 70)
    wind = _f(ctx.get("wind_mph"), 0)

    direction = str(
        ctx.get("wind_dir") or ""
    ).lower()

    mult = 1 + ((temp - 70) * 0.004)

    if "out" in direction:
        mult *= 1 + wind * 0.012

    elif "in" in direction:
        mult *= 1 - wind * 0.010

    return _clamp(mult, 0.70, 1.40)


def _opp_pitcher_mult(ctx: dict) -> float:
    """
    Higher xFIP = easier matchup for hitter.
    """

    xfip = _f(
        ctx.get("opp_pitcher_xfip"),
        LG_XFIP,
    )

    return _clamp(
        xfip / LG_XFIP,
        0.75,
        1.30,
    )


def _team_total_mult(ctx: dict) -> float:

    total = _f(
        ctx.get("vegas_team_total"),
        LG_RUNS,
    )

    return _clamp(
        total / LG_RUNS,
        0.80,
        1.30,
    )


def _opp_offense_mult(ctx: dict) -> float:

    total = _f(
        ctx.get("opp_team_total"),
        LG_RUNS,
    )

    return _clamp(
        total / LG_RUNS,
        0.80,
        1.30,
    )


def _platoon_mult(
    form: dict,
    ctx: dict,
    stat: str,
) -> float:

    hand = str(
        ctx.get("opp_pitcher_hand") or ""
    ).upper()[:1]

    if hand not in ("L", "R"):
        return 1.0

    splits = form.get(
        "platoon",
        {},
    )

    split = splits.get(
        f"vs_{hand}",
        {},
    )

    base = _f(
        form.get(stat),
        0,
    )

    value = _f(
        split.get(stat),
        base,
    )

    if base <= 0:
        return 1.0

    return _clamp(
        value / base,
        0.70,
        1.30,
    )


# ─────────────────────────────────────────────
# Workload models
# ─────────────────────────────────────────────


def _expected_pa(
    form: dict,
    ctx: dict,
) -> float:

    base = _f(
        form.get("exp_pa"),
        4.2,
    )

    vegas = _team_total_mult(ctx)

    lineup = _f(
        ctx.get("lineup_slot_factor"),
        1.0,
    )

    return _clamp(
        base *
        (0.90 + vegas * 0.10) *
        lineup,
        1.5,
        7.0,
    )


def _expected_bf(
    form: dict,
) -> float:

    return _clamp(
        _f(form.get("exp_bf"), 23),
        8,
        45,
    )


# ─────────────────────────────────────────────
# Correlated hitter PA engine
# ─────────────────────────────────────────────


def _simulate_pa_events(
    pa_trials,
    rates: dict,
):

    results = {
        "hits": [],
        "hr": [],
        "tb": [],
        "bb": [],
        "k": [],
        "singles": [],
        "doubles": [],
        "triples": [],
    }


    for pa in pa_trials:

        hits = 0
        hr = 0
        tb = 0
        bb = 0
        k = 0

        singles = 0
        doubles = 0
        triples = 0


        for _ in range(
            int(pa)
        ):

            r = mc.random_uniform()

            if r < rates["hr"]:

                hits += 1
                hr += 1
                tb += 4


            elif r < rates["hr"] + rates["k"]:

                k += 1


            elif r < (
                rates["hr"]
                +
                rates["k"]
                +
                rates["bb"]
            ):

                bb += 1


            elif r < (
                rates["hr"]
                +
                rates["k"]
                +
                rates["bb"]
                +
                rates["1b"]
            ):

                hits += 1
                singles += 1
                tb += 1


            elif r < (
                rates["hr"]
                +
                rates["k"]
                +
                rates["bb"]
                +
                rates["1b"]
                +
                rates["2b"]
            ):

                hits += 1
                doubles += 1
                tb += 2


            else:

                hits += 1
                triples += 1
                tb += 3


        results["hits"].append(hits)
        results["hr"].append(hr)
        results["tb"].append(tb)
        results["bb"].append(bb)
        results["k"].append(k)

        results["singles"].append(singles)
        results["doubles"].append(doubles)
        results["triples"].append(triples)


    return results

# ─────────────────────────────────────────────
# Batter projection
# ─────────────────────────────────────────────


def project_batter(
    form: dict,
    ctx: dict,
    n: int = mc.N_SIMS,
    use_ensemble: bool = True,
) -> dict[str, Projection]:

    park = _park_mult(ctx)
    weather = _weather_hr_mult(ctx)
    pitcher = _opp_pitcher_mult(ctx)
    vegas = _team_total_mult(ctx)


    # Vegas already captures some park/pitcher information.
    # Reduce stacking by applying environmental factors partially.
    offense_mult = (
        vegas
        *
        math.sqrt(
            park * pitcher
        )
    )


    drivers = [
        f"park ×{park:.2f}",
        f"SP matchup ×{pitcher:.2f}",
        f"Vegas ×{vegas:.2f}",
    ]

    if ctx.get("outdoor"):
        drivers.append(
            f"weather HR ×{weather:.2f}"
        )


    # ─────────────────────────────────────────
    # PA simulation
    # ─────────────────────────────────────────

    exp_pa = _expected_pa(
        form,
        ctx,
    )

    pa_trials = mc.trial_counts(
        exp_pa,
        n,
        lo=1,
        hi=8,
    )


    # ─────────────────────────────────────────
    # Batter skill rates
    # ─────────────────────────────────────────

    hit_rate = (
        _rate(
            form.get("p_hit"),
            LG_HIT_RATE,
        )
        *
        _platoon_mult(
            form,
            ctx,
            "p_hit",
        )
    )


    hr_rate = (
        _rate(
            form.get("p_hr"),
            LG_HR_RATE,
        )
        *
        _platoon_mult(
            form,
            ctx,
            "p_hr",
        )
    )


    bb_rate = _rate(
        form.get("p_bb"),
        LG_BB_RATE,
    )


    k_rate = _rate(
        form.get("p_k"),
        0.22,
    )


    # Opposing pitcher strikeout ability
    opp_k = (
        _f(
            ctx.get("opp_pitcher_k_per_pa"),
            LG_K_RATE,
        )
        /
        LG_K_RATE
    )


    k_rate *= _clamp(
        opp_k,
        0.75,
        1.30,
    )


    # ─────────────────────────────────────────
    # Context adjustment
    # ─────────────────────────────────────────


    hr_rate *= (
        park
        *
        weather
        *
        math.sqrt(
            pitcher * vegas
        )
    )


    hit_rate *= offense_mult


    # ─────────────────────────────────────────
    # Hit distribution
    # ─────────────────────────────────────────

    non_hr_hits = max(
        0.001,
        hit_rate - hr_rate,
    )


    one_b = (
        non_hr_hits
        *
        _f(
            form.get("p_1b_pct"),
            0.72,
        )
    )

    two_b = (
        non_hr_hits
        *
        _f(
            form.get("p_2b_pct"),
            0.24,
        )
    )


    three_b = max(
        0,
        non_hr_hits
        -
        one_b
        -
        two_b,
    )


    # Ensure valid PA probability tree

    total = (
        hr_rate
        +
        k_rate
        +
        bb_rate
        +
        one_b
        +
        two_b
        +
        three_b
    )


    if total > 0.98:

        scale = 0.98 / total

        hr_rate *= scale
        k_rate *= scale
        bb_rate *= scale
        one_b *= scale
        two_b *= scale
        three_b *= scale


    events = _simulate_pa_events(
        pa_trials,
        {
            "hr": hr_rate,
            "k": k_rate,
            "bb": bb_rate,
            "1b": one_b,
            "2b": two_b,
            "3b": three_b,
        },
    )


    out = {}


    # ─────────────────────────────────────────
    # Direct markets
    # ─────────────────────────────────────────


    out["plate_appearances"] = summarize(
        pa_trials,
        "plate_appearances",
        [
            f"expected PA {exp_pa:.2f}"
        ],
    )


    out["hits"] = summarize(
        events["hits"],
        "hits",
        drivers,
    )


    out["home_runs"] = summarize(
        events["hr"],
        "home_runs",
        drivers + [
            f"HR rate {hr_rate:.3f}"
        ],
    )


    out["total_bases"] = summarize(
        events["tb"],
        "total_bases",
        drivers,
    )


    out["walks"] = summarize(
        events["bb"],
        "walks",
        [
            f"BB {bb_rate:.3f}"
        ],
    )


    out["strikeouts"] = summarize(
        events["k"],
        "strikeouts",
        [
            f"K {k_rate:.3f}"
        ],
    )


    # ─────────────────────────────────────────
    # Runs / RBI opportunity model
    # ─────────────────────────────────────────

    run_trials = []
    rbi_trials = []


    base_runs = _f(
        form.get("exp_runs"),
        0.45,
    )

    base_rbi = _f(
        form.get("exp_rbi"),
        0.45,
    )


    for hits, walks in zip(
        events["hits"],
        events["bb"],
    ):

        opportunities = (
            hits
            +
            walks
        )


        on_base_factor = (
            0.75
            +
            opportunities /
            max(1, exp_pa)
        )


        run_expectation = (
            base_runs
            *
            offense_mult
            *
            on_base_factor
        )


        rbi_expectation = (
            base_rbi
            *
            offense_mult
            *
            on_base_factor
        )


        run_trials.append(
            mc.negbinom_value(
                max(
                    0.05,
                    run_expectation,
                ),
                0.65,
            )
        )


        rbi_trials.append(
            mc.negbinom_value(
                max(
                    0.05,
                    rbi_expectation,
                ),
                0.65,
            )
        )


    out["runs"] = summarize(
        run_trials,
        "runs",
        drivers,
    )


    out["rbis"] = summarize(
        rbi_trials,
        "rbis",
        drivers,
    )


    # ─────────────────────────────────────────
    # Stolen bases
    # ─────────────────────────────────────────


    sb_base = _f(
        form.get("exp_sb"),
        0.08,
    )


    sb_rate = (
        sb_base
        *
        offense_mult
    )


    out["stolen_bases"] = summarize(
        mc.negbinom_count(
            max(
                0.01,
                sb_rate,
            ),
            0.25,
            n,
        ),
        "stolen_bases",
        [
            f"SB rate {sb_rate:.2f}"
        ],
    )


    if use_ensemble:
        _apply_ensemble(
            "mlb",
            out,
            form,
        )


    return out



# ─────────────────────────────────────────────
# Pitcher event simulation
# ─────────────────────────────────────────────


def _simulate_pitcher_bf(
    bf_trials,
    rates: dict,
):

    results = {
        "strikeouts": [],
        "walks": [],
        "hits": [],
    }


    for bf in bf_trials:

        k = 0
        bb = 0
        h = 0


        for _ in range(int(bf)):

            r = mc.random_uniform()


            if r < rates["k"]:
                k += 1


            elif r < (
                rates["k"]
                +
                rates["bb"]
            ):
                bb += 1


            elif r < (
                rates["k"]
                +
                rates["bb"]
                +
                rates["h"]
            ):
                h += 1


        results["strikeouts"].append(k)
        results["walks"].append(bb)
        results["hits"].append(h)


    return results



# ─────────────────────────────────────────────
# Pitcher projection
# ─────────────────────────────────────────────


def project_pitcher(
    form: dict,
    ctx: dict,
    n: int = mc.N_SIMS,
    use_ensemble: bool = True,
) -> dict[str, Projection]:

    park = _park_mult(ctx)

    opp = _opp_offense_mult(ctx)


    drivers = [
        f"opp offense ×{opp:.2f}",
        f"park ×{park:.2f}",
    ]


    exp_bf = _expected_bf(
        form
    )


    exp_outs = _f(
        form.get("exp_outs"),
        17,
    )


    bf_trials = mc.trial_counts(
        exp_bf,
        n,
        lo=8,
        hi=45,
    )


    outs_trials = mc.outs_recorded(
        exp_outs,
        sd=4,
        n=n,
    )


    k_rate = _rate(
        form.get("p_k"),
        0.235,
    )

    bb_rate = _rate(
        form.get("p_bb"),
        0.075,
    )

    hit_rate = _rate(
        form.get("p_h"),
        0.21,
    )


    # Matchup adjustments

    k_rate *= (
        1.05
        -
        0.15 * opp
    )

    bb_rate *= (
        0.90
        +
        0.15 * opp
    )

    hit_rate *= opp


    k_rate = _clamp(
        k_rate,
        0.05,
        0.45,
    )

    bb_rate = _clamp(
        bb_rate,
        0.01,
        0.20,
    )

    hit_rate = _clamp(
        hit_rate,
        0.05,
        0.40,
    )


    events = _simulate_pitcher_bf(
        bf_trials,
        {
            "k": k_rate,
            "bb": bb_rate,
            "h": hit_rate,
        },
    )

# ─────────────────────────────────────────────
# Batter projection
# ─────────────────────────────────────────────

def project_batter(
    form: dict,
    ctx: dict,
    n: int = mc.N_SIMS,
    use_ensemble: bool = True,
) -> dict[str, Projection]:

    park = _park_multiplier(ctx)
    weather = _weather_hr_multiplier(ctx)
    pitcher = _pitcher_quality_multiplier(ctx)
    vegas = _team_environment_multiplier(ctx)

    # Vegas already captures some park + pitcher information.
    # Prevent double counting.
    offense_mult = vegas * math.sqrt(
        park * pitcher
    )

    drivers = [
        f"park ×{park:.2f}",
        f"SP quality ×{pitcher:.2f}",
        f"Vegas ×{vegas:.2f}",
    ]

    if ctx.get("outdoor"):
        drivers.append(
            f"weather HR ×{weather:.2f}"
        )


    # ─────────────────────────────────────────
    # Plate appearance distribution
    # ─────────────────────────────────────────

    exp_pa = _expected_pa(
        form,
        ctx,
    )

    pa_trials = mc.trial_counts(
        exp_pa,
        n,
        lo=1,
        hi=8,
    )


    # ─────────────────────────────────────────
    # Base rates
    # ─────────────────────────────────────────

    hit_rate = (
        _safe_rate(
            form.get("p_hit"),
            LG_HIT_RATE,
        )
        *
        _platoon_multiplier(
            form,
            ctx,
            "p_hit",
        )
    )


    hr_rate = (
        _safe_rate(
            form.get("p_hr"),
            LG_HR_RATE,
        )
        *
        _platoon_multiplier(
            form,
            ctx,
            "p_hr",
        )
    )


    bb_rate = _safe_rate(
        form.get("p_bb"),
        LG_BB_RATE,
    )


    k_rate = _safe_rate(
        form.get("p_k"),
        0.22,
    )


    # opposing pitcher strikeout environment

    opp_k = (
        _f(
            ctx.get("opp_pitcher_k_per_pa"),
            LG_K_RATE,
        )
        /
        LG_K_RATE
    )

    k_rate *= _clamp(
        opp_k,
        0.75,
        1.25,
    )


    # ─────────────────────────────────────────
    # Environment adjustment
    # ─────────────────────────────────────────

    hr_rate *= (
        park
        *
        weather
        *
        math.sqrt(
            pitcher * vegas
        )
    )

    hit_rate *= offense_mult


    # HR is included inside hits
    # Remove it before splitting singles/doubles/triples

    non_hr_hits = max(
        0.01,
        hit_rate - hr_rate,
    )


    one_b = (
        non_hr_hits *
        _f(
            form.get("p_1b_pct"),
            0.72,
        )
    )

    two_b = (
        non_hr_hits *
        _f(
            form.get("p_2b_pct"),
            0.24,
        )
    )

    three_b = max(
        0,
        non_hr_hits - one_b - two_b
    )


    # prevent impossible PA probabilities

    total_probability = (
        hr_rate
        +
        k_rate
        +
        bb_rate
        +
        one_b
        +
        two_b
        +
        three_b
    )


    if total_probability > 0.98:

        scale = (
            0.98 /
            total_probability
        )

        hr_rate *= scale
        k_rate *= scale
        bb_rate *= scale
        one_b *= scale
        two_b *= scale
        three_b *= scale



    events = _simulate_hitter_events(
        pa_trials,
        {
            "hr": hr_rate,
            "k": k_rate,
            "bb": bb_rate,
            "1b": one_b,
            "2b": two_b,
            "3b": three_b,
        },
    )


    out = {}


    out["plate_appearances"] = summarize(
        pa_trials,
        "plate_appearances",
        [
            f"expected PA {exp_pa:.2f}"
        ],
    )


    out["hits"] = summarize(
        events["hits"],
        "hits",
        drivers,
    )


    out["home_runs"] = summarize(
        events["hr"],
        "home_runs",
        drivers + [
            f"HR rate {hr_rate:.3f}"
        ],
    )


    out["total_bases"] = summarize(
        events["tb"],
        "total_bases",
        drivers,
    )


    out["walks"] = summarize(
        events["bb"],
        "walks",
        [
            f"BB {bb_rate:.3f}"
        ],
    )


    out["strikeouts"] = summarize(
        events["k"],
        "strikeouts",
        [
            f"K {k_rate:.3f}"
        ],
    )


    # ─────────────────────────────────────────
    # Runs / RBI
    # ─────────────────────────────────────────

    avg_ob = (
        (
            events["hits"][0]
            +
            events["bb"][0]
        )
        /
        max(
            1,
            pa_trials[0]
        )
    )


    run_expectation = (
        _f(
            form.get("exp_runs"),
            0.45,
        )
        *
        offense_mult
        *
        (
            0.75 +
            avg_ob
        )
    )


    rbi_expectation = (
        _f(
            form.get("exp_rbi"),
            0.45,
        )
        *
        offense_mult
        *
        (
            0.75 +
            avg_ob
        )
    )


    out["runs"] = summarize(
        mc.negbinom_count(
            max(
                0.05,
                run_expectation,
            ),
            0.65,
            n,
        ),
        "runs",
        drivers,
    )


    out["rbis"] = summarize(
        mc.negbinom_count(
            max(
                0.05,
                rbi_expectation,
            ),
            0.65,
            n,
        ),
        "rbis",
        drivers,
    )


    # stolen bases

    sb = _f(
        form.get("exp_sb"),
        0.08,
    )

    sb *= (
        0.90 +
        0.20 * offense_mult
    )


    out["stolen_bases"] = summarize(
        mc.negbinom_count(
            max(
                0.01,
                sb,
            ),
            0.25,
            n,
        ),
        "stolen_bases",
        [
            f"SB {sb:.2f}"
        ],
    )


    if use_ensemble:
        _apply_ensemble(
            "mlb",
            out,
            form,
        )


    return out

# ─────────────────────────────────────────────
# Pitcher projection
# ─────────────────────────────────────────────

def project_pitcher(
    form: dict,
    ctx: dict,
    n: int = mc.N_SIMS,
    use_ensemble: bool = True,
) -> dict[str, Projection]:

    park = _park_multiplier(ctx)
    opp_offense = _opponent_offense_multiplier(ctx)

    drivers = [
        f"opp lineup ×{opp_offense:.2f}",
        f"park ×{park:.2f}",
    ]


    # ─────────────────────────────────────────
    # Workload model
    # ─────────────────────────────────────────

    exp_bf = _expected_bf(form)

    exp_outs = _f(
        form.get("exp_outs"),
        17.0,
    )


    bf_trials = mc.trial_counts(
        exp_bf,
        n,
        lo=8,
        hi=45,
    )


    outs_trials = mc.outs_recorded(
        exp_outs,
        sd=4.0,
        n=n,
    )


    # ─────────────────────────────────────────
    # Pitcher skill rates
    # ─────────────────────────────────────────

    k_rate = _safe_rate(
        form.get("p_k"),
        0.235,
    )

    bb_rate = _safe_rate(
        form.get("p_bb"),
        0.075,
    )

    hit_rate = _safe_rate(
        form.get("p_h"),
        0.210,
    )


    # matchup adjustment

    hit_rate *= opp_offense


    bb_rate *= (
        0.90 +
        0.15 * opp_offense
    )


    k_rate *= (
        1.05 -
        0.15 * opp_offense
    )


    k_rate = _clamp(
        k_rate,
        0.05,
        0.45,
    )


    # probability safety

    total = (
        k_rate +
        bb_rate +
        hit_rate
    )

    if total > 0.90:

        scale = (
            0.90 /
            total
        )

        k_rate *= scale
        bb_rate *= scale
        hit_rate *= scale



    # ─────────────────────────────────────────
    # Batter faced event simulation
    # ─────────────────────────────────────────

    strikeouts = []
    walks = []
    hits = []


    for bf in bf_trials:

        k = 0
        bb = 0
        h = 0


        for _ in range(int(bf)):

            r = mc.random_uniform()


            if r < k_rate:

                k += 1


            elif r < k_rate + bb_rate:

                bb += 1


            elif r < (
                k_rate +
                bb_rate +
                hit_rate
            ):

                h += 1



        strikeouts.append(k)
        walks.append(bb)
        hits.append(h)



    out = {}


    out["strikeouts"] = summarize(
        strikeouts,
        "strikeouts",
        [
            f"K {k_rate:.3f}/BF"
        ],
    )


    out["walks_allowed"] = summarize(
        walks,
        "walks_allowed",
        [
            f"BB {bb_rate:.3f}/BF"
        ],
    )


    out["hits_allowed"] = summarize(
        hits,
        "hits_allowed",
        [
            f"H {hit_rate:.3f}/BF"
        ],
    )


    out["outs_recorded"] = summarize(
        outs_trials,
        "outs_recorded",
        [
            f"expected outs {exp_outs:.1f}"
        ],
    )


    out["innings_pitched"] = summarize(
        [
            x / 3
            for x in outs_trials
        ],
        "innings_pitched",
        [
            f"expected IP {exp_outs/3:.1f}"
        ],
    )



    # ─────────────────────────────────────────
    # Run prevention
    # ─────────────────────────────────────────

    xera = _f(
        form.get("xera"),
        4.00,
    )


    expected_ip = (
        exp_outs /
        3.0
    )


    expected_er = (
        xera /
        9.0
        *
        expected_ip
        *
        opp_offense
        *
        park
    )


    out["earned_runs"] = summarize(
        mc.negbinom_count(
            max(
                0.05,
                expected_er,
            ),
            0.75,
            n,
        ),
        "earned_runs",
        drivers + [
            f"xERA {xera:.2f}"
        ],
    )


    # calibrated league relationship

    UNEARNED_RATIO = 1.0904


    out["runs_allowed"] = summarize(
        mc.negbinom_count(
            max(
                0.05,
                expected_er *
                UNEARNED_RATIO,
            ),
            0.75,
            n,
        ),
        "runs_allowed",
        drivers + [
            f"xERA {xera:.2f}",
            f"unearned ×{UNEARNED_RATIO:.3f}",
        ],
    )



    if use_ensemble:

        _apply_ensemble(
            "mlb",
            out,
            form,
        )


    return out



# ─────────────────────────────────────────────
# Ensemble calibration
# ─────────────────────────────────────────────

def _apply_ensemble(
    sport: str,
    projections: dict[str, Projection],
    form: dict,
) -> None:

    """
    Applies XGBoost residual correction.

    The model does NOT replace simulation.
    It only recenters distributions.
    """

    for stat, proj in projections.items():

        key = (
            sport,
            stat,
        )


        model = XGB_CACHE.get(
            key
        )


        if model is None:

            model = ensemble.XGBStatModel(
                sport,
                stat,
            )


            model = (
                model
                if model.train_from_db()
                else False
            )


            XGB_CACHE[key] = model



        if model:

            prediction = model.predict(
                form
            )


            samples, _ = ensemble.blend_distribution(
                proj.samples,
                proj.mean,
                prediction,
                sport,
                stat,
            )


            if (
                prediction is not None
                and
                proj.samples is not None
            ):

                projections[stat] = summarize(
                    samples,
                    stat,
                    proj.drivers +
                    [
                        f"XGB {prediction:.2f}"
                    ],
                )



# ─────────────────────────────────────────────
# Public dispatcher
# ─────────────────────────────────────────────

def project(
    form: dict,
    ctx: dict,
    role: str | None = None,
    **kwargs,
) -> dict[str, Projection]:

    """
    Main MLB projection entry point.
    """

    role = (
        role
        or form.get("role")
        or (
            "pitcher"
            if "exp_bf" in form
            else "batter"
        )
    )


    if role == "pitcher":

        return project_pitcher(
            form,
            ctx,
            **kwargs,
        )


    return project_batter(
        form,
        ctx,
        **kwargs,
    )
