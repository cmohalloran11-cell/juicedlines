"""
provenance.py — reproducibility metadata for every projection.

Spec Principle 4 (Reproducibility): every prediction must be reproducible, stored with
model_version, feature_version, data_snapshot, simulation_version, and a timestamp. This
module is the single source of truth for those identifiers and the helper that stamps them
onto a projection payload — plus the flat subset persisted to the CLV ledger.

Bump the *_VERSION constants on a DELIBERATE math change so a historical projection stays
attributable to the exact logic that produced it. They are intentionally hand-managed (not
derived from the git sha) so a cosmetic commit doesn't invalidate a model's identity; the
short git sha is recorded alongside them for exact code provenance.
"""
from __future__ import annotations

import functools
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# ── version identifiers (bump on a deliberate change to the corresponding math) ──
# Per-sport model logic (projector/models, analytics resolvers, sport board modules).
MODEL_VERSIONS: dict[str, str] = {
    # 1.1.0 (2026-08): the vendored Monte Carlo engine (projector/models/mlb_model.py) had
    # drifted into a broken state that crashed on every call, silently falling back to the
    # bare empirical model for 100% of live MLB projections. Every graded row before this fix
    # reflects that fallback, not the real engine — bumped so db.py's calibration queries
    # (stat_gammas/prob_calibration/interval_width/stat_biases, all model_version-scoped)
    # can't blend the two models' outcomes into one calibration fit.
    #
    # 1.2.0 (2026-08): two-stage uncertainty propagation. Every stat used to draw its
    # outcome from a rate treated as a FIXED constant across all Monte Carlo trials — only
    # outcome (sampling) variance was represented. Each trial now first draws its own
    # per-stat rate-uncertainty multiplier (Gamma, scaled by how much real evidence backs
    # that rate — form["_eff_pa"]) before the outcome draw: Var(Y) = E[Var(Y|theta)] +
    # Var(E[Y|theta]), not outcome variance alone. Widens intervals for thin samples
    # (call-ups, injury returns) without moving point estimates — approved for production
    # baseline 2026-08 (see backtest.version_history for pre/post comparison once enough
    # games are graded under this version).
    "MLB": "mlb-1.2.0",
    # 1.1.0 (2026-08): fixed tiebreak_prob() -- it averaged two point-win probabilities into
    # one constant instead of solving the real alternating-server sequence, a provably
    # different (not just approximate) process. Shifts tiebreak/match win probability by a
    # real, measurable amount for any asymmetric matchup. Bumped for the same
    # don't-blend-pre/post-fix graded rows reason as MLB above.
    #
    # 1.2.0 (2026-08): two-stage uncertainty propagation (see MLB 1.2.0 comment for the
    # general principle). Each sim now draws its own serve probability from
    # Beta(rate*eff_n, (1-rate)*eff_n) instead of reusing one shared point estimate for the
    # whole match — pulls thin-sample favorites' win probability toward 0.5 relative to the
    # naive point-estimate calculation (a real, expected effect of honest parameter
    # uncertainty on a nonlinear win-probability function). Approved for production
    # baseline 2026-08.
    "Tennis": "tennis-1.2.0",
    # 1.1.0 (2026-08): two-stage uncertainty propagation (same principle as MLB/Tennis
    # above) PLUS real measured pts/reb/ast combo correlation (basketball/model/combo_corr.py
    # — previously combos only correlated incidentally through shared minutes/pace, ~0.05-0.08
    # vs. real box scores' ~0.15-0.47). Note: the WNBA minutes-shrinkage fix
    # (minutes_shrink_games 0->1.5, basketball/config.py) shipped BEFORE this bump and was
    # deliberately NOT version-bumped on its own — it only materially affects thin-sample
    # (debut-game) players, and bumping then would have discarded ~6,790 rows of still-valid
    # calibration history for a narrow-effect change. This bump covers the two materially
    # broader changes above. Approved for production baseline 2026-08.
    "WNBA": "wnba-1.1.0",
    # 1.0.0 (2026-08): NEW engine, not a bump of an existing one — nothing to orphan, no
    # calibration history to blend, so this entry costs nothing to add. Per-snap opportunity x
    # per-attempt efficiency, both empirical-Bayes shrunk toward positional priors measured
    # from the league (config priors and shrinkage k are holdout-fitted on 2022-2025, not
    # chosen), scaled by a sampled playing-time distribution and simulated with two-stage
    # uncertainty. Regular season and preseason are separate models sharing the data layer and
    # the simulator; they are tagged proj_kind "nfl_regular" vs "nfl_preseason" specifically so
    # a calibration query can never blend a projection built on measured snap history with one
    # built on assumed preseason rotation tiers.
    # 1.1.0 (2026-08): preseason playing-time model rebuilt after live production data showed
    # it clustering -- 25 of 34 real preseason board lines projected the literal identical
    # 19.6 expected_snaps (76.5% landed within +-2 of 20), because the old model multiplied
    # ONE flat per-tier snap-SHARE mean by the SAME constant league-baseline team-snaps number
    # for every preseason game (no preseason schedule ever exists to vary it), and two of the
    # six tiers (confirmed_starter, third_team) happened to share that mean (0.30). Replaced
    # with a drives-first model -- expected_snaps = tier_drives_mean(tier) x
    # position_snaps_per_drive(position) -- across 7 tiers instead of 6 (rank-2 depth-chart
    # players now split into first_team_rotation vs second_team on real prior-workload
    # evidence, the same test rank-1 already used). Two independent, tier-AND-position-varying
    # factors instead of one shared table entry structurally cannot collide the old way.
    # Preseason-only: proj_kind already tags nfl_preseason separately from nfl_regular
    # (unaffected by this change), and NFL preseason has only ever been live a few days, so
    # this orphans a negligible amount of graded history. See model/rotation.py's module
    # docstring for the full hierarchy and provenance.MODEL_CHANGELOG for the measured
    # before/after distribution.
    #
    # 1.2.0 (2026-08): fixed a mean-vs-median anchoring bug in board.py's market blend that
    # systematically biased model_prob toward Under. board.py blends the model's own MEAN
    # toward the market line, then shifted the whole simulated array by (blended_mean -
    # raw_mean) to recenter it -- which for a right-skewed stat (Gamma yards -- every
    # yardage market this engine has) sits the array's MEAN on the line while its MEDIAN
    # (what model_prob is actually computed from -- see dataos.validate_direction) sits
    # BELOW it, purely from the shape's own skew. At low trust (common in preseason by
    # design, but not exclusive to it -- any thin-sample regular-season line hits the same
    # path) the market line IS the anchor, so this produced model_prob well under 50% on
    # lines with NO real model/market disagreement at all. Found live 2026-08: 94% of the
    # live preseason board recommending Under, including lines where model_proj sat exactly
    # on the market line. Fixed by blending/shifting on the MEDIAN instead of the mean -- a
    # market line is definitionally the book's implied 50/50 point, so a zero-trust line now
    # correctly reads as a coinflip; "Proj" (still the array's mean, unchanged as the
    # informative headline number) honestly differs from the line by the stat's own skew
    # rather than being forced to equal it. Measured on the exact scenario the bug was found
    # on (nfl/tests/test_board.py's own regression tests): before, a zero-trust 45.5-yard
    # line read model_proj=45.5 (dead on the line) / model_prob=0.381 (a fabricated 38%
    # Under lean); after, model_proj=53.0 (the stat's honest skewed mean) / model_prob=0.500
    # (a correct coinflip). Applies to every NFL line, both season types -- proj_kind still
    # distinguishes nfl_preseason from nfl_regular, unaffected by which fields determine
    # their values.
    "NFL": "nfl-1.2.0",
}
_DEFAULT_MODEL_VERSION = "1.0.0"

# Structured, runtime-queryable mirror of the version-bump comments above — the same
# facts, but in a shape the Model Health dashboard / API can actually read instead of
# a human having to open this file. Oldest first per sport. Pair with
# backtest.version_history(sport) to see whether a given change actually helped:
# that's measured, this is the "what and why" it can't infer from the ledger alone.
MODEL_CHANGELOG: dict[str, list[dict]] = {
    "MLB": [
        {"version": "mlb-1.1.0", "date": "2026-08",
         "summary": "Re-vendored the Monte Carlo engine (mlb_model.py) after it had "
                    "drifted into a broken state that crashed on every call, silently "
                    "falling back to the bare empirical model for 100% of live MLB "
                    "projections."},
        {"version": "mlb-1.2.0", "date": "2026-08",
         "summary": "Two-stage uncertainty propagation: each simulated trial now draws "
                    "its own per-stat rate-uncertainty multiplier (scaled by how much "
                    "real evidence backs that rate) before the outcome draw, instead of "
                    "treating the fitted rate as fixed across every trial. Widens "
                    "intervals for thin samples (call-ups, injury returns) without "
                    "moving point estimates."},
    ],
    "Tennis": [
        {"version": "tennis-1.1.0", "date": "2026-08",
         "summary": "Fixed tiebreak_prob(): it averaged two point-win probabilities into "
                    "one constant instead of solving the real alternating-server "
                    "sequence — a provably different process, not an approximation."},
        {"version": "tennis-1.2.0", "date": "2026-08",
         "summary": "Two-stage uncertainty propagation: each simulated match now draws "
                    "its own serve probability from a Beta distribution around the "
                    "fitted rate instead of reusing one shared point estimate for the "
                    "whole match. Pulls thin-sample favorites' win probability toward "
                    "0.5 relative to the naive point-estimate calculation."},
    ],
    "WNBA": [
        {"version": "wnba-1.1.0", "date": "2026-08",
         "summary": "Two-stage uncertainty propagation (same principle as MLB/Tennis) "
                    "plus real measured pts/reb/ast combo correlation (previously combos "
                    "only correlated incidentally through shared minutes/pace, "
                    "~0.05-0.08 vs. real box scores' ~0.15-0.47). Note: the WNBA minutes-"
                    "shrinkage fix (debut-game players no longer projected with zero "
                    "regression to the mean) shipped before this version bump and was "
                    "deliberately left unbumped on its own -- narrow effect, didn't "
                    "justify discarding ~6,790 rows of still-valid calibration history."},
    ],
    "NFL": [
        {"version": "nfl-1.0.0", "date": "2026-08",
         "summary": "Initial NFL engine (regular season + preseason). Opportunity and "
                    "efficiency are fitted separately -- per-snap carries/targets/pass "
                    "attempts, and per-attempt yards/catch rate/completion rate -- each "
                    "empirical-Bayes shrunk toward a positional prior measured across the "
                    "league, with the shrinkage strength holdout-fitted on 2022-2025 nflverse "
                    "data rather than chosen. Playing time is a sampled Beta distribution "
                    "whose concentration scales with how much real snap history exists. "
                    "Preseason is a SEPARATE model: rotation tiers from the depth chart drive "
                    "playing time, because no data source publishes preseason snap counts -- "
                    "those tier shapes are labelled assumptions, preseason confidence is "
                    "structurally low, and the board market-anchors accordingly. Simulation "
                    "is two-stage (per-trial parameter draw before the outcome draw) with "
                    "Poisson counts, beta-binomial conversions and Gamma yards -- each family "
                    "chosen against a measured fit, not defaulted to Normal. Tagged "
                    "proj_kind nfl_regular / nfl_preseason so the two never blend in a "
                    "calibration query."},
        {"version": "nfl-1.1.0", "date": "2026-08",
         "summary": "Rebuilt the preseason playing-time model after live production data "
                    "showed it clustering: 25 of 34 real preseason board lines projected the "
                    "literal identical 19.6 expected_snaps (76.5% within +-2 of 20), because "
                    "the old model multiplied one flat per-tier snap-share mean by the same "
                    "constant league-baseline team-snaps number for every preseason game "
                    "(no preseason schedule ever exists to vary it), and two of the six tiers "
                    "happened to share that mean. New model: expected_snaps = "
                    "tier_drives_mean(tier) x position_snaps_per_drive(position), across 7 "
                    "tiers instead of 6 (rank-2 depth-chart players now split into "
                    "first_team_rotation vs second_team on real prior-workload evidence). "
                    "Measured on a synthetic population spanning every tier x position "
                    "(nfl/tests/test_preseason_snap_model.py): 19 distinct expected_snaps "
                    "values vs the old model's 3, 14.3% within +-2 of 20 vs 76.5%. "
                    "Regular-season projections are untouched by this change."},
        {"version": "nfl-1.2.0", "date": "2026-08",
         "summary": "Fixed a mean-vs-median anchoring bug in board.py's market blend: it "
                    "recentered the simulated array so its MEAN sat on the market line but "
                    "left its MEDIAN (what model_prob is actually computed from) sitting "
                    "below it for any right-skewed stat (every yardage market this engine "
                    "has), because a right-skewed distribution's mean always sits above its "
                    "median. At low trust -- common in preseason by design, but not "
                    "exclusive to it -- the market line IS the anchor, so this produced "
                    "model_prob well under 50% on lines with no real model/market "
                    "disagreement at all. Found live 2026-08: 94% of the live preseason "
                    "board recommending Under, including lines where model_proj sat exactly "
                    "on the market line. Fixed by blending/shifting on the MEDIAN instead of "
                    "the mean -- a market line is definitionally the book's implied 50/50 "
                    "point. Measured on the exact scenario the bug was found on "
                    "(nfl/tests/test_board.py): before, a zero-trust 45.5-yard line read "
                    "model_proj=45.5 / model_prob=0.381 (a fabricated 38% Under lean); "
                    "after, model_proj=53.0 (the stat's honest skewed mean) / "
                    "model_prob=0.500 (a correct coinflip). Applies to every NFL line, both "
                    "season types."},
    ],
}

FEATURE_VERSION = "1.0.0"            # feature pipeline (projector/features + projector_bridge)
SIMULATION_VERSION = "montecarlo-1.0.0"   # projector/models/montecarlo
CALIBRATION_VERSION = "1.0.0"       # the honesty stack shape (gammas / Platt P(over) / width)
SCHEMA_VERSION = "1.0.0"            # projection output contract (payload keys)

_ROOT = Path(__file__).parent


@functools.lru_cache(maxsize=1)
def code_sha() -> str | None:
    """Short git sha of the deployed code, or None outside a git checkout. Cached."""
    try:
        out = subprocess.run(
            ["git", "-C", str(_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3)
        return out.stdout.strip() or None
    except Exception:
        return None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def model_version(sport: str | None) -> str:
    return MODEL_VERSIONS.get(sport or "", _DEFAULT_MODEL_VERSION)


def changelog(sport: str | None = None) -> dict:
    """The structured version changelog — one sport, or all. Newest entry last (same
    order as MODEL_CHANGELOG)."""
    if sport:
        return {sport: MODEL_CHANGELOG.get(sport, [])}
    return dict(MODEL_CHANGELOG)


def versions() -> dict:
    """The current version set — served by /api/version for the transparency principle."""
    return {
        "model_versions": dict(MODEL_VERSIONS),
        "feature_version": FEATURE_VERSION,
        "simulation_version": SIMULATION_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "schema_version": SCHEMA_VERSION,
        "code_sha": code_sha(),
    }


def base(*, sport: str | None = None, method: str | None = None,
         sims: int | None = None, n: int | None = None,
         data_snapshot: str | None = None, calibration: str | None = None) -> dict:
    """Build the provenance object attached to a single projection."""
    p = {
        "model_version": model_version(sport),
        "feature_version": FEATURE_VERSION,
        "simulation_version": SIMULATION_VERSION,
        "calibration_version": calibration or CALIBRATION_VERSION,
        "schema_version": SCHEMA_VERSION,
        "code_sha": code_sha(),
        "generated_at": now_iso(),
    }
    if method is not None:
        p["method"] = method
    if sims is not None:
        p["sims"] = int(sims)
    if n is not None:
        p["n"] = int(n)
    if data_snapshot is not None:
        p["data_snapshot"] = data_snapshot
    return p


def _board_sims() -> int | None:
    """Monte-Carlo sim count the board runs (from the engine bridge), if importable."""
    try:
        import projector_bridge
        return int(getattr(projector_bridge, "_BOARD_SIMS", 0)) or None
    except Exception:
        return None


def stamp_lines(lines: list[dict], *, data_snapshot: str | None = None) -> int:
    """
    Attach a `provenance` object to every line that carries a model projection, and mirror
    the ledger-persisted subset (model_version / feature_version / data_snapshot) as flat
    fields so db.log_clv can store them. Idempotent. Returns the number stamped.

    `data_snapshot` is the slate/UTC date the underlying data belongs to (defaults to today).
    """
    ds = (data_snapshot or now_iso())[:10]
    sims_engine = _board_sims()
    stamped = 0
    for l in lines:
        if l.get("model_proj") is None:
            continue
        method = l.get("proj_kind")
        prov = base(
            sport=l.get("sport"),
            method=method,
            sims=sims_engine if method == "engine" else None,
            n=l.get("model_n"),
            data_snapshot=ds,
        )
        l["provenance"] = prov
        # flat fields for the ledger (see db.log_clv)
        l["model_version"] = prov["model_version"]
        l["feature_version"] = prov["feature_version"]
        l["data_snapshot"] = ds
        stamped += 1
    return stamped
