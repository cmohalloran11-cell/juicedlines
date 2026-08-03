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
    "MLB": "mlb-1.1.0",
    # 1.1.0 (2026-08): fixed tiebreak_prob() -- it averaged two point-win probabilities into
    # one constant instead of solving the real alternating-server sequence, a provably
    # different (not just approximate) process. Shifts tiebreak/match win probability by a
    # real, measurable amount for any asymmetric matchup. Bumped for the same
    # don't-blend-pre/post-fix graded rows reason as MLB above.
    "Tennis": "tennis-1.1.0",
    "WNBA": "wnba-1.0.0",
}
_DEFAULT_MODEL_VERSION = "1.0.0"

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
