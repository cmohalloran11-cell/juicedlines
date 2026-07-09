"""
Orchestrator — data → ratings → simulation → projections table + value finder.

    from wc import projections
    projections.matches()          # upcoming fixtures
    projections.run()              # full projections + value for every fixture
    projections.run("wc-qf1")      # one fixture
    projections.to_csv(results) / projections.to_json(results)

Everything runs on whatever sources config.yaml selects (sample by default).
"""

from __future__ import annotations

import csv as _csv
import io
import json

import numpy as np

from .data import get_sources
from .model.strength import ratings
from .sim.engine import simulate
from .value.finder import find_value


def _sources():
    return get_sources()


def matches() -> list[dict]:
    fx_src, *_ = _sources()
    return [{"id": f.id, "home": f.home, "away": f.away, "date": f.date,
             "stage": f.stage, "knockout": f.knockout, "rivalry": f.rivalry}
            for f in fx_src.fixtures()]


def _projection_rows(sim: dict) -> list[dict]:
    """Flatten per-player market probabilities into a browsable table."""
    rows = []
    for name, e in sim["players"].items():
        base = {"player": name, "team": e["team"], "pos": e["pos"],
                "confidence": e["confidence"]}
        if "goal" in e:
            rows.append({**base, "market": "Anytime Goal",
                         "prob": e["goal"]["anytime"], "exp": e["goal"]["exp"]})
            if e["goal"]["two_plus"] >= 0.02:
                rows.append({**base, "market": "2+ Goals",
                             "prob": e["goal"]["two_plus"], "exp": e["goal"]["exp"]})
            for l, p in e["sot"]["over"].items():
                rows.append({**base, "market": f"Shots on Target {l}+",
                             "prob": p, "exp": e["sot"]["exp"]})
            rows.append({**base, "market": "Anytime Card", "prob": e["card"]["yes"], "exp": None})
        if "saves" in e:
            for l, p in e["saves"]["over"].items():
                rows.append({**base, "market": f"Saves Over {l}",
                             "prob": p, "exp": e["saves"]["exp"]})
    rows.sort(key=lambda r: r["prob"], reverse=True)
    return rows


def run(fixture_id: str | None = None, seed: int = 42) -> list[dict]:
    fx_src, st_src, pl_src, od_src = _sources()
    fixtures = [f for f in fx_src.fixtures() if not fixture_id or f.id == fixture_id]

    teams = {t for f in fixtures for t in (f.home, f.away)}
    rt = ratings([s for s in (st_src.strength(t) for t in teams) if s])

    rng = np.random.default_rng(seed)      # seeded → reproducible demo
    out = []
    for f in fixtures:
        sim = simulate(f, rt, pl_src.players(f.home), pl_src.players(f.away),
                       st_src.strength(f.home), st_src.strength(f.away), rng)
        out.append({
            "fixture": {"id": f.id, "home": f.home, "away": f.away, "date": f.date,
                        "stage": f.stage, "knockout": f.knockout, "rivalry": f.rivalry},
            "meta": {"lambda_home": sim["lambda_home"], "mu_away": sim["mu_away"],
                     "exp_total_goals": sim["exp_total_goals"],
                     "card_intensity": sim["card_intensity"]},
            "projections": _projection_rows(sim),
            "value": find_value(sim, od_src.odds(f.id)),
        })
    return out


# ── export ────────────────────────────────────────────────────────────────────

def to_json(results: list[dict]) -> str:
    return json.dumps(results, indent=2)


def to_csv(results: list[dict]) -> str:
    buf = io.StringIO()
    w = _csv.writer(buf)
    w.writerow(["fixture", "player", "team", "pos", "market", "prob", "confidence"])
    for r in results:
        fid = r["fixture"]["id"]
        for p in r["projections"]:
            w.writerow([fid, p["player"], p["team"], p["pos"], p["market"],
                        p["prob"], p["confidence"]])
    return buf.getvalue()
