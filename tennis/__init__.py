"""
tennis — ATP + WTA player-prop projection system (inside JUICED).

1v1, no teams. Core is a serve/return point model (per-player serve-points-won /
return-points-won, tour- and surface-specific) wrapped in a Monte-Carlo match
simulator, with a surface-weighted Elo backbone for match-win probability and for
shrinking thin-sample players. Produces distributions for every prop market.

ATP and WTA share the architecture but are fit and stored as SEPARATE parameter
sets (WTA has lower hold rates / more breaks — never share baselines).

Data is behind swappable adapters (data/base.py): a prototype source (Sackmann-
format history + ESPN live) for building/backtesting, and a licensed feed for
production. Sackmann-derived data is CC non-commercial → build/validate only.
"""

TOURS = ("ATP", "WTA")
SURFACES = ("Hard", "Clay", "Grass", "Carpet")
