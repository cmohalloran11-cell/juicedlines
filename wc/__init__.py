"""
wc — World Cup 2026 player-prop projection engine (lives inside JUICED).

Pipeline: team goal expectancy (Dixon-Coles) → Monte-Carlo match simulation →
player allocation (goals via xG share, SoT via per-90 scaled to tempo, cards via
fouls/discipline + knockout intensity, GK saves via opponent SoT) → value finder
vs sportsbook odds.

Every data input is behind an adapter interface (`data/base.py`) so a live API,
a CSV, or the built-in sample data can be swapped via config with no model
changes. Surfaced in JUICED through /api/wc/* and the "WC Model" tab.
"""
