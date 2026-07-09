"""
Live API adapters — STUBS. Fill these in with your provider + key, then set the
matching source to `api` in config.yaml. The model only sees the dataclasses in
base.py, so you just map the provider's JSON into those and everything else works.

Suggested providers (per the spec):
  • fixtures + lineups  → API-Football (config keys.api_football)
  • team strength / xG  → Understat, FBref, or API-Football team stats
  • player history      → API-Football player stats (last 12 months + this tournament)
  • odds                → The Odds API (config keys.odds_api)

Keep pulls cached and recency-weight the player/team windows (see model/strength.py
for the decay). Raise a clear error until implemented so it never silently returns
empty data.
"""

from __future__ import annotations

from ..config import load
from .base import (FixtureSource, TeamStrengthSource, PlayerSource, OddsSource)

_HINT = ("wc: `api` source selected but not implemented yet. Implement wc/data/api.py "
         "against your provider (key in config.yaml keys), or set the source back to "
         "`sample`/`csv` in config.yaml.")


class ApiFixtures(FixtureSource):
    def fixtures(self):
        raise NotImplementedError(_HINT + f"  [api_football key set: {bool(load()['keys']['api_football'])}]")


class ApiStrength(TeamStrengthSource):
    def strength(self, team):
        raise NotImplementedError(_HINT)


class ApiPlayers(PlayerSource):
    def players(self, team):
        raise NotImplementedError(_HINT)


class ApiOdds(OddsSource):
    def odds(self, match_id):
        raise NotImplementedError(_HINT + f"  [odds_api key set: {bool(load()['keys']['odds_api'])}]")
