"""
Built-in sample data — a realistic WC 2026 knockout slate so the whole pipeline
runs end-to-end with no API keys. Swap for real sources via config.yaml.

Two quarterfinals, four rosters (stars + GK), plausible per-90 rates, and a set
of sportsbook odds to exercise the value finder.
"""

from __future__ import annotations

from .base import (Fixture, TeamStrength, Player, OddsLine,
                   FixtureSource, TeamStrengthSource, PlayerSource, OddsSource)

_FIXTURES = [
    Fixture("wc-qf1", "Brazil", "England", "2026-07-11", stage="qf",
            neutral=True, knockout=True, host_home=None),
    Fixture("wc-qf2", "Argentina", "France", "2026-07-11", stage="qf",
            neutral=True, knockout=True, rivalry=True, host_home=None),
]

_STRENGTH = {
    "Brazil":    TeamStrength("Brazil",    2.15, 0.75, 2.05, 0.85, 15.4, 0.58),
    "England":   TeamStrength("England",   1.85, 0.80, 1.75, 0.95, 14.1, 0.56),
    "Argentina": TeamStrength("Argentina", 1.95, 0.70, 1.80, 0.80, 13.8, 0.55),
    "France":    TeamStrength("France",    2.05, 0.85, 1.95, 0.90, 14.9, 0.54),
}

# name, pos, minutes, shots90, sot90, xg90, xg_share, fouls90, yellow90, red90, save_pct, start_prob
_ROSTERS = {
    "Brazil": [
        ("Vinicius Junior", "FW", 1350, 3.6, 1.35, 0.52, 0.30, 1.5, 0.28, 0.01, 0, 0.95),
        ("Rodrygo",         "FW", 1180, 2.6, 0.95, 0.34, 0.18, 0.9, 0.14, 0.00, 0, 0.86),
        ("Raphinha",        "FW", 1220, 2.9, 1.05, 0.36, 0.20, 1.2, 0.22, 0.00, 0, 0.88),
        ("Bruno Guimaraes", "MF", 1400, 1.2, 0.40, 0.12, 0.07, 2.1, 0.34, 0.02, 0, 0.90),
        ("Marquinhos",      "DF", 1440, 0.6, 0.22, 0.06, 0.04, 1.0, 0.18, 0.01, 0, 0.92),
        ("Alisson",         "GK", 1440, 0.0, 0.00, 0.00, 0.00, 0.1, 0.03, 0.00, 0.74, 0.97),
    ],
    "England": [
        ("Harry Kane",      "FW", 1410, 3.9, 1.55, 0.58, 0.32, 0.8, 0.12, 0.00, 0, 0.96),
        ("Bukayo Saka",     "FW", 1300, 2.8, 1.00, 0.34, 0.18, 1.0, 0.16, 0.00, 0, 0.90),
        ("Phil Foden",      "MF", 1260, 2.4, 0.90, 0.30, 0.16, 0.9, 0.14, 0.00, 0, 0.84),
        ("Jude Bellingham", "MF", 1380, 2.0, 0.75, 0.26, 0.14, 1.6, 0.30, 0.02, 0, 0.92),
        ("John Stones",     "DF", 1350, 0.5, 0.18, 0.05, 0.03, 0.8, 0.14, 0.00, 0, 0.80),
        ("Jordan Pickford", "GK", 1440, 0.0, 0.00, 0.00, 0.00, 0.1, 0.02, 0.00, 0.71, 0.98),
    ],
    "Argentina": [
        ("Lautaro Martinez", "FW", 1240, 3.4, 1.30, 0.50, 0.27, 1.1, 0.18, 0.00, 0, 0.88),
        ("Julian Alvarez",   "FW", 1300, 2.9, 1.05, 0.40, 0.22, 1.3, 0.20, 0.00, 0, 0.90),
        ("Lionel Messi",     "FW", 1180, 3.1, 1.15, 0.42, 0.24, 0.6, 0.08, 0.00, 0, 0.82),
        ("Enzo Fernandez",   "MF", 1400, 1.4, 0.50, 0.13, 0.08, 1.8, 0.32, 0.02, 0, 0.91),
        ("Cristian Romero",  "DF", 1370, 0.7, 0.25, 0.06, 0.04, 1.7, 0.36, 0.04, 0, 0.90),
        ("Emiliano Martinez","GK", 1440, 0.0, 0.00, 0.00, 0.00, 0.2, 0.06, 0.00, 0.76, 0.97),
    ],
    "France": [
        ("Kylian Mbappe",     "FW", 1380, 4.1, 1.60, 0.62, 0.34, 0.9, 0.12, 0.00, 0, 0.97),
        ("Ousmane Dembele",   "FW", 1210, 2.7, 0.95, 0.32, 0.17, 1.1, 0.18, 0.00, 0, 0.85),
        ("Marcus Thuram",     "FW", 1150, 2.5, 0.90, 0.34, 0.18, 1.4, 0.20, 0.00, 0, 0.80),
        ("Aurelien Tchouameni","MF", 1390, 1.0, 0.32, 0.08, 0.05, 2.0, 0.34, 0.03, 0, 0.90),
        ("William Saliba",    "DF", 1420, 0.5, 0.18, 0.05, 0.03, 0.9, 0.16, 0.01, 0, 0.91),
        ("Mike Maignan",      "GK", 1440, 0.0, 0.00, 0.00, 0.00, 0.1, 0.03, 0.00, 0.73, 0.96),
    ],
}

# Sample sportsbook odds (American) to exercise the value finder.
_ODDS = {
    "wc-qf1": [
        OddsLine("wc-qf1", "Vinicius Junior", "goal", None, "yes", 155),
        OddsLine("wc-qf1", "Harry Kane",      "goal", None, "yes", 130),
        OddsLine("wc-qf1", "Rodrygo",         "goal", None, "yes", 320),
        OddsLine("wc-qf1", "Harry Kane",      "sot",  1.5,  "over", -105),
        OddsLine("wc-qf1", "Vinicius Junior", "sot",  1.5,  "over", 120),
        OddsLine("wc-qf1", "Bruno Guimaraes", "card", 0.5,  "yes", 210),
        OddsLine("wc-qf1", "Jude Bellingham", "card", 0.5,  "yes", 240),
        OddsLine("wc-qf1", "Alisson",         "saves", 2.5, "over", -115),
        OddsLine("wc-qf1", "Jordan Pickford", "saves", 3.5, "over", 140),
    ],
    "wc-qf2": [
        OddsLine("wc-qf2", "Kylian Mbappe",    "goal", None, "yes", 115),
        OddsLine("wc-qf2", "Lautaro Martinez", "goal", None, "yes", 170),
        OddsLine("wc-qf2", "Lionel Messi",     "goal", None, "yes", 240),
        OddsLine("wc-qf2", "Kylian Mbappe",    "sot",  1.5,  "over", -130),
        OddsLine("wc-qf2", "Cristian Romero",  "card", 0.5,  "yes", 165),
        OddsLine("wc-qf2", "Emiliano Martinez","saves", 3.5, "over", 135),
    ],
}


def _mk_player(team, row) -> Player:
    (name, pos, minutes, s90, sot90, xg90, share, f90, y90, r90, sv, sp) = row
    return Player(name=name, team=team, position=pos, minutes=minutes, shots90=s90,
                  sot90=sot90, xg90=xg90, xg_share=share, fouls90=f90, yellow90=y90,
                  red90=r90, save_pct=sv, start_prob=sp)


class SampleFixtures(FixtureSource):
    def fixtures(self): return list(_FIXTURES)


class SampleStrength(TeamStrengthSource):
    def strength(self, team): return _STRENGTH.get(team)


class SamplePlayers(PlayerSource):
    def players(self, team): return [_mk_player(team, r) for r in _ROSTERS.get(team, [])]


class SampleOdds(OddsSource):
    def odds(self, match_id): return list(_ODDS.get(match_id, []))
