"""Source factory — returns the adapters selected in config.yaml `sources`."""

from __future__ import annotations

from ..config import load


def get_sources():
    """(FixtureSource, TeamStrengthSource, PlayerSource, OddsSource) per config."""
    from . import sample, csv as csvmod, api
    sel = load()["sources"]

    def pick(kind, api_cls, csv_cls, sample_cls):
        s = sel.get(kind, "sample")
        return (api_cls() if s == "api" else csv_cls() if s == "csv" else sample_cls())

    return (
        pick("fixtures", api.ApiFixtures, csvmod.CsvFixtures, sample.SampleFixtures),
        pick("strength", api.ApiStrength, csvmod.CsvStrength, sample.SampleStrength),
        pick("players",  api.ApiPlayers,  csvmod.CsvPlayers,  sample.SamplePlayers),
        pick("odds",     api.ApiOdds,     csvmod.CsvOdds,     sample.SampleOdds),
    )
