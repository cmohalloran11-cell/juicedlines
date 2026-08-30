"""
cfb.player_matching — resolves a source's raw player record to our canonical cfb_players id.

Same pipeline as fantasy.player_matching (this is the exact same problem -- reconcile a raw
name from an external source against a canonical table -- already solved once in this
codebase, reused here rather than reinvented): exact source-id mapping first, then a
difflib-based fuzzy match on normalized name (+ position/team tie-breakers), and anything
below MATCH_THRESHOLD goes to cfb_unmatched_players for human review instead of being
auto-linked. `normalize_name`/`find_best_match` are imported directly from
fantasy.player_matching -- they're pure functions over plain dicts (full_name/position/team
keys) with no fantasy-specific dependency, so importing is a real reuse, not a rebuild.

Odds API note: the-odds-api's player-prop outcomes carry only a player display NAME (no
team, no stable numeric id -- see cfb/data/odds_provider.py). That means Odds-API-sourced
matches are name-only (no team tie-breaker bonus), which is a genuinely thinner signal than
CFBD's athlete-id-backed roster matches; MATCH_THRESHOLD is kept at fantasy's proven 0.88
rather than loosened for this, so a low-confidence match still goes to review instead of
being silently auto-linked.
"""
from __future__ import annotations

from typing import Optional

from store.database import Database
from fantasy.player_matching import normalize_name, find_best_match, MATCH_THRESHOLD
from .repositories import PlayerMappingRepository, PlayerRepository, UnmatchedPlayerRepository

__all__ = ["normalize_name", "find_best_match", "MATCH_THRESHOLD", "resolve_or_log"]


def resolve_or_log(db: Database, source: str, source_id: str, raw_name: str,
                   position: Optional[str] = None, team: Optional[str] = None) -> Optional[str]:
    """Full resolution pipeline for one source record: exact mapping -> fuzzy match -> review
    log. Returns the canonical player_id if resolved (exactly, or a high-confidence fuzzy
    match that gets auto-linked), else None -- an unmatched-review row was written instead."""
    mapper = PlayerMappingRepository(db)
    existing = mapper.resolve(source, source_id)
    if existing:
        return existing

    players = PlayerRepository(db)
    candidates = players.find_by_position(position) if position else players.all()
    best, score = find_best_match(raw_name, position, team, candidates)

    if best and score >= MATCH_THRESHOLD:
        mapper.map(best["id"], source, source_id, confidence=round(score, 3))
        return best["id"]

    UnmatchedPlayerRepository(db).log(
        source=source, source_id=source_id, raw_name=raw_name, raw_team=team,
        raw_position=position,
        best_guess_player_id=best["id"] if best else None,
        best_guess_score=round(score, 3) if best else None)
    return None
