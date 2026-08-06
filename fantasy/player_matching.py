"""
fantasy.player_matching — resolves a source's raw player record to our canonical player_id.

Exact path: if the source already has a fantasy_player_ids row for this source_id, use it.
Fallback path: fuzzy-match on normalized name (+ position/team as tie-breakers) against the
canonical table using stdlib difflib (no new dependency for phase 1). Anything below
MATCH_THRESHOLD is NOT auto-linked -- it's logged to fantasy_unmatched_players for human
review instead, per the task's explicit requirement. Silently mis-mapping a player is worse
than surfacing "unmatched."
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Optional

from store.database import Database
from .repositories import PlayerMappingRepository, PlayerRepository, UnmatchedPlayerRepository

MATCH_THRESHOLD = 0.88  # exact-ish only; anything softer goes to the review log, not auto-linked

_SUFFIXES = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
_PUNCT = re.compile(r"[.'’]")
_SPACES = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    name = _PUNCT.sub("", name.lower().strip())
    name = _SUFFIXES.sub("", name)
    return _SPACES.sub(" ", name).strip()


def find_best_match(raw_name: str, position: Optional[str], team: Optional[str],
                    candidates: list[dict]) -> tuple[Optional[dict], float]:
    """Best fuzzy candidate among `candidates` (canonical player rows), plus its score. Pure
    function -- callers fetch candidates (typically pre-filtered by position) themselves."""
    target = normalize_name(raw_name)
    best: Optional[dict] = None
    best_score = 0.0
    for cand in candidates:
        score = SequenceMatcher(None, target, normalize_name(cand["full_name"])).ratio()
        if position and cand.get("position") and cand["position"].upper() == position.upper():
            score += 0.03
        if team and cand.get("team") and cand["team"].upper() == team.upper():
            score += 0.02
        if score > best_score:
            best, best_score = cand, score
    return best, min(best_score, 1.0)


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
