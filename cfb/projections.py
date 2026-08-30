"""
Public API -- fit the CFB league from real CFBD data, project one player-game to a
distribution per market, and read any prop market off it.

    from cfb import projections as P
    proj = P.project_player("Will Howard", team="Ohio State", position="QB", game=game)
    P.market_prob(proj, "Passing Yards", 245.5, "over")

This module is the only place in the CFB engine that does I/O. Everything under cfb/model/ and
cfb/sim/ is a pure function of already-fetched dataclasses, which is what makes each fit
testable against fixture rows -- the same split nfl/projections.py vs nfl/model/* uses.

THE WINDOW. Two seasons of schedule, efficiency and box scores are fetched (current and
previous), and every league-level fit -- priors, shrinkage strengths, opponent adjustment,
pace, garbage time -- runs on the pooled pair. That is deliberate rather than incidental: in
September the current season carries a handful of games, and a league prior, a per-defense
strength or a blowout-usage slope fitted on it alone would be noise wearing a measurement's
clothes. A third (older) season of box scores is fetched for exactly one purpose, the
year-over-year carryover measurement in model/rates.py, which needs two COMPLETED seasons to
mean anything.

WITHOUT A CFBD_API_KEY every fetch returns [] (see cfb/data/cfbd_client.py), the priors fit
finds nothing to fit, `league_data()` returns None, and every CFB line reaches the board
unprojected -- the same honest degradation WNBA shows with no BALLDONTLIE_API_KEY. Nothing
here substitutes a fallback league.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from fantasy.player_matching import normalize_name

from .data.base import ScheduleGame
from .data.cfbd_client import CFBDClient
from .model import GameContext
from .model import garbage_time as GT
from .model import opponent as OPP
from .model import pace as PACE
from .model import priors as PR
from .model import rates as RT
from .model.config import POWER_CONFERENCES
from .sim import engine as E

_LEAGUE_TTL = 3600.0
_PROJ_TTL = 900.0
_league_cache: dict = {}
_proj_cache: dict = {}
_source = None
_positions_override: Optional[dict] = None


def set_positions_override(mapping: Optional[dict]) -> None:
    """Supply {normalized player name: position} directly instead of reading the canonical
    cfb_players table. Exists so every fit in cfb/model/ can be exercised end to end without a
    database, the same escape hatch nfl.data.espn.set_test_override provides for ESPN. None
    restores the real DB lookup."""
    global _positions_override
    _positions_override = mapping
    clear_cache()


def set_source(source) -> None:
    """Swap the CFBD adapter (tests, or an alternative provider). None restores the live
    CFBDClient."""
    global _source
    _source = source
    clear_cache()


def cfb_source():
    return _source if _source is not None else CFBDClient()


def clear_cache() -> None:
    _league_cache.clear()
    _proj_cache.clear()


def current_season(today: Optional[str] = None) -> int:
    """The CFB season a date belongs to. A season is named for the calendar year it starts in
    and runs into January's bowl/playoff games, so January belongs to the previous season."""
    d = datetime.fromisoformat(today) if today else datetime.now(timezone.utc)
    return d.year - 1 if d.month <= 1 else d.year


# ── league fit ───────────────────────────────────────────────────────────────────────────

def _contexts(schedule: list[ScheduleGame]) -> dict:
    """{(game_id, team): GameContext} for both teams in every game, with the spread flipped
    into each team's own perspective."""
    out: dict = {}
    for g in schedule:
        for team, is_home in ((g.home_team, True), (g.away_team, False)):
            opponent = g.opponent_of(team)
            if not opponent:
                continue
            spread = None
            if g.spread is not None:
                # CFBD publishes the home team's spread (negative = home favoured), so the
                # home team's own "how big a favourite am I" number is its negation.
                spread = -float(g.spread) if is_home else float(g.spread)
            out[(g.id, team)] = GameContext(
                game_id=g.id, season=g.season, week=g.week, team=team, opponent=opponent,
                is_home=is_home,
                opponent_classification=(g.away_classification if is_home
                                         else g.home_classification),
                spread=spread, over_under=g.over_under, margin=g.margin_for(team))
    return out


def _schedule_index(schedule: list[ScheduleGame]) -> dict:
    """{(date, team): game} across the day either side of kickoff -- a Saturday night kickoff
    lands on the following calendar day in UTC, so a +/-1 day window is what makes the lookup
    correct rather than the timezone being guessed at (the same construction
    nfl/board.py::_schedule_index uses)."""
    idx: dict = {}
    for g in schedule:
        if not g.start_date:
            continue
        try:
            day = datetime.fromisoformat(g.start_date.replace("Z", "+00:00")).date()
        except ValueError:
            continue
        for offset in (-1, 0, 1):
            d = (day + timedelta(days=offset)).isoformat()
            for team in (g.home_team, g.away_team):
                idx.setdefault((d, team), g)
    return idx


def _db_positions() -> dict:
    """{normalized player name: position} from the canonical cfb_players table that
    cfb/players_sync.py fills daily. The box-score feed carries no position, so this is the
    only real source of one; an empty table (no roster sync yet) simply means every player
    falls into the pooled prior bucket, which is a real measurement over the whole league
    rather than a guess at anyone's position."""
    if _positions_override is not None:
        return _positions_override
    try:
        import store
        from .repositories import PlayerRepository
        rows = PlayerRepository(store.get_database()).all()
    except Exception as exc:
        print(f"[cfb.projections] canonical player positions unavailable: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return {}
    return {normalize_name(r["full_name"]): (r.get("position") or "")
            for r in rows if r.get("full_name")}


def _merge_totals(a: dict, b: dict) -> dict:
    out = {pid: t for pid, t in a.items()}
    for pid, t in b.items():
        cur = out.get(pid)
        if cur is None:
            out[pid] = t
            continue
        merged = PR.PlayerTotals(player_id=pid, player=cur.player or t.player,
                                 team=cur.team or t.team, games=cur.games + t.games,
                                 weight=cur.weight + t.weight)
        for k in merged.totals:
            merged.totals[k] = cur.totals[k] + t.totals[k]
        out[pid] = merged
    return out


def _team_plays(efficiency: list, context: dict) -> dict:
    """{(game_id, team): offensive plays}. The box-score feed carries no week and the
    efficiency feed carries no game id, so the schedule context is the join between them."""
    by_week_team = {(e.season, e.week, e.team): e.plays for e in efficiency if e.plays}
    out = {}
    for (game_id, team), c in context.items():
        plays = by_week_team.get((c.season, c.week, team))
        if plays:
            out[(game_id, team)] = float(plays)
    return out


def league_data(season: Optional[int] = None) -> Optional[dict]:
    """Every fit the engine needs for one season, fetched once and memoized. None when the
    real data behind a league prior doesn't exist (no API key, or a season CFBD hasn't
    published yet) -- callers must treat that as "CFB is not projectable right now"."""
    season = season or current_season()
    hit = _league_cache.get(season)
    if hit and time.time() - hit[0] < _LEAGUE_TTL:
        return hit[1]

    src = cfb_source()
    schedule = list(src.schedule(season)) + list(src.schedule(season - 1))
    efficiency = list(src.team_efficiency(season)) + list(src.team_efficiency(season - 1))
    current_rows = list(src.player_game_stats(season))
    prior_rows = list(src.player_game_stats(season - 1))
    older_rows = list(src.player_game_stats(season - 2))

    window_rows = current_rows + prior_rows
    if not window_rows:
        # The no-CFBD_API_KEY state (and a season CFBD hasn't published). Short-circuit before
        # the canonical-player lookup so the honest "no data source" case doesn't also emit a
        # database diagnostic that reads like a separate failure.
        _league_cache[season] = (time.time(), None)
        return None

    context = _contexts(schedule)
    team_plays = _team_plays(efficiency, context)

    opponent_model = OPP.fit_opponent_model(efficiency, window_rows, context)

    def production_factor(row):
        ctx = context.get((row.game_id, row.team))
        if ctx is None:
            return 1.0
        return OPP.production_factor(opponent_model, ctx.opponent, ctx.opponent_classification)

    current_totals = PR.accumulate_totals(current_rows, team_plays, production_factor)
    history_totals = PR.accumulate_totals(prior_rows, team_plays, production_factor)
    window_totals = _merge_totals(current_totals, history_totals)

    name_index = {}
    for r in window_rows:
        name_index.setdefault(normalize_name(r.player), r.player_id)
    db_positions = _db_positions()
    positions = {pid: db_positions.get(normalize_name(t.player), "")
                 for pid, t in window_totals.items()}

    priors = PR.fit_positional_priors(window_totals, window_rows, positions)
    if not priors:
        _league_cache[season] = (time.time(), None)
        return None

    classification_of_team, conference_of_team = {}, {}
    for g in schedule:
        if g.home_classification:
            classification_of_team.setdefault(g.home_team, g.home_classification)
        if g.away_classification:
            classification_of_team.setdefault(g.away_team, g.away_classification)
    for t in src.teams(season):
        conference_of_team[t.school] = t.conference
        if t.classification:
            classification_of_team[t.school] = t.classification
    tier_of_team = {
        school: PR.team_tier(conference_of_team.get(school),
                             classification_of_team.get(school), POWER_CONFERENCES)
        for school in set(classification_of_team) | set(conference_of_team)
    }

    level_factors = PR.fit_level_factors(
        history_totals, {pid: tier_of_team.get(t.team) for pid, t in history_totals.items()})

    # The carryover measurement needs the OLDER season's own play counts as well: a per-play
    # usage rate has no denominator without them, and usage is the single most important rate
    # for a returning/transferring player's prior. Both seasons enter unadjusted -- applying
    # the opponent normalisation to only one side of a correlation would bias it.
    older_context = _contexts(src.schedule(season - 2))
    older_plays = _team_plays(src.team_efficiency(season - 2), older_context)
    older_totals = PR.accumulate_totals(older_rows, older_plays)
    prior_unadjusted = PR.accumulate_totals(prior_rows, team_plays)
    carryover = RT.fit_carryover(prior_unadjusted, older_totals)

    prior_name_index = {}
    for r in prior_rows:
        prior_name_index.setdefault(normalize_name(r.player), r.player_id)
    recruiting_prior = PR.fit_recruiting_prior(src.recruiting(season - 1), history_totals,
                                               prior_name_index, positions)
    ratings = {normalize_name(rec.name): rec.rating for rec in src.recruiting(season)
               if rec.rating is not None}

    team_opportunities: dict = {}
    for t in current_totals.values():
        o = t.totals["rush_attempts"] + t.totals["receptions"]
        team_opportunities[t.team] = team_opportunities.get(t.team, 0.0) + o
    team_share = {}
    for pid, t in current_totals.items():
        total = team_opportunities.get(t.team, 0.0)
        if total > 0:
            team_share[pid] = (t.totals["rush_attempts"] + t.totals["receptions"]) / total

    data = {
        "season": season,
        "priors": priors,
        "pace": PACE.fit_pace_model(efficiency, context),
        "opponent": opponent_model,
        "garbage": GT.fit_garbage_time_model(context, window_rows, efficiency),
        "level_factors": level_factors,
        "carryover": carryover,
        "recruiting_prior": recruiting_prior,
        "ratings": ratings,
        "current_totals": current_totals,
        "history_totals": history_totals,
        "positions": positions,
        "name_index": name_index,
        "tier_of_team": tier_of_team,
        "classification_of_team": classification_of_team,
        "context": context,
        "schedule_index": _schedule_index(schedule),
        "team_share": team_share,
    }
    _league_cache[season] = (time.time(), data)
    return data


def find_game(data: dict, team: Optional[str], start_time: Optional[str]) -> Optional[ScheduleGame]:
    if not team or not start_time:
        return None
    day = str(start_time)[:10]
    return data["schedule_index"].get((day, team))


# ── project one player ───────────────────────────────────────────────────────────────────

def project_player(player: str, team: Optional[str] = None, position: Optional[str] = None,
                   game: Optional[ScheduleGame] = None, season: Optional[int] = None,
                   n: Optional[int] = None, rng=None) -> Optional[dict]:
    """One player-game projection, or None when the engine genuinely can't price it: the name
    doesn't resolve to anyone CFBD has a stat line or a recruiting rating for, no positional
    (or pooled) prior exists yet, or the game has no team whose tempo has ever been observed.
    Every one of those is "we don't know", and inventing a projection for it would be worse
    than leaving the line unprojected."""
    data = league_data(season)
    if data is None:
        return None

    nname = normalize_name(player)
    pid = data["name_index"].get(nname)
    current = data["current_totals"].get(pid) if pid else None
    history = data["history_totals"].get(pid) if pid else None
    rating = data["ratings"].get(nname)
    if pid is None and rating is None:
        return None

    ck = (nname, team, position, game.id if game else None, n)
    hit = _proj_cache.get(ck)
    if hit and time.time() - hit[0] < _PROJ_TTL:
        return hit[1]

    team = team or (current.team if current else None) or (history.team if history else None)
    position = (position or data["positions"].get(pid) or "").upper()
    priors = PR.priors_for(position, data["priors"])
    if priors is None:
        return None

    ctx = data["context"].get((game.id, team)) if (game and team) else None
    opponent = ctx.opponent if ctx else None
    pace_proj = PACE.projected_plays(data["pace"], team, opponent, ctx)
    if pace_proj is None:
        return None

    rates = RT.fit_player_rates(
        pid or nname, priors, current, history, player=player, team=team or "",
        position=position, carryover=data["carryover"], level_factors=data["level_factors"],
        origin_tier=data["tier_of_team"].get(history.team) if history else None,
        destination_tier=data["tier_of_team"].get(team) if team else None,
        recruiting_prior=PR.priors_for(position, data["recruiting_prior"]),
        recruiting_rating=rating,
        team_opportunity_share=data["team_share"].get(pid) if pid else None)

    garbage = data["garbage"]
    blowout = GT.blowout_estimate(garbage, ctx)
    starterness = GT.starterness(garbage, rates.team_opportunity_share)
    usage_multiplier = GT.usage_multiplier(
        garbage, blowout.probability if blowout else None, starterness)
    opponent_factor = OPP.production_factor(
        data["opponent"], opponent,
        ctx.opponent_classification if ctx else None)

    sim = E.simulate(rates, pace_proj.plays, pace_proj.sd, priors.unit_sd,
                     garbage_multiplier=usage_multiplier, opponent_factor=opponent_factor,
                     n=n, rng=rng)

    proj = {
        "player": rates.player or player, "team": team, "position": position,
        "season": data["season"], "rates": rates, "pace": pace_proj,
        "proj_kind": rates.tier, "tier_reason": rates.tier_reason,
        "n_games": rates.n_games, "sample_weight": rates.sample_weight,
        "level_factor": rates.level_factor_applied,
        "projected_plays": round(float(pace_proj.plays), 1),
        "pace_basis": pace_proj.basis,
        "blowout_probability": round(blowout.probability, 4) if blowout else None,
        "expected_margin": round(blowout.expected_margin, 1) if blowout else None,
        "blowout_basis": blowout.basis if blowout else None,
        "garbage_time_multiplier": round(float(usage_multiplier), 4),
        "starterness": round(float(starterness), 3),
        "opponent": opponent,
        "opponent_factor": round(float(opponent_factor), 4),
        "opponent_is_bottom_quartile": data["opponent"].is_weak(opponent),
        "usage_share": {k: round(v, 6) for k, v in rates.opportunity.items()},
        "sim": sim,
    }
    _proj_cache[ck] = (time.time(), proj)
    return proj


# ── read markets ─────────────────────────────────────────────────────────────────────────

def market_sample_weight(proj: dict, stat_type: str) -> float:
    """Trust input for ONE market: how much of that market's simulation is the player's own
    data rather than the league prior. Per-market rather than one blended number, because a
    back's rushing projection can be almost entirely his own while his receiving projection is
    almost entirely the prior."""
    return RT.market_sample_weight(proj["rates"], E.MARKET_RATES.get(stat_type, ()))


def market_dist(proj: dict, stat_type: str) -> Optional[np.ndarray]:
    if not E.supports(proj.get("position"), stat_type):
        return None
    return E.market_array(proj["sim"], stat_type)


def market_prob(proj: dict, stat_type: str, line: float, side: str = "over") -> Optional[float]:
    arr = market_dist(proj, stat_type)
    if arr is None:
        return None
    p = E.prob_over(arr, float(line))
    return round(p if side.lower() in ("over", "higher", "yes") else 1.0 - p, 4)


def market_summary(proj: dict, stat_type: str) -> Optional[dict]:
    arr = market_dist(proj, stat_type)
    return E.summary(arr) if arr is not None else None
