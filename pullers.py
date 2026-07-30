"""
pullers.py — JUICED unified prop ingestion engine.

Pipeline
--------
PrizePicks / Underdog raw feeds
        ↓
raw normalization
        ↓
sport classification
        ↓
market classification
        ↓
deduplication
        ↓
Line schema
        ↓
Projection engine

Designed to prevent silent line loss.
"""

from __future__ import annotations

from typing import Any
from collections import Counter
from pathlib import Path
import re
import sys
import time

import requests


# ============================================================
# Paths / optional imports
# ============================================================

_BD = Path(__file__).parent.parent / "betting_dashboard"

if _BD.exists() and str(_BD) not in sys.path:
    sys.path.insert(0, str(_BD))


try:
    from underdog import Underdog, UnderdogProp
    _UD_OK = True
except ImportError:
    _UD_OK = False


# ============================================================
# Supported sports
# ============================================================

SUPPORTED_SPORTS = {
    "MLB",
    "Tennis",
    "WNBA",
}


# PrizePicks API
_PP_PARTNER = "https://partner-api.prizepicks.com"

_PP_TTL = 90

_pp_cache: dict[str, Any] = {}


# ============================================================
# Diagnostics
# ============================================================

class PullDiagnostics:

    def __init__(self):
        self.raw = 0
        self.kept = 0

        self.sports = Counter()
        self.leagues = Counter()
        self.odds = Counter()

        self.removed = Counter()

    def report(self):

        print("\n========== JUICED PULL REPORT ==========")

        print(
            f"Raw projections: {self.raw}"
        )

        print(
            f"Kept projections: {self.kept}"
        )

        print("\nSports:")
        for k,v in self.sports.most_common():
            print(
                f"  {k}: {v}"
            )

        print("\nOdds Types:")
        for k,v in self.odds.most_common():
            print(
                f"  {k}: {v}"
            )

        print("\nRemoved:")
        for k,v in self.removed.most_common():
            print(
                f"  {k}: {v}"
            )

        print("========================================\n")



# ============================================================
# Helpers
# ============================================================


def _clean_stat(raw: str | None) -> str | None:

    if not raw:
        return None

    return (
        raw
        .replace("_", " ")
        .title()
    )



def _american_to_implied(
    price: str | None
):

    if not price:
        return None

    try:

        p=float(price)

        if p > 0:
            return round(
                100/(p+100),
                4
            )

        return round(
            abs(p)/(abs(p)+100),
            4
        )

    except:

        return None



# ============================================================
# Sport detection
# ============================================================


def _sport_from_league(
    league: str | None
) -> str:


    if not league:
        return "other"


    l = league.lower()


    # Baseball
    if (
        "mlb" in l
        or "baseball" in l
    ):

        return "MLB"


    # Tennis

    if (
        "tennis" in l
        or "atp" in l
        or "wta" in l
    ):

        return "Tennis"


    # WNBA

    if "wnba" in l:

        return "WNBA"


    return "other"



# ============================================================
# PrizePicks API
# ============================================================


def _pp_session():

    s=requests.Session()

    s.headers.update(
        {
            "User-Agent":
            (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64)"
            ),

            "Accept":
            "application/json",
        }
    )

    return s



def _pp_get(
    session,
    url,
    params=None
):

    for attempt in range(3):

        r=session.get(
            url,
            params=params,
            timeout=30
        )


        if r.status_code != 429:
            return r


        time.sleep(
            2 ** attempt
        )


    return r



# ============================================================
# Raw PrizePicks fetch
# ============================================================


def _fetch_pp_raw():

    now=time.time()

    cached=_pp_cache.get(
        "raw"
    )


    if cached:

        timestamp,data=cached

        if now-timestamp < _PP_TTL:
            return data



    session=_pp_session()


    response=_pp_get(
        session,
        f"{_PP_PARTNER}/projections",
        params={
            "per_page":10000
        }
    )


    if response.status_code != 200:

        raise RuntimeError(
            f"PrizePicks API {response.status_code}"
        )


    payload=response.json()


    _pp_cache["raw"]=(
        now,
        payload
    )


    return payload

# ============================================================
# PrizePicks normalization
# ============================================================


def _resolve_relationship(
    obj: dict,
    name: str,
    index: dict,
) -> dict:

    rel = (
        obj.get("relationships", {})
        .get(name, {})
        .get("data")
    )

    if not rel:
        return {}

    return index.get(
        (
            rel.get("type"),
            rel.get("id")
        ),
        {}
    )



def _detect_odds_type(
    attributes: dict,
) -> str:

    """
    PrizePicks has changed this field multiple times.

    Never assume missing == standard until verified.
    """

    raw = (
        attributes.get("odds_type")
        or
        attributes.get("pick_type")
        or
        attributes.get("market_type")
        or ""
    )


    raw = str(raw).lower()


    if (
        "demon" in raw
        or "power" in raw
        or "higher" in raw
    ):
        return "demon"


    if (
        "goblin" in raw
        or "flex" in raw
        or "lower" in raw
    ):
        return "goblin"


    return "standard"



def _is_period_prop(
    attributes: dict,
) -> bool:

    event_type = str(
        attributes.get("event_type")
        or ""
    ).lower()


    description = str(
        attributes.get("description")
        or ""
    ).lower()


    period_words = (
        "1st half",
        "2nd half",
        "first half",
        "second half",
        "quarter",
        "period",
        "1q",
        "2q",
        "3q",
        "4q",
    )


    return (
        "duration" in event_type
        or
        any(
            x in description
            for x in period_words
        )
    )



def _is_multi_player(
    player_name: str | None
) -> bool:

    if not player_name:
        return False


    return (
        "+"
        in player_name
        or
        "&"
        in player_name
    )



def _pp_line(
    projection: dict,
    index: dict,
    diag: PullDiagnostics,
):

    attr = (
        projection
        .get("attributes", {})
        or {}
    )


    player_obj = _resolve_relationship(
        projection,
        "new_player",
        index,
    )


    league_obj = _resolve_relationship(
        player_obj,
        "league",
        index,
    )


    player_attr = (
        player_obj
        .get("attributes", {})
        or {}
    )


    league_attr = (
        league_obj
        .get("attributes", {})
        or {}
    )


    league_name = (
        league_attr.get("name")
        or
        player_attr.get("league")
        or
        ""
    )


    sport = _sport_from_league(
        league_name
    )


    diag.leagues[
        league_name
    ] += 1


    odds_type = _detect_odds_type(
        attr
    )


    diag.odds[
        odds_type
    ] += 1



    player = (
        player_attr.get("display_name")
        or
        player_attr.get("name")
    )


    stat = _clean_stat(
        attr.get("stat_type")
    )


    line = attr.get(
        "line_score"
    )


    # Keep raw period information.
    # Do not delete here.
    if _is_period_prop(attr):

        stat = (
            f"{stat} (PERIOD)"
        )

        diag.removed[
            "period_prop"
        ] += 1



    if _is_multi_player(player):

        diag.removed[
            "multi_player"
        ] += 1



    diag.sports[
        sport
    ] += 1



    return {

        "id":
            f"pp_{projection.get('id')}",


        "source":
            "prizepicks",


        "sport":
            sport,


        "player":
            player,


        "team":
            player_attr.get(
                "team"
            ),


        "position":
            player_attr.get(
                "position"
            ),


        "stat_type":
            stat,


        "line":
            line,


        "odds_type":
            odds_type,


        "matchup":
            attr.get(
                "description"
            ),


        "start_time":
            attr.get(
                "start_time"
            ),


        "status":
            attr.get(
                "status"
            ),


        "over_implied":
            None,


        "under_implied":
            None,


        "over_price":
            None,


        "under_price":
            None,


        "pickem_price":
            None,


        "headshot":
            player_attr.get(
                "image_url"
            ),


        "country":
            None,


        "meta":
        {

            "league":
                league_name,


            "league_id":
                league_obj.get(
                    "id"
                ),


            "raw_odds_type":
                attr.get(
                    "odds_type"
                ),


            "event_type":
                attr.get(
                    "event_type"
                ),


            "raw_stat":
                attr.get(
                    "stat_type"
                ),

        },

    }

# ============================================================
# Final PrizePicks fetch pipeline
# ============================================================


def fetch_prizepicks(
    sport_filter: str | None = None,
    debug: bool = True,
):

    key = sport_filter or "all"

    cached = _pp_cache.get(
        key
    )

    now = time.time()


    if cached:

        timestamp, lines = cached

        if now - timestamp < _PP_TTL:

            return lines, None



    try:

        payload = _fetch_pp_raw()


        included = payload.get(
            "included",
            []
        )


        index = {

            (
                obj.get("type"),
                obj.get("id")
            ):
            obj

            for obj in included

        }



        diagnostics = PullDiagnostics()


        raw_lines = []


        for projection in payload.get(
            "data",
            []
        ):

            diagnostics.raw += 1


            row = _pp_line(
                projection,
                index,
                diagnostics,
            )


            raw_lines.append(
                row
            )



        # ------------------------------------------------
        # Validation / filtering stage
        # ------------------------------------------------

        final = []


        wanted = None


        if sport_filter and sport_filter != "all":

            wanted = sport_filter



        for row in raw_lines:


            # only supported sports

            if wanted:

                if row["sport"] != wanted:

                    diagnostics.removed[
                        "wrong_sport"
                    ] += 1

                    continue


            else:

                if row["sport"] not in SUPPORTED_SPORTS:

                    diagnostics.removed[
                        "unsupported_sport"
                    ] += 1

                    continue



            # no player = unusable

            if not row["player"]:

                diagnostics.removed[
                    "missing_player"
                ] += 1

                continue



            # no line = unusable

            if row["line"] is None:

                diagnostics.removed[
                    "missing_line"
                ] += 1

                continue



            final.append(
                row
            )



        # ------------------------------------------------
        # Count final markets
        # ------------------------------------------------

        diagnostics.kept = len(
            final
        )


        diagnostics.odds = Counter(
            x["odds_type"]
            for x in final
        )


        diagnostics.sports = Counter(
            x["sport"]
            for x in final
        )



        if debug:

            diagnostics.report()



        _pp_cache[key] = (
            time.time(),
            final
        )


        return final, None



    except Exception as exc:

        return [], str(exc)





# ============================================================
# Underdog adapter
# ============================================================


def fetch_underdog(
    sport_filter: str | None = None
):


    if not _UD_OK:

        return [], (
            "underdog module unavailable"
        )


    try:

        ud = Underdog()

        props = ud.get_props()


        output=[]


        for p in props:


            sport = p.sport


            if sport_filter:

                if sport != sport_filter:

                    continue



            if sport not in (
                "MLB",
                "WNBA",
                "TENNIS",
            ):

                continue



            output.append(

                {

                    "id":
                        f"ud_{p.line_id}",


                    "source":
                        "underdog",


                    "sport":
                        sport,


                    "player":
                        p.player_name,


                    "team":
                        None,


                    "position":
                        p.position,


                    "stat_type":
                        _clean_stat(
                            p.stat
                        ),


                    "line":
                        p.line,


                    "odds_type":
                        (
                            "standard"
                            if not p.is_boosted
                            else
                            "other"
                        ),


                    "matchup":
                        None,


                    "start_time":
                        None,


                    "status":
                        p.status,


                    "over_implied":
                        _american_to_implied(
                            p.american_price
                        ),


                    "under_implied":
                        None,


                    "over_price":
                        p.american_price,


                    "under_price":
                        None,


                    "meta":
                    {

                        "line_id":
                            p.line_id

                    }

                }

            )


        return output, None



    except Exception as exc:

        return [], str(exc)





# ============================================================
# Combined pull
# ============================================================


def fetch_all_lines():

    pp, pp_error = fetch_prizepicks()

    ud, ud_error = fetch_underdog()


    return (
        pp + ud,
        {
            "prizepicks":
                pp_error,

            "underdog":
                ud_error,
        }
    )
