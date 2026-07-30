"""
pullers.py — JUICED unified prop ingestion layer.

Purpose:
- Pull live props from PrizePicks + Underdog clients
- Normalize both sources into the JUICED Line schema
- Keep ingestion separate from projection/model logic

Sources:
    prizepicks.py  -> authenticated PrizePicks board
    underdog.py    -> Underdog public board

No direct API scraping happens here.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------

_BD = Path(__file__).parent.parent / "betting_dashboard"

if _BD.exists() and str(_BD) not in sys.path:
    sys.path.insert(0, str(_BD))

try:
    from prizepicks import PrizePicks
    _PP_OK = True
except Exception:
    PrizePicks = None
    _PP_OK = False

try:
    from underdog import Underdog, UnderdogProp
    _UD_OK = True
except Exception:
    Underdog = None
    UnderdogProp = None
    _UD_OK = False


# ---------------------------------------------------------------------------
# Unified JUICED Line schema
# ---------------------------------------------------------------------------

"""
{
"id": str,
"source": "prizepicks" | "underdog",
"sport": str,
"player": str | None,
"team": str | None,
"position": str | None,
"stat_type": str | None,
"line": float | None,
"odds_type": "standard" | "demon" | "goblin",
"matchup": str | None,
"start_time": str | None,
"status": str | None,

"over_implied": float | None,
"under_implied": float | None,

"over_price": str | None,
"under_price": str | None,

"pickem_price": int | None,

"headshot": str | None,

"meta": dict
}
"""


# ---------------------------------------------------------------------------
# Sport normalization
# ---------------------------------------------------------------------------

_SUPPORTED = {
    "MLB": "MLB",
    "WNBA": "WNBA",
    "Tennis": "Tennis",
}


def normalize_sport(value: str | None) -> str:
    if not value:
        return "other"

    s = value.lower()

    if "mlb" in s or "baseball" in s:
        return "MLB"

    if "wnba" in s:
        return "WNBA"

    if "tennis" in s or "atp" in s or "wta" in s:
        return "Tennis"

    return "other"



# ---------------------------------------------------------------------------
# Stat cleanup
# ---------------------------------------------------------------------------

def clean_stat(stat: str | None) -> str | None:
    if not stat:
        return None

    return (
        stat
        .replace("_", " ")
        .strip()
        .title()
    )



# ---------------------------------------------------------------------------
# PrizePicks adapter
# ---------------------------------------------------------------------------


def _pp_to_line(p) -> dict[str, Any]:
    """
    Convert PrizePicks Projection object
    into JUICED Line schema.
    """

    odds = (p.odds_type or "standard").lower()

    return {
        "id": f"pp_{p.id}",

        "source": "prizepicks",

        "sport": normalize_sport(p.league),

        "player": p.player_name,

        "team": p.team,

        "position": p.position,

        "stat_type": clean_stat(p.stat_type),

        "line": p.line_score,

        "odds_type": odds,

        "matchup": p.description,

        "start_time": p.start_time,

        "status": p.status,


        # PrizePicks has no over/under prices
        "over_implied": None,
        "under_implied": None,

        "over_price": None,
        "under_price": None,


        # used by EV engine for standard PP legs
        "pickem_price": -137 if odds == "standard" else None,


        "headshot": p.image_url,


        "meta": {
            "player_id": p.player_id,
            "league": p.league,
            "league_id": p.league_id,
            "is_promo": p.is_promo,
            "rank": p.rank,
        },
    }



def fetch_prizepicks(
    sport_filter: str | None = None
) -> tuple[list[dict], str | None]:

    if not _PP_OK:
        return [], "PrizePicks client unavailable"


    try:

        pp = PrizePicks()

        projections = list(
            pp.get_all_projections()
        )


        lines = []

        wanted = (
            normalize_sport(sport_filter)
            if sport_filter and sport_filter != "all"
            else None
        )


        for p in projections:

            row = _pp_to_line(p)


            # remove unsupported sports
            if row["sport"] not in {
                "MLB",
                "WNBA",
                "Tennis"
            }:
                continue


            if wanted and row["sport"] != wanted:
                continue


            # remove combos
            player = row["player"] or ""

            stat = row["stat_type"] or ""

            if "+" in player:
                continue

            if "combo" in stat.lower():
                continue


            lines.append(row)



        return lines, None


    except Exception as exc:

        return [], str(exc)

# ---------------------------------------------------------------------------
# Underdog adapter
# ---------------------------------------------------------------------------


def _american_to_implied(price: str | int | None) -> float | None:
    """
    Convert American odds into implied probability.
    """

    if price is None:
        return None

    try:
        p = float(price)

        if p > 0:
            return round(
                100 / (p + 100),
                4
            )

        return round(
            abs(p) / (abs(p) + 100),
            4
        )

    except Exception:
        return None



def tennis_opponent(raw_title: str | None) -> str | None:

    if not raw_title:
        return None

    match = re.search(
        r"\(vs\.?\s+(.+?)\)",
        raw_title,
        re.I
    )

    return (
        match.group(1).strip()
        if match
        else None
    )



def _ud_to_lines(
    props: list["UnderdogProp"]
) -> list[dict[str, Any]]:

    """
    Underdog returns:
        player/stat/line/OVER
        player/stat/line/UNDER

    Combine them into one JUICED Line.
    """


    grouped: dict[str, dict[str, Any]] = {}


    for p in props:


        sport = normalize_sport(
            p.sport
        )


        if sport not in {
            "MLB",
            "WNBA",
            "Tennis"
        }:
            continue



        uid = (
            f"ud_{p.line_id}"
        )



        if uid not in grouped:


            grouped[uid] = {

                "id": uid,

                "source": "underdog",

                "sport": sport,


                "player": p.player_name,

                "team": None,

                "position": p.position,


                "stat_type": clean_stat(
                    p.stat
                ),


                "line": p.line,


                "odds_type": (
                    "standard"
                    if not p.is_boosted
                    else "boosted"
                ),



                "matchup": None,

                "start_time": None,

                "status": p.status,



                "over_implied": None,

                "under_implied": None,


                "over_price": None,

                "under_price": None,


                "pickem_price": None,


                "headshot": p.image_url,



                "meta": {

                    "match_id": p.match_id,

                    "line_id": p.line_id,

                    "country": p.country,

                    "boosted": p.is_boosted,

                    "multiplier": p.payout_multiplier,

                }

            }



        row = grouped[uid]


        choice = (
            p.choice or ""
        ).lower()



        implied = _american_to_implied(
            p.american_price
        )



        if choice in (
            "over",
            "higher"
        ):

            row["over_price"] = (
                p.american_price
            )

            row["over_implied"] = implied


        else:

            row["under_price"] = (
                p.american_price
            )

            row["under_implied"] = implied



    return list(grouped.values())



def fetch_underdog(
    sport_filter: str | None = None
) -> tuple[list[dict], str | None]:


    if not _UD_OK:

        return [], "Underdog client unavailable"



    try:

        ud = Underdog()


        props = ud.get_props()



        if sport_filter and sport_filter != "all":

            wanted = normalize_sport(
                sport_filter
            )

            props = [

                p for p in props

                if normalize_sport(
                    p.sport
                ) == wanted

            ]



        lines = _ud_to_lines(
            props
        )


        return lines, None



    except Exception as exc:

        return [], str(exc)

# ---------------------------------------------------------------------------
# Combined JUICED board fetch
# ---------------------------------------------------------------------------


def fetch_all(
    sport_filter: str | None = None
) -> tuple[list[dict], dict]:

    """
    Pull all supported props from every source.

    Returns:
        lines
        diagnostics
    """

    all_lines: list[dict] = []

    diagnostics = {

        "sources": {},

        "total": 0,

        "errors": [],

    }



    # -------------------------
    # PrizePicks
    # -------------------------

    pp_lines, pp_error = fetch_prizepicks(
        sport_filter
    )


    if pp_error:

        diagnostics["errors"].append(
            f"PrizePicks: {pp_error}"
        )


    all_lines.extend(
        pp_lines
    )


    diagnostics["sources"]["PrizePicks"] = (
        summarize_lines(pp_lines)
    )



    # -------------------------
    # Underdog
    # -------------------------

    ud_lines, ud_error = fetch_underdog(
        sport_filter
    )


    if ud_error:

        diagnostics["errors"].append(
            f"Underdog: {ud_error}"
        )


    all_lines.extend(
        ud_lines
    )


    diagnostics["sources"]["Underdog"] = (
        summarize_lines(ud_lines)
    )



    diagnostics["total"] = len(
        all_lines
    )


    return all_lines, diagnostics




# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def summarize_lines(
    lines: list[dict]
) -> dict:


    result = {

        "total": len(lines),

        "sports": {},

        "odds_types": {},

    }


    for line in lines:


        sport = (
            line.get("sport")
            or "unknown"
        )


        odds = (
            line.get("odds_type")
            or "standard"
        )


        result["sports"][sport] = (
            result["sports"].get(sport, 0)
            + 1
        )


        result["odds_types"][odds] = (
            result["odds_types"].get(odds, 0)
            + 1
        )


    return result



def print_diagnostics(
    diagnostics: dict
):

    print("\n==============================")

    print(" JUICED INGESTION HEALTH")

    print("==============================\n")


    for source, data in diagnostics["sources"].items():

        print(source)

        print("------------------------------")

        print(
            f"Total: {data['total']}"
        )

        print(
            "Sports:",
            data["sports"]
        )

        print(
            "Odds:",
            data["odds_types"]
        )

        print()



    if diagnostics["errors"]:

        print("ERRORS:")

        for e in diagnostics["errors"]:

            print(
                " -",
                e
            )


    print(
        "TOTAL LINES:",
        diagnostics["total"]
    )

    print("==============================\n")



# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------


def get_lines(
    sport_filter: str | None = None
) -> list[dict]:

    """
    Existing JUICED API calls can keep using this.
    """

    lines, _ = fetch_all(
        sport_filter
    )

    return lines
