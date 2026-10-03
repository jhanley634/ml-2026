#! /usr/bin/env python

# Source - https://stackoverflow.com/a/80007615
# Posted by J_H, modified by community. See post 'Timeline' for change history
# Retrieved 2026-10-03, License - CC BY-SA 4.0
# pyright: basic, reportArgumentType=false

from operator import itemgetter
from typing import Any

import requests
from glom import Iter, glom

URL = "https://sports.core.api.espn.com/v2/sports/hockey/leagues/nhl/athletes/1/statistics"


# furas_print and glom_print do the same thing.


def furas_print() -> None:
    for cat in hockey["splits"]["categories"]:
        for stat in cat["stats"]:
            if stat["abbreviation"] in STAT_CODES:
                print(cat["name"], stat["abbreviation"], stat["displayValue"])


STAT_CODES = {"GA", "RPI", "PIM"}


def is_wanted(stat: dict[str, Any]) -> bool:
    return stat["abbreviation"] in STAT_CODES


def glom_print() -> None:
    spec = Iter().filter(is_wanted).map(itemgetter("abbreviation", "displayValue"))

    for cat in glom(hockey, "splits.categories"):
        for abbrev, value in glom(cat["stats"], spec):
            print(cat["name"], abbrev, value)


if __name__ == "__main__":
    hockey = requests.get(URL).json()
    furas_print()
    print("----")
    glom_print()
