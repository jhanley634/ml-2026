#! /usr/bin/env python

# from https://stackoverflow.com/beta/challenges/80015893/challenge-25-count-your-syllables

import json
import random
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from pprint import pp
from typing import TYPE_CHECKING

from beartype import beartype

if TYPE_CHECKING:
    from collections.abc import Generator


@beartype
class PartOfSpeech(Enum):
    adjective = auto()
    adverb = auto()
    conjunction = auto()
    determiner = auto()
    interjection = auto()
    noun = auto()
    preposition = auto()
    pronoun = auto()
    verb = auto()

    def __str__(self) -> str:
        return self.name


@beartype
@dataclass
class Word:
    word: str
    part_of_speech: PartOfSpeech
    syllables: int


_STR_TO_POS = {member.name.capitalize(): member for member in PartOfSpeech}

WORD_FILE = Path("/tmp/k/haiku-words.json")


def _get_haiku_words() -> Generator[Word]:
    with Path(WORD_FILE).open(encoding="utf-8") as fin:
        for entry in json.load(fin):
            del entry["id"]
            entry["part_of_speech"] = _STR_TO_POS[entry["part_of_speech"]]
            yield Word(**entry)


class HaikuGen:

    def __init__(self) -> None:
        self.words = list(_get_haiku_words())
        starter = random.choice(self.words)
        pp(starter)


def test_pos_name() -> None:
    assert str(PartOfSpeech.verb) == "verb"  # would default to "PartOfSpeech.verb"


if __name__ == "__main__":
    HaikuGen()
