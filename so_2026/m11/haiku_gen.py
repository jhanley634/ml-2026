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


_STR_TO_POS = {member.name.capitalize(): member for member in PartOfSpeech}


@beartype
@dataclass
class Word:
    word: str
    part_of_speech: PartOfSpeech
    syllables: int


WORD_FILE = Path("/tmp/haiku-words.json")


def _get_haiku_words() -> Generator[Word]:
    with Path(WORD_FILE).open(encoding="utf-8") as fin:
        for entry in json.load(fin):
            del entry["id"]
            entry["part_of_speech"] = _STR_TO_POS[entry["part_of_speech"]]
            yield Word(**entry)


class HaikuGen:
    def __init__(self) -> None:
        self.seen: set[str] = set()  # Words already used, and so are ineligible.
        self.words = list(_get_haiku_words())
        starter = next(self._get_starters())
        pp(starter)
        # for word in self.words:            print(word.syllables)

    def _get_starters(self) -> Generator[Word]:
        # "choose ... 5 starter words, of at least 3 different syllable lengths"
        desired = random.sample(range(1, 6), 5)
        candidate = Word("Sentinel", PartOfSpeech.noun, syllables=99)  # won't match
        for i in range(10**6):

            while candidate.syllables != desired[i % len(desired)]:
                candidate = random.choice(self.words)

            self.seen.add(candidate.word)
            yield candidate


def test_pos_name() -> None:
    assert str(PartOfSpeech.verb) == "verb"  # would default to "PartOfSpeech.verb"


if __name__ == "__main__":
    HaikuGen()
