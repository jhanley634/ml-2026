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

    def _get_starters(self) -> Generator[Word]:
        # "choose ... 5 starter words, of at least 3 different syllable lengths"
        desired = random.sample(range(1, 6), 5)
        candidate = Word("Sentinel", PartOfSpeech.noun, syllables=99)  # won't match
        for i in range(10**6):

            while candidate.syllables != desired[i % len(desired)]:
                candidate = random.choice(self.words)

            self.seen.add(candidate.word)
            yield candidate

    def _get_word(self, max_allowed_syl: int) -> Word:
        candidate = Word("Sentinel", PartOfSpeech.noun, syllables=99)  # won't match
        while candidate.syllables > max_allowed_syl:
            candidate = random.choice(self.words)

        return candidate

    @staticmethod
    def _fmt_line(line: list[Word]) -> str:
        return " ".join((w.word for w in line))

    @staticmethod
    def _total_syl(line: list[Word]) -> int:
        return sum(w.syllables for w in line)

    def _generate_line(self, num_syl: int) -> list[Word]:
        assert num_syl in {5, 7}
        starter = next(self._get_starters())
        line = [starter]
        while self._total_syl(line) < num_syl:
            w = self._get_word(num_syl - self._total_syl(line))
            line.append(w)
            self.seen.add(w.word)

        return line

    def generate_poem(self, col_width: int = 35) -> str:

        poem = list(map(self._generate_line, (5, 7, 5)))

        totals = list(map(self._total_syl, poem))
        assert totals == [5, 7, 5]

        ret = ""
        for line, total in zip(poem, totals):
            ret += self._fmt_line(line).ljust(col_width) + f" ({total})\n"
        return ret


def test_pos_name() -> None:
    assert str(PartOfSpeech.verb) == "verb"  # would default to "PartOfSpeech.verb"


def test_generate_poem() -> None:
    assert HaikuGen().generate_poem().endswith(" (5)\n")


if __name__ == "__main__":
    print(HaikuGen().generate_poem())
