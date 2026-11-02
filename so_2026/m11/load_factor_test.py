import re
import unittest
from array import array
from pathlib import Path

import mmh3
from requests_cache import CachedSession
from sympy import primerange

from so_2026.m11.load_factor import EMPTY, HashSet

TEMP = Path("/tmp")
LEXICON_URL = "https://github.com/drewtuley/WordLadder/raw/713cb88/brit-a-z.txt"


def _get_lexicon(url: str = LEXICON_URL) -> Path:
    """Does a web GET of the word list if necessary, then returns its local Path."""
    path = TEMP / url.rsplit("/", maxsplit=1)[-1]

    with CachedSession() as session:
        response = session.get(url)
        response.raise_for_status()
        path.write_text(response.text)

    return path


valid_re = re.compile(r"^[a-z-]+$")  # Mostly this discards "'", and also SPACE.


def get_words() -> list[str]:
    words = _get_lexicon().read_text().split("\n")
    assert len(words) == 79_766, len(words)
    return list(filter(valid_re.search, words))


class LoadFactorTest(unittest.TestCase):

    def test_get_words(self) -> None:
        words = get_words()
        self.assertEqual(75_048, len(words))
        self.assertEqual(
            ["zygotes", "zymurgy"],
            words[-2:],
        )

    def test_murmur(self) -> None:
        lexicon = _get_lexicon()
        d = mmh3.hash(lexicon.read_text())
        self.assertEqual(-2_103_894_192, d)

    def test_primes(self) -> None:
        primes = array("L", primerange(120_000))
        self.assertEqual(11_301, len(primes))
        self.assertEqual(642_361_192, sum(primes))

    def test_hash_set(self) -> None:
        words = get_words()
        s = HashSet(len(words))

        assert s
        self.assertEqual(5, 2 + 3)

    def test_put(self) -> None:
        s = HashSet(10)
        s.put("apple")
        s.put("banana")
        s.put("cherry")

        self.assertIn("apple", s.arr)
        self.assertIn("banana", s.arr)
        self.assertIn("cherry", s.arr)

        s.put("apple")  # Adding duplicate
        self.assertEqual(s.size, 3)

        self.assertEqual(s.arr.count(EMPTY), s.capacity - s.size)
