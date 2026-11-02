import re
import unittest
from pathlib import Path

from requests_cache import CachedSession

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
    assert 79_766 == len(words), len(words)
    return list(filter(valid_re.search, words))


class LoadFactorTest(unittest.TestCase):

    def test_get_words(self) -> None:
        words = get_words()
        self.assertEqual(75_048, len(words))
        self.assertEqual(
            ["zygotes", "zymurgy"],
            words[-2:],
        )
