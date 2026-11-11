import unittest

# ruff: ignore[camelcase-imported-as-lowercase]
from roman import fromRoman as from_roman
from roman import toRoman as to_roman  # ruff: ignore[camelcase-imported-as-lowercase]


class RomanNumeralsTest(unittest.TestCase):

    def test_pypi_library_roundtrip(self, *, verbose: bool = False) -> None:
        for i in range(1, 5_000):
            self.assertEqual(i, from_roman(to_roman(i)))
            if verbose:
                print(f"{i:5d}  ", to_roman(i))
