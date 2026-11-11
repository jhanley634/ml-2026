import unittest

# ruff: ignore[camelcase-imported-as-lowercase]
from roman import fromRoman as from_roman
from roman import toRoman as to_roman  # ruff: ignore[camelcase-imported-as-lowercase]

from so_2026.m11.roman_numerals import roman_to_decimal


class RomanNumeralsTest(unittest.TestCase):

    def test_pypi_library_roundtrip(self, *, verbose: bool = False) -> None:
        for i in range(1, 5_000):
            self.assertEqual(i, from_roman(to_roman(i)))
            if verbose:
                print(f"{i:5d}  ", to_roman(i))

    def test_roman_to_decimal(self) -> None:
        for i in range(15, 4_000):
            roman = to_roman(i)
            print(i, roman)
            self.assertEqual(i, roman_to_decimal(roman))
