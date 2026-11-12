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
        for i in range(1, 4_000):
            roman = to_roman(i)
            self.assertEqual(i, roman_to_decimal(roman))

    def test_validations(self) -> None:
        long = "I" * 15
        roman_to_decimal(long)  # works fine, even though "III" should be the max
        with self.assertRaises(ValueError):
            roman_to_decimal(long + "I")

        with self.assertRaises(ValueError):
            roman_to_decimal("Hello world!")

        self.assertEqual(3888, roman_to_decimal("MMMDCCCLXXXVIII"))  # longest valid Roman
        self.assertEqual(3999, roman_to_decimal("MMMCMXCIX"))
        with self.assertRaises(ValueError):
            roman_to_decimal("MMMM")
