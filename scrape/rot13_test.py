import string
import unittest

from hypothesis import given
from hypothesis import strategies as st


def rot13(clear: str) -> str:
    """Implements a Caesar cipher."""

    def shift_char(ch: str, shift: int) -> str:
        if "a" <= ch <= "z":
            return chr((ord(ch) - ord("a") + shift) % 26 + ord("a"))
        if "A" <= ch <= "Z":
            return chr((ord(ch) - ord("A") + shift) % 26 + ord("A"))
        return ch

    return "".join(shift_char(ch, 13) for ch in clear)


class Rot13Test(unittest.TestCase):

    def test_rot13_roundtrip(self) -> None:
        """rot13(rot13(s)) is the identity function."""
        test_strings: list[str] = [
            "Hello, World!",
            string.ascii_lowercase,
            string.ascii_uppercase,
            "",
            "12345!@#$%",
            "This is a test.",
        ]

        for s in test_strings:
            self.assertEqual(s, rot13(rot13(s)))

    def test_with_hypothesis(self) -> None:
        @given(st.text())
        def check_rot13_property(text: str) -> None:
            self.assertEqual(text, rot13(rot13(text)))

        check_rot13_property()
