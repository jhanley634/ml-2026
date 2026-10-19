from unittest import TestCase

from so_2026.m10.factor_n_bang import decomp, is_prime

_no_star = str.maketrans("*", " ")


def as_list(s: str) -> list[int]:
    return list(map(int, s.translate(_no_star).split()))


class FactorNBangTest(TestCase):

    def test_decomp(self) -> None:
        print(decomp(7))

    def test_as_list(self) -> None:
        self.assertEqual("2 * 3", decomp(4))
        self.assertEqual(
            [2, 3],
            as_list(decomp(4)),
        )

        self.assertEqual("2 * 3 * 5 * 7", decomp(7))
        self.assertEqual(
            [2, 3, 5, 7],
            as_list(decomp(7)),
        )

    def test_is_prime(self) -> None:
        expected = [
            (2, True),
            (3, True),
            (4, False),
            (5, True),
            (6, False),
            (7, True),
            (8, False),
            (9, False),
            (10, False),
            (11, True),
            (12, False),
            (13, True),
        ]
        for n, c in expected:
            self.assertEqual(c, is_prime(n))
