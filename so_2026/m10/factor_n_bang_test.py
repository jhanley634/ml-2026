from math import factorial, prod
from unittest import TestCase

from so_2026.m10.factor_n_bang import decomp, fmt, is_prime


def as_list(f: dict[int, int]) -> list[int]:
    return [p for p, exp in sorted(f.items()) for _ in range(exp)]


class FactorNBangTest(TestCase):

    def test_decomp(self) -> None:
        self.assertEqual(
            decomp(4),
            {
                2: 3,
                3: 1,
            },
        )

    def test_as_list(self) -> None:
        self.assertEqual(fmt(decomp(4)), "2^3 * 3")
        self.assertEqual(fmt(decomp(9)), "2^7 * 3^4 * 5 * 7")
        for n, factors in [
            (2, [2]),
            (3, [2, 3]),
            (4, [2, 2, 2, 3]),
            (5, [2, 2, 2, 3, 5]),
            (6, [2, 2, 2, 2, 3, 3, 5]),
            (7, [2, 2, 2, 2, 3, 3, 5, 7]),
            (8, [2, 2, 2, 2, 2, 2, 2, 3, 3, 5, 7]),
            (9, [2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 5, 7]),
        ]:
            self.assertEqual(factors, as_list(decomp(n)))

            self.assertEqual(prod(factors), factorial(n))

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
