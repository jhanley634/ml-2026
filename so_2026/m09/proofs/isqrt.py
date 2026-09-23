#! /usr/bin/env python

from math import sqrt

import hypothesis.strategies as st
from hypothesis import given

TOO_BIG = 72_057_594_037_927_932


def isqrt(n: int) -> int:
    """
    Finds the integer square root of N.
    """
    if n < 0:
        raise ValueError

    r = n
    while r > 0 and abs(r - (n // r)) > 1:
        r = (r + n // r) // 2

    if r * r > n:
        r -= 1

    assert r**2 <= n < (r + 1) ** 2
    if n < TOO_BIG:
        assert r == int(sqrt(n)), (sqrt(n), r, n)
    return r


@given(st.integers(min_value=0))
def _test_isqrt(n: int) -> None:
    r = isqrt(n)
    assert r**2 <= n < (r + 1) ** 2


if __name__ == "__main__":
    isqrt(TOO_BIG - 1)  # This is 0x1.fffffffffffffp+55
    isqrt(TOO_BIG)
    _test_isqrt()

    for i in range(20):
        print(i, isqrt(i))
