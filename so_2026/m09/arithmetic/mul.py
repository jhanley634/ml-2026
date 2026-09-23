#! /usr/bin/env python

from hypothesis import given
from hypothesis import strategies as st


def mul(a: int, b: int) -> int:
    """
    Finds the product of A and B using only addition and bitwise masking / shifting.
    """

    sgnum = -1 if (a < 0) ^ (b < 0) else 1

    a, b = map(abs, (a, b))
    p = 0
    while b > 0:
        if b & 1:
            p += a
        a <<= 1
        b >>= 1

    return sgnum * p


@given(st.integers(), st.integers())
def test_mul(a: int, b: int) -> None:
    p = mul(a, b)
    expected = a * b
    assert p == expected, f"{p=},  {expected=}"


BIG = 72_057_594_037_927_932

if __name__ == "__main__":
    assert mul(6, -7) == -42
    assert mul(BIG, BIG) == BIG**2

    test_mul()
