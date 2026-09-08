#! /usr/bin/env python

from math import sqrt


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
    assert r == int(sqrt(n))
    return r


if __name__ == "__main__":
    for i in range(30):
        print(i, isqrt(i))
