#! /usr/bin/env python


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


if __name__ == "__main__":
    print(mul(7, 6))
    print(mul(6, -7))
