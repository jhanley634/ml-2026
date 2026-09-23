#! /usr/bin/env python

from hypothesis import given
from hypothesis import strategies as st


def div(a: int, b: int) -> tuple[int, int]:
    """
    Performs integer division of A by B using only subtraction and bitwise operations.
    Returns the quotient and remainder.
    """

    if b == 0:
        raise ZeroDivisionError

    sgnum = -1 if (a < 0) ^ (b < 0) else 1
    orig_b = b

    a, b = map(abs, (a, b))
    quo = 0
    rem = a

    for i in range(a.bit_length() - 1, -1, -1):
        if rem >= (b << i):
            quo |= 1 << i
            rem -= b << i

    assert quo >= 0
    assert rem in range(b)

    if sgnum == -1:
        quo += 1
        rem = abs(rem - b) % b

    return sgnum * quo, rem if orig_b >= 0 else -rem


@given(st.integers(), st.integers().filter(lambda x: x != 0))
def test_div(a: int, b: int) -> None:
    q, r = div(a, b)
    assert q * b + r == a, f"{a=}  {b=};  {q=}  {r=}  {q * b + r=}"

    if b == 0 or max((a, b)) >= 9_007_199_254_740_993:
        return

    expected_q = a // b
    expected_q, expected_r = divmod(a, b)

    assert q == expected_q, f"  {q=}, {expected_q=};   {r=},  {expected_r=}"


if __name__ == "__main__":
    # see https://github.com/python/cpython/blob/main/Objects/longobject.c
    # > The / and % operators are now defined in terms of divmod().
    assert divmod(13, 10) == (1, 3)
    assert divmod(-13, 10) == (-2, 7)
    assert divmod(13, -10) == (-2, -7)
    assert divmod(-13, -10) == (1, -3)

    assert divmod(43, 7) == (6, 1)
    assert divmod(-43, 7) == (-7, 6)
    assert divmod(43, -7) == (-7, -6)
    assert divmod(-43, -7) == (6, -1)

    assert div(42, 7) == (6, 0)
    assert div(43, 7) == (6, 1)
    assert div(-43, 7) == (-7, 6)
    assert div(43, -7) == (-7, -6), div(43, -7)
    assert div(-43, -7) == (6, -1)

    # test_div()
