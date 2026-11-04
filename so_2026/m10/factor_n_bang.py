# Source - https://codereview.stackexchange.com/q/252593
# Posted by Martins Micheal, modified by community. See post 'Timeline' for change history
# Retrieved 2026-10-18, License - CC BY-SA 4.0

from math import floor, sqrt


def is_prime(x: int) -> bool:

    def test_1_or_5(x: int, i: int) -> bool:
        return x % (i + 1) == 0 or i % (i + 5) == 0

    if x in {2, 3, 5}:
        return True
    if (x % 2 == 0) or (x % 3 == 0) or (x % 5 == 0) or (x == 1):
        return False
    return all(not test_1_or_5(x, i) for i in range(6, floor(sqrt(x)), 6))


def count_prime_power_in_factorial(n: int, p: int) -> int:
    count = 0
    power = p
    while power <= n:
        count += n // power
        power *= p
    return count


def decomp(n: int) -> str:
    assert n >= 2

    pri = list(filter(is_prime, range(2, n + 1)))

    a = {}
    for i in pri:
        a[i] = count_prime_power_in_factorial(n, i)

    return format_factors(a)


def format_factors(a: dict[int, int]) -> str:
    factors = []
    for key in sorted(a.keys()):
        count = a[key]
        if count > 1:
            factors.extend([str(key)] * count)
        else:
            factors.append(str(key))
    return " * ".join(factors) if factors else "1"
