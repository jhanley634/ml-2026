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


def decomp(n: int) -> dict[int, int]:
    assert n >= 2

    primes = set(filter(is_prime, range(2, n + 1)))
    f = {}
    for p in primes:
        f[p] = count_prime_power_in_factorial(n, p)

    return f


def fmt(f: dict[int, int]) -> str:
    factors = []
    for p, exp in sorted(f.items()):
        if exp > 1:
            factors.append(f"{p}^{exp}")
        else:
            factors.append(f"{p}")
    assert factors
    return " * ".join(factors)
