from array import array

import mmh3
from sympy import primerange


class HashSet:
    """Tracks a bunch of 'str' keys, using open addressing."""

    def __init__(self, initial_num_elts: int, load_factor: float = 0.75) -> None:
        self.load_factor = load_factor
        self.capacity = 1

        primes = array("L", primerange(120_000))
        for p in primes:
            if load_factor * p > initial_num_elts:
                self.capacity = p
                break

        self.arr = [None] * self.capacity

    def put(self, key: str) -> None: ...
