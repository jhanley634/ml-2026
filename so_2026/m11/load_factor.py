from array import array

import mmh3
from sympy import primerange

EMPTY = ""


class HashSet:
    """Tracks a bunch of 'str' keys, using open addressing."""

    def __init__(self, initial_num_elts: int, load_factor: float = 0.75) -> None:
        self.load_factor = load_factor
        self.capacity = 1
        self.size = 0  # no elements stored, yet

        primes = array("L", primerange(120_000))
        for p in primes:
            if load_factor * p > initial_num_elts:
                self.capacity = p
                break

        self.arr = [EMPTY for _ in range(self.capacity)]

    def _hash(self, key: str) -> int:
        return mmh3.hash(key) % self.capacity

    def put(self, key: str) -> None:
        assert key
        if self.size + 1 >= self.load_factor * self.capacity:
            self._resize()

        idx = self._hash(key)
        while self.arr[idx] != EMPTY:
            if self.arr[idx] == key:
                return  # key already exists, so do nothing
            idx = (idx + 1) % self.capacity  # linear probing

        self.arr[idx] = key
        self.size += 1

    def _resize(self) -> None:
        old_arr = self.arr
        self.capacity *= 2
        self.arr = [EMPTY for _ in range(self.capacity)]
        self.size = 0

        for key in old_arr:
            if key != EMPTY:
                self.put(key)

    def get(self, key: str) -> bool:
        idx = self._hash(key)
        while self.arr[idx] != EMPTY:
            if self.arr[idx] == key:
                return True
            idx = (idx + 1) % self.capacity
        return False
