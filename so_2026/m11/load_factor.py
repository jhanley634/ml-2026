from collections import Counter

import mmh3
import numpy as np
from sympy import primerange

EMPTY = ""


# https://en.wikipedia.org/wiki/Quadratic_probing#Alternating_signs
# https://en.wikipedia.org/wiki/Linear_probing#Primary_clustering


class HashSet:
    """Tracks a bunch of 'str' keys, using open addressing."""

    def __init__(self, initial_num_elts: int, load_factor: float = 0.75) -> None:
        self.load_factor = load_factor
        self.capacity = 1
        self.size = 0  # no elements stored, yet
        self.chain_length = Counter([0])

        primes = np.array(list(primerange(120_000))).astype(np.int32)
        assert len(primes) == 11_301
        assert max(primes) == 119_993

        for p in primes:
            if load_factor * p > initial_num_elts:
                self.capacity = p
                break

        self.arr = [EMPTY for _ in range(self.capacity)]

    def _hash(self, key: str) -> int:
        return mmh3.hash(key) % self.capacity

    def _put1(self, key: str) -> int:
        """Stores key, and returns the probing chain length."""
        assert self.size < self.capacity
        idx = self._hash(key)
        probes = 0
        while self.arr[idx] != EMPTY:
            if self.arr[idx] == key:
                return probes  # key already exists
            idx = (idx + 1) % self.capacity  # linear probing
            probes += 1

        self.arr[idx] = key
        self.size += 1
        return probes

    def put(self, key: str) -> None:
        assert key
        if self.size + 1 >= self.load_factor * self.capacity:
            self._resize()

        probes = self._put1(key)
        self.chain_length[probes] += 1
        # print({i: self.chain_length[i] for i in range(10)})

    def _resize(self) -> None:
        old_arr = self.arr
        self.capacity = int(1 + 1.1 * self.capacity)  # or double it: self.capacity *= 2
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
