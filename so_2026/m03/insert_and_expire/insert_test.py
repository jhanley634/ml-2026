import unittest
from time import time

from so_2026.m03.insert_and_expire.insert import create_empty_table, ins


class InsertTest(unittest.TestCase):
    def test_insert(self, *, verbose: bool = False) -> None:
        create_empty_table()
        ins(1)

        t0 = time()
        ins(40_000)
        elapsed = time() - t0

        if verbose:
            print(f"\n{elapsed=:06f}")
        self.assertLess(elapsed, 1.0)
