import unittest
from time import time

from so_2026.m03.insert_and_expire.insert import ReqLog, create_empty_table, get_session, ins


class InsertTest(unittest.TestCase):
    def test_insert(self, *, verbose: bool = False) -> None:
        create_empty_table()
        ins(1)

        elapsed_limit = 0.1
        num_recs = 4_000
        t0 = time()
        ins(num_recs)  # at least 40k (often ~ 50k) rows per second
        elapsed = time() - t0

        if verbose:
            print(f"\n{elapsed=:06f}")
        self.assertLess(elapsed, elapsed_limit)

        with get_session() as sess:
            result = sess.query(ReqLog).all()
            self.assertEqual(num_recs + 1, len(result))
