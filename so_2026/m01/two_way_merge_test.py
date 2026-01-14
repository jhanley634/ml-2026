import unittest

from so_2026.m01.two_way_merge import merge


class TwoWayMergeTest(unittest.TestCase):
    def test_find_median(self) -> None:
        a = list(range(2))
        b = list(range(4))
        self.assertEqual(
            [0, 0, 1, 1, 2, 3],
            list(merge(a, b)),
        )
        self.assertEqual(
            [0, 0, 1, 1, 2, 3],
            list(merge(b, a)),
        )
