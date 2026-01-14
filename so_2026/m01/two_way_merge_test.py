import unittest

from hypothesis import given
from hypothesis import strategies as st

from so_2026.m01.two_way_merge import merge


class TwoWayMergeTest(unittest.TestCase):
    def test_merge(self) -> None:
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

    @given(
        st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=0),
        st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=0),
    )
    def test_hypothesis_merge(self, a: list[float], b: list[float]) -> None:
        a.sort()
        b.sort()
        merged_list = list(merge(a, b))

        self.assertEqual(len(a) + len(b), len(merged_list))
        self.assertTrue(all(item in merged_list for item in a))
        self.assertTrue(all(item in merged_list for item in b))
        self.assertEqual(merged_list, sorted(merged_list))  # monotonic
        self.assertEqual(merged_list, sorted(a + b))
