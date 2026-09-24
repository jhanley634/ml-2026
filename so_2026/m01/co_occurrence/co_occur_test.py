import unittest
from typing import TYPE_CHECKING

import numpy as np

from so_2026.m01.co_occurrence.co_occur import find_coincidences, generate_decay_events

if TYPE_CHECKING:
    from numpy.typing import NDArray


def np_array_int32(xs: list[int]) -> NDArray[np.int32]:
    return np.array(xs, dtype=np.int32)


class CoOccurTest(unittest.TestCase):
    def test_generate_decay_events(self) -> None:
        a_b = generate_decay_events()
        self.assertEqual(2, len(a_b))

    def test_find_coincidences(self) -> None:
        a, b = generate_decay_events()
        c = list(find_coincidences(a, b))
        self.assertEqual(36, len(c))
        self.assertEqual(
            [(89, 90), (139, 139), (160, 161), (165, 166)],
            c[:4],
        )

        a, b = map(
            np_array_int32,
            (
                [0, 1, 2, 7, 8, 30],
                [5, 6, 7, 8, 9, 10],
            ),
        )
        c = list(find_coincidences(a, b))
        self.assertEqual(
            [(7, 6), (7, 7), (8, 8)],  # (8, 9)],
            list(map(tuple, np.array(c).tolist())),
        )

    def test_delta_three(self) -> None:
        a, b = map(
            np_array_int32,
            (
                [0, 1, 2, 3, 8, 30],
                [5, 6, 7, 8, 9, 10],
            ),
        )
        c = list(find_coincidences(a, b, max_delta=3))
        self.assertEqual(
            [(2, 5), (3, 5), (8, 5), (8, 6), (8, 7), (8, 8)],
            list(map(tuple, np.array(c).tolist())),
        )

    def test_no_coincidences(self) -> None:
        a, b = map(
            np_array_int32,
            (
                [0, 1, 2],
                [7, 8, 9],
            ),
        )
        c = list(find_coincidences(a, b))
        self.assertEqual(
            [],
            np.array(c).tolist(),
        )
