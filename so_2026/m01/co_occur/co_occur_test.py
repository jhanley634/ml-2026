import unittest

import numpy as np

from so_2026.m01.co_occur.co_occur import find_coincidences, generate_decay_events


class CoOccurTest(unittest.TestCase):
    def test_generate_decay_events(self) -> None:
        a_b = generate_decay_events()
        self.assertEqual(2, len(a_b))

    def test_find_coincidences(self) -> None:
        a, b = generate_decay_events()
        c = list(find_coincidences(a, b))
        self.assertEqual(38, len(c))
        self.assertEqual(
            [(86, 87), (89, 90), (139, 139), (160, 161)],
            c[:4],
        )

        a, b = map(
            np.array,
            (
                [0, 1, 2, 7, 8, 30],
                [5, 6, 7, 8, 9, 10],
            ),
        )
        c = list(find_coincidences(a, b))
        self.assertEqual(
            [(7, 6), (7, 7), (7, 8), (8, 8), (8, 9)],
            list(map(tuple, np.array(c).tolist())),
        )

    def test_no_coincidences(self) -> None:
        a, b = map(
            np.array,
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
