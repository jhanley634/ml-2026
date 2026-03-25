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

        a, b = map(
            np.array,
            (
                [0, 1, 7],
                [6, 7, 8],
            ),
        )
        c = list(find_coincidences(a, b))
        self.assertEqual(
            [(7, 6), (7, 7), (7, 8)],
            list(map(tuple, np.array(c).tolist())),
        )
