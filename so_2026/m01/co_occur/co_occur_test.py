import unittest

from so_2026.m01.co_occur.co_occur import generate_decay_events


class CoOccurTest(unittest.TestCase):
    def test_generate_decay_events(self) -> None:
        a_b = generate_decay_events()
        self.assertEqual(2, len(a_b))
