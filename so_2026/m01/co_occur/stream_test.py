import unittest

from so_2026.m01.co_occur.co_occur import generate_decay_events


class StreamTest(unittest.TestCase):
    def test_generate(self) -> None:
        a, b = generate_decay_events()

        self.assertGreater(len(a), 0)
        self.assertGreater(len(b), 0)
