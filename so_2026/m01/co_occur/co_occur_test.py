import unittest

from so_2026.m01.co_occur.co_occur import foo


class CoOccurTest(unittest.TestCase):
    def test_foo(self) -> None:
        self.assertIsNone(foo())
