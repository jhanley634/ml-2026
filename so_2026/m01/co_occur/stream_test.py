import unittest

from so_2026.m01.co_occur.co_occur import generate_decay_events
from so_2026.m01.co_occur.stream import merge_event_streams


class StreamTest(unittest.TestCase):
    def test_merge_streams(self) -> None:
        a, b = generate_decay_events()
        self.assertEqual(len(a), len(b))
        a = a[:80]

        events = list(merge_event_streams(a, b))
        self.assertEqual(
            events,
            sorted(events),
        )
