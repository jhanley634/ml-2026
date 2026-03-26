import unittest
from typing import TYPE_CHECKING

from so_2026.m01.co_occur.co_occur import generate_decay_events
from so_2026.m01.co_occur.stream import merge_event_streams, window

if TYPE_CHECKING:

    import numpy as np
    from numpy.typing import NDArray


class StreamTest(unittest.TestCase):

    def __init__(self, method_name: str) -> None:
        self.a: NDArray[np.int32]
        self.b: NDArray[np.int32]
        super().__init__(method_name)

    def setUp(self) -> None:
        a, b = generate_decay_events()
        self.assertEqual(len(a), len(b))
        a = a[:80]
        self.a = a
        self.b = b
        return super().setUp()

    def test_merge_streams(self) -> None:

        events = list(merge_event_streams(self.a, self.b))
        self.assertEqual(
            events,
            sorted(events),
        )

    def test_window(self, window_size: int = 3) -> None:

        merged = merge_event_streams(self.a, self.b)
        for e, s_to_e in window(merged, window_size=window_size):
            for i in range(window_size):
                if event := s_to_e.get(e.stamp + i):
                    print(event)
