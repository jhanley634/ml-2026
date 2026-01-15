import unittest

import numpy as np

from camera.face_finder import order_by_size


class FaceFinderTest(unittest.TestCase):
    def test_face_finder(self) -> None:
        rects = np.array(
            [
                [200, 100, 40, 40],
                [200, 100, 30, 30],
            ],
        )
        s = np.array(rects)
        s.sort(axis=0)

        self.assertTrue(np.array_equal(s, np.array(sorted(rects, key=order_by_size))))
