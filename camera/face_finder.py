from time import time
from typing import TYPE_CHECKING

import cv2
import cv2.data
import numpy as np

if TYPE_CHECKING:
    from numpy._typing import NDArray


class FPSCounter:
    def __init__(self, samples: int = 30):
        self.samples = samples
        self.start = time()
        self.count = 0
        self.fps = 10.0

    def update(self) -> float:
        self.count += 1
        if self.count >= self.samples:
            now = time()
            self.fps = self.samples / (now - self.start)
            self.start = now
            self.count = 0
        return self.fps


def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0)
    assert cap.isOpened()
    return cap


def key() -> str:
    return chr(cv2.waitKey(1) & 0xFF).upper()


def order_by_size(rectangle: NDArray[np.int32]) -> int:
    r = rectangle
    assert r.shape == (4,)  # x, y, w, h
    assert r[2] == r[3], r
    return int(r[3])


SMOOTH_FRAC = 0.02
MAX_MOVE = 10
MAX_FACES = 2
VIOLET = (255, 0, 255)  # BGR
GREEN = (0, 255, 0)
FRAME_INTERVAL = 30  # we do 10 FPS, so recompute FPS every three seconds or so
FONT = cv2.FONT_HERSHEY_SIMPLEX


def face_finder() -> None:
    """
    Draws a violet bounding box around the single face in the captured camera image.
    """

    cap = open_camera()
    face_cascade = cv2.CascadeClassifier(
        f"{cv2.data.haarcascades}haarcascade_frontalface_default.xml",
    )
    fps_counter = FPSCounter()
    prev_rect = None

    while key() != "Q":
        ret, frame = cap.read()
        assert ret
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = np.array(face_cascade.detectMultiScale(gray, 1.1, 4))
        faces2 = sorted(faces, reverse=True, key=order_by_size)[:MAX_FACES]
        if len(faces2) > 0:
            current_rect = faces2[0]

            if prev_rect is not None:
                smoothed = SMOOTH_FRAC * (current_rect - prev_rect)
                x, y, w, h = map(int, np.round(prev_rect + smoothed))
                cv2.rectangle(frame, (x, y), (x + w, y + h), VIOLET, 4)

            prev_rect = current_rect

        cv2.putText(frame, f"FPS: {fps_counter.update():.1f}", (100, 200), FONT, 1.9, GREEN, 2)
        cv2.imshow("Face Finder", frame)

    cap.release()
    cv2.destroyAllWindows()
