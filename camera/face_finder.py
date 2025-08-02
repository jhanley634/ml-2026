from time import time

import cv2
import cv2.data
import numpy as np
from numpy._typing import NDArray


def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0)
    assert cap.isOpened()
    return cap


def key() -> str:
    return chr(cv2.waitKey(1) & 0xFF).upper()


def order_by_size(rectangle: NDArray[np.int32]) -> int:
    r = rectangle
    assert r.shape == (4,)  # x, y, w, h
    assert r[2] == r[3]
    return int(r[3])


SMOOTH_FRAC = 0.02
MAX_MOVE = 10
MAX_FACES = 2
VIOLET = (255, 0, 255)  # BGR


def face_finder() -> None:
    """
    Draws a violet bounding box around the single face in the captured camera image.
    """

    cap = open_camera()
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    start_time = time()
    frame_count = 0
    fps = 10.01
    prev_rect = None

    while key() != "Q":
        ret, frame = cap.read()
        assert ret
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = np.array(face_cascade.detectMultiScale(gray, 1.1, 4))
        faces2 = sorted(faces, reverse=True, key=order_by_size)[:MAX_FACES]
        if len(faces2) > 0:

            x2, y2, w2, h2 = current_rect = faces2[0]

            if prev_rect is not None:
                x1, y1, w1, h1 = prev_rect

                # Smooth the movement of the rectangle
                x = int(x1 + SMOOTH_FRAC * (x2 - x1))
                y = int(y1 + SMOOTH_FRAC * (y2 - y1))
                w = int(w1 + SMOOTH_FRAC * (w2 - w1))
                h = int(h1 + SMOOTH_FRAC * (h2 - h1))

                cv2.rectangle(frame, (x, y), (x + w, y + h), VIOLET, 4)

            prev_rect = current_rect

        frame_count += 1
        if frame_count % 30 == 0:
            end_time = time()
            fps = 30 / (end_time - start_time)
            start_time = time()
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (100, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.9,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Face Finder", frame)

    cap.release()
    cv2.destroyAllWindows()
