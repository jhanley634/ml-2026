import cv2
import cv2.data
import numpy as np
from numpy._typing import NDArray


def _key() -> str:
    return chr(cv2.waitKey(1) & 0xFF).upper()


def _order_by_size(rectangle: NDArray[np.int32]) -> int:
    r = rectangle
    assert r.shape == (4,)  # x, y, w, h
    assert r.dtype == np.dtype(np.int32)
    assert r[2] == r[3]
    return int(r[3])


MAX_FACES = 2

VIOLET = (255, 0, 255)  # BGR


def face_finder() -> None:
    """
    Draws a violet bounding box around the single face in the captured camera image.
    """

    cap = cv2.VideoCapture(0)
    assert cap

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    while _key() != "Q":
        ret, frame = cap.read()
        assert ret
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = np.array(face_cascade.detectMultiScale(gray, 1.1, 4))
        faces2 = sorted(faces, reverse=True, key=_order_by_size)[:MAX_FACES]

        for x, y, w, h in faces2:
            cv2.rectangle(frame, (x, y), (x + w, y + h), VIOLET, 4)

        cv2.imshow("Face Finder", frame)

    cap.release()
    cv2.destroyAllWindows()
