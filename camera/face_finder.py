import contextlib
import queue
from dataclasses import dataclass
from queue import Queue
from threading import Event, Thread
from time import time
from typing import Any

import cv2
import cv2.data
import numpy as np
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
    assert r[2] == r[3], r  # square
    return int(r[3])


SMOOTH_FRAC = 0.02
MAX_MOVE = 10
MAX_FACES = 2
VIOLET = (255, 0, 255)  # BGR
GREEN = (0, 255, 0)
FRAME_INTERVAL = 30  # we do 10 FPS, so recompute FPS every three seconds or so
FONT = cv2.FONT_HERSHEY_SIMPLEX


# Queue to hold frames for face detection
frame_queue: Queue[Any] = Queue(maxsize=2)  # Limit queue size to avoid excessive memory usage


@dataclass
class SmoothedRect:
    rect: tuple[int, int, int, int] | None = None


smooth = SmoothedRect()


def face_detection_thread(stop_event: Event) -> None:
    """
    Background thread for face detection.  Processes frames from the queue.
    """

    face_cascade = cv2.CascadeClassifier(
        f"{cv2.data.haarcascades}haarcascade_frontalface_default.xml",
    )
    prev_rect = None
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)  # Wait for a frame
            if len(frame.shape) == 3:  # (w, h, 3)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(frame, 1.1, 4)
            faces2 = np.array(faces)
            faces3 = sorted(faces2, reverse=True, key=order_by_size)[:MAX_FACES]

            if len(faces3) > 0:
                current_rect = faces3[0]
                if prev_rect is not None:
                    smoothed = SMOOTH_FRAC * (current_rect - prev_rect)
                    x, y, w, h = np.round(prev_rect + smoothed).astype(int)
                    smooth.rect = (x, y, w, h)
                prev_rect = current_rect

            frame_queue.task_done()
        except queue.Empty:
            # No frame available, continue to next iteration
            pass


def face_finder() -> None:
    """
    Main function to capture camera frames, display them, and draw bounding boxes.
    Uses a background thread for face detection.
    """

    cap = open_camera()
    fps_counter = FPSCounter()
    stop_event = Event()  # Event to signal thread termination

    # Start the face detection thread
    detection_thread = Thread(target=face_detection_thread, args=(stop_event,))
    detection_thread.daemon = True  # Allow the program to exit even if the thread is running
    detection_thread.start()
    want_gray = False

    while (k := key()) != "Q":
        if k == "G":
            want_gray = bool(1 - want_gray)  # toggle

        ret, frame = cap.read()
        assert ret
        frame = cv2.flip(frame, 1)
        if want_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        with contextlib.suppress(queue.Full):
            frame_queue.put(frame, block=False)  # Non-blocking put

        fps = fps_counter.update()
        cv2.putText(frame, f"FPS: {fps:.1f}", (100, 200), FONT, 1.9, GREEN, 2)
        if smooth.rect:
            x, y, w, h = smooth.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), VIOLET, 2)

        cv2.imshow("Face Finder", frame)
        smooth.rect = None  # reset the var, so only the newest rectangle is being drawn

    # Signal the thread to stop
    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()
    frame_queue.join()  # Wait for task_done() calls to complete before exiting
