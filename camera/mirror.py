#! /usr/bin/env python

import unicodedata

import cv2
from cv2 import THRESH_BINARY, THRESH_OTSU
from cv2.version import opencv_version

from camera.face_finder import FONT, GREEN, FPSCounter, face_finder, key, open_camera

assert opencv_version > "4.12.0"


def mirror(*, want_otsu: bool = False) -> None:
    """
    On a 14 inch MacBook Pro, pops up a window and displays what the laptop camera sees, in B&W.
    """

    cap = open_camera()
    fps_counter = FPSCounter()
    fps = 10.01

    w, h = 0, 0
    want_gray = True

    while (k := key()) != "Q":
        if k == "G":
            want_gray = bool(1 - want_gray)  # toggle

        ret, frame = cap.read()
        assert ret
        frame = cv2.flip(frame, 1)
        if want_gray or want_otsu:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if want_otsu:
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            _, frame = cv2.threshold(blurred, 0, 255, THRESH_BINARY + THRESH_OTSU)

        h, w, *_ = frame.shape
        fps = fps_counter.update()
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 400, h - 60), FONT, 1.9, GREEN, 2)
        cv2.imshow("Mirror", frame)

    mul = unicodedata.lookup("MULTIPLICATION SIGN")  # 00d7
    print(f"{w} {mul} {h}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Hit G to toggle grayscale, or Q to quit")
    face_finder()
    # mirror()
