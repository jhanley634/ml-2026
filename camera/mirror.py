#! /usr/bin/env python

import cv2

from camera.face_finder import face_finder, key, open_camera


def mirror() -> None:
    """
    On a 14 inch MacBook Pro, pops up a window and displays what the laptop camera sees, in B&W.
    """

    cap = open_camera()
    w, h = 0, 0
    want_gray = True

    while (k := key()) != "Q":
        if k == "G":
            want_gray = bool(1 - want_gray)  # toggle

        ret, frame = cap.read()
        assert ret
        frame = cv2.flip(frame, 1)
        if want_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow("Mirror", frame)
        h, w = frame.shape

    print(f"{w} × {h}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Hit G to toggle grayscale, or Q to quit")
    face_finder()
    # mirror()
