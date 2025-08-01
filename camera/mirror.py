#! /usr/bin/env python

import cv2

from camera.face_finder import _key, face_finder


def mirror() -> None:
    """
    On a 14 inch MacBook Pro, pops up a window and displays what the laptop camera sees, in B&W.
    """

    cap = cv2.VideoCapture(0)
    assert cap

    while _key() != "Q":
        ret, frame = cap.read()
        assert ret
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow("Mirror", gray)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Hit Q to quit")
    face_finder()
    # mirror()
