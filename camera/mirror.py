#! /usr/bin/env python

import cv2


def _key() -> str:
    return chr(cv2.waitKey(1) & 0xFF).upper()


def mirror() -> None:
    """
    On a 14 inch MacBook Pro, pops up a window and displays what the laptop camera sees, in B&W.
    """

    cap = cv2.VideoCapture(0)
    assert cap
    print("Hit Q to quit")

    while _key() != "Q":
        ret, frame = cap.read()
        assert ret

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow("Mirror", gray)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    mirror()
