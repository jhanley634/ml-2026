#! /usr/bin/env python

import cv2

from camera.face_finder import FONT, GREEN, FPSCounter, key, open_camera


def mirror() -> None:
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
        if want_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        h, w, *_ = frame.shape
        fps = fps_counter.update()
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 400, h - 60), FONT, 1.9, GREEN, 2)
        cv2.imshow("Mirror", frame)

    print(f"{w} × {h}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Hit G to toggle grayscale, or Q to quit")
    face_finder()
    # mirror()
