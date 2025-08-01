import cv2
import cv2.data


def _key() -> str:
    return chr(cv2.waitKey(1) & 0xFF).upper()


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
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), VIOLET, 4)

        cv2.imshow("Face Finder", frame)

    cap.release()
    cv2.destroyAllWindows()
