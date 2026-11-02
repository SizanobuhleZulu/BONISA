"""Stand-in for SAM 3.1 used only in offline tests: finds the orange ball and the
white sheet by colour. Same return types as bonisa.track."""
import cv2
import numpy as np

from bonisa.track import Box


def _boxes(mask, frame):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if n <= 1:
        return []
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[i, :4]
    return [Box(frame, "0", float(x), float(y), float(x + w), float(y + h), 0.99)]


async def find_in_image(path, prompt):
    img = cv2.imread(str(path))
    return _boxes(np.all(img > 235, axis=2), 0)


async def track_object(path, prompt):
    cap = cv2.VideoCapture(str(path))
    out, i = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        out += _boxes(cv2.inRange(hsv, (5, 120, 120), (25, 255, 255)) > 0, i)
        i += 1
    return out
