# calibrate.py

import cv2
import numpy as np
import sys

clicked = []
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
        clicked.append((x, y))

# Usage: python calibrate.py [video_path]
src = sys.argv[1] if len(sys.argv) > 1 else 0
cap = cv2.VideoCapture(src)
if not cap.isOpened():
    print("❌ cannot open source {!r}".format(src))
    sys.exit(1)

# Skip initial black frames
frame = None
for i in range(1000):
    ret, f = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    if gray.mean() > 80:  # adjust threshold if needed
        frame = f
        break
# Fallback to last frame read
if frame is None and ret:
    frame = f

cap.release()

if frame is None:
    print("❌ could not read any frame from {!r}".format(src))
    sys.exit(1)

cv2.namedWindow("Calibrate ROI")
cv2.setMouseCallback("Calibrate ROI", on_mouse)
print("Click in this order: bottom-left, bottom-right, top-right, top-left")

while len(clicked) < 4:
    vis = frame.copy()
    for i, (x, y) in enumerate(clicked):
        cv2.circle(vis, (x, y), 8, (0,255,0), -1)
        cv2.putText(vis, str(i+1), (x+10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Calibrate ROI", vis)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to abort
        break

cv2.destroyAllWindows()

if len(clicked) == 4:
    print("\nCopy this block into `openpilot_demo.py` as the new SRC:")
    print("SRC = np.float32([")
    for x, y in clicked:
        print("    [{:.1f}, {:.1f}],".format(x, y))
    print("])")
else:
    print("\nCalibration aborted: only {} points clicked.".format(len(clicked)))
