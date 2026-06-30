"""
webcam_test.py — ClearRoute visual test
Runs YOLO11n on the webcam in real time and shows bounding boxes.
Press Q to quit.
"""

import cv2
import time
from ultralytics import YOLO

# Load the model — run this script from the project root so the path resolves
model = YOLO("models/yolo11n.pt")

# Open the default webcam (index 0 = built-in camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    print("Make sure no other app (Teams, Zoom, etc.) is using the camera.")
    raise SystemExit(1)

print("Webcam opened. Press Q in the video window to quit.")

# Variables to calculate FPS
prev_time = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        print("ERROR: Could not read frame from webcam.")
        break

    # Run YOLO on the current frame — returns annotated results
    results = model(frame, verbose=False)

    # Grab the annotated frame (bounding boxes + labels drawn by YOLO)
    annotated = results[0].plot()

    # ── FPS counter ───────────────────────────────────────────────────────────
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    # Draw FPS in the top-left corner
    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (10, 30),                        # position (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,                             # font size
        (0, 255, 0),                     # green text
        2,                               # line thickness
    )

    # Show the frame in a window
    cv2.imshow("ClearRoute — Webcam Test (press Q to quit)", annotated)

    # Wait 1 ms for a keypress; quit if Q is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")
