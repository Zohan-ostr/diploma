#!/usr/bin/env python3
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--fps", type=int, default=30)
args = parser.parse_args()

cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
cap.set(cv2.CAP_PROP_FPS, args.fps)

if not cap.isOpened():
    raise SystemExit(f"Cannot open camera {args.camera}")

print("Opened camera:", args.camera)
print("Actual width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Actual height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Actual fps:", cap.get(cv2.CAP_PROP_FPS))
print("Press q to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Failed to read frame")
        break

    cv2.imshow("camera_test", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
