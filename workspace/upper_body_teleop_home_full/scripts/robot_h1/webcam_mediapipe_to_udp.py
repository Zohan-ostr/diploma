#!/usr/bin/env python3
import argparse
import json
import socket
import time

import cv2
import mediapipe as mp
import numpy as np


mp_pose = mp.solutions.pose

LANDMARK_NAMES = [lm.name.lower() for lm in mp_pose.PoseLandmark]
CONNECTIONS = list(mp_pose.POSE_CONNECTIONS)

WORK_POINTS = {
    "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    "left_elbow": mp_pose.PoseLandmark.LEFT_ELBOW.value,
    "right_elbow": mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    "left_wrist": mp_pose.PoseLandmark.LEFT_WRIST.value,
    "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST.value,
}


def unit(v):
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return v / n


def project_perp(v, axis):
    au = unit(axis)
    if au is None:
        return None
    p = v - float(np.dot(v, au)) * au
    return unit(p)


def compute_standard_z(points):
    """
    Standard body Z:
      body plane: shoulders + hips
      Z = 0 in the body plane
      Z > 0 forward
      Z < 0 backward

    Forward sign is chosen automatically by nose:
      nose must have positive Z.
    """
    required = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value,
    ]

    if any(i >= len(points) for i in required):
        return {}

    ls = points[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs = points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    lh = points[mp_pose.PoseLandmark.LEFT_HIP.value]
    rh = points[mp_pose.PoseLandmark.RIGHT_HIP.value]

    origin = 0.5 * (ls + rs)

    x_axis = unit(rs - ls)  # left_shoulder -> right_shoulder
    if x_axis is None:
        return {}

    mid_hip = 0.5 * (lh + rh)
    y0 = origin - mid_hip  # bottom -> top

    y_axis = project_perp(y0, x_axis)
    if y_axis is None:
        return {}

    z_axis = unit(np.cross(x_axis, y_axis))
    if z_axis is None:
        return {}

    nose_i = mp_pose.PoseLandmark.NOSE.value
    if nose_i < len(points):
        nose_forward = float(np.dot(points[nose_i] - origin, z_axis))
        if nose_forward < 0.0:
            z_axis = -z_axis

    z_values = {}
    for name, idx in WORK_POINTS.items():
        z_values[name] = float(np.dot(points[idx] - origin, z_axis))

    return z_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--udp_host", default="127.0.0.1")
    parser.add_argument("--udp_port", type=int, default=50060)
    parser.add_argument("--mirror", action="store_true", default=True)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    window_name = "Webcam MediaPipe pose + body-Z"
    frame_i = 0
    t0 = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            show = cv2.flip(frame, 1) if args.mirror else frame.copy()

            names = []
            xs = []
            ys = []
            zs = []
            visibility = []
            valid = False

            if result.pose_landmarks:
                valid = True

                px = []
                for lm in result.pose_landmarks.landmark:
                    u = int(lm.x * w)
                    v = int(lm.y * h)
                    if args.mirror:
                        u = w - 1 - u
                    px.append((u, v))

                # Draw full skeleton, but labels only for working points.
                for a, b in CONNECTIONS:
                    if a < len(px) and b < len(px):
                        ua, va = px[a]
                        ub, vb = px[b]
                        cv2.line(show, (ua, va), (ub, vb), (0, 255, 0), 1, cv2.LINE_AA)

                if result.pose_world_landmarks:
                    world_points = [
                        np.array([lm.x, lm.y, lm.z], dtype=float)
                        for lm in result.pose_world_landmarks.landmark
                    ]
                else:
                    world_points = [
                        np.array([lm.x, lm.y, lm.z], dtype=float)
                        for lm in result.pose_landmarks.landmark
                    ]

                z_body = compute_standard_z(world_points)

                for i, name in enumerate(LANDMARK_NAMES):
                    lm2d = result.pose_landmarks.landmark[i]
                    p3 = world_points[i]

                    names.append(name)
                    xs.append(float(p3[0]))
                    ys.append(float(p3[1]))
                    zs.append(float(p3[2]))
                    visibility.append(float(lm2d.visibility))

                for name, idx in WORK_POINTS.items():
                    u, v = px[idx]
                    if 0 <= u < w and 0 <= v < h:
                        z = z_body.get(name, 0.0)

                        cv2.circle(show, (u, v), 5, (0, 255, 0), -1)
                        cv2.putText(
                            show,
                            f"{name.replace('_', ' ')} z={z:+.2f}",
                            (u + 6, v - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )

                payload = {
                    "stamp": time.time(),
                    "names": names,
                    "x": xs,
                    "y": ys,
                    "z": zs,
                    "visibility": visibility,
                    "valid": valid,
                }

                sock.sendto(
                    json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                    (args.udp_host, args.udp_port),
                )

            frame_i += 1
            fps = frame_i / max(1e-6, time.time() - t0)

            cv2.putText(
                show,
                f"FPS {fps:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, show)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
