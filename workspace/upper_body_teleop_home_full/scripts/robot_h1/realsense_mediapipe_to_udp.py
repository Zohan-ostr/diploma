#!/usr/bin/env python3
import argparse
import json
import socket
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp


LANDMARKS = [
    ("nose", mp.solutions.pose.PoseLandmark.NOSE),
    ("left_eye_inner", mp.solutions.pose.PoseLandmark.LEFT_EYE_INNER),
    ("left_eye", mp.solutions.pose.PoseLandmark.LEFT_EYE),
    ("left_eye_outer", mp.solutions.pose.PoseLandmark.LEFT_EYE_OUTER),
    ("right_eye_inner", mp.solutions.pose.PoseLandmark.RIGHT_EYE_INNER),
    ("right_eye", mp.solutions.pose.PoseLandmark.RIGHT_EYE),
    ("right_eye_outer", mp.solutions.pose.PoseLandmark.RIGHT_EYE_OUTER),
    ("left_ear", mp.solutions.pose.PoseLandmark.LEFT_EAR),
    ("right_ear", mp.solutions.pose.PoseLandmark.RIGHT_EAR),
    ("mouth_left", mp.solutions.pose.PoseLandmark.MOUTH_LEFT),
    ("mouth_right", mp.solutions.pose.PoseLandmark.MOUTH_RIGHT),

    ("left_shoulder", mp.solutions.pose.PoseLandmark.LEFT_SHOULDER),
    ("right_shoulder", mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER),
    ("left_elbow", mp.solutions.pose.PoseLandmark.LEFT_ELBOW),
    ("right_elbow", mp.solutions.pose.PoseLandmark.RIGHT_ELBOW),
    ("left_wrist", mp.solutions.pose.PoseLandmark.LEFT_WRIST),
    ("right_wrist", mp.solutions.pose.PoseLandmark.RIGHT_WRIST),

    ("left_pinky", mp.solutions.pose.PoseLandmark.LEFT_PINKY),
    ("right_pinky", mp.solutions.pose.PoseLandmark.RIGHT_PINKY),
    ("left_index", mp.solutions.pose.PoseLandmark.LEFT_INDEX),
    ("right_index", mp.solutions.pose.PoseLandmark.RIGHT_INDEX),
    ("left_thumb", mp.solutions.pose.PoseLandmark.LEFT_THUMB),
    ("right_thumb", mp.solutions.pose.PoseLandmark.RIGHT_THUMB),

    ("left_hip", mp.solutions.pose.PoseLandmark.LEFT_HIP),
    ("right_hip", mp.solutions.pose.PoseLandmark.RIGHT_HIP),
    ("left_knee", mp.solutions.pose.PoseLandmark.LEFT_KNEE),
    ("right_knee", mp.solutions.pose.PoseLandmark.RIGHT_KNEE),
    ("left_ankle", mp.solutions.pose.PoseLandmark.LEFT_ANKLE),
    ("right_ankle", mp.solutions.pose.PoseLandmark.RIGHT_ANKLE),
    ("left_heel", mp.solutions.pose.PoseLandmark.LEFT_HEEL),
    ("right_heel", mp.solutions.pose.PoseLandmark.RIGHT_HEEL),
    ("left_foot_index", mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX),
    ("right_foot_index", mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX),
]


def median_depth_m(depth_frame, u, v, radius=2):
    vals = []
    w = depth_frame.get_width()
    h = depth_frame.get_height()

    for yy in range(max(0, v - radius), min(h, v + radius + 1)):
        for xx in range(max(0, u - radius), min(w, u + radius + 1)):
            d = depth_frame.get_distance(xx, yy)
            if 0.05 < d < 10.0:
                vals.append(d)

    if not vals:
        return 0.0

    return float(np.median(vals))


def ray_from_pixel(intr, u, v):
    # Unit ray in RealSense camera coordinates.
    p = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], 1.0)
    r = np.array(p, dtype=float)
    n = np.linalg.norm(r)
    if n < 1e-9:
        return None
    return r / n


def point_on_ray_with_distance_from_center(ray, center, length, preferred_depth=0.0):
    """
    Find P = t * ray such that ||P - center|| = length.

    If two intersections exist, choose the one closer to preferred_depth if available;
    otherwise choose the positive one nearer to current center depth.
    """
    ray = np.array(ray, dtype=float)
    center = np.array(center, dtype=float)
    length = float(length)

    # ||t r - c||^2 = L^2
    # t^2 - 2 t dot(r,c) + ||c||^2 - L^2 = 0
    b = -2.0 * float(np.dot(ray, center))
    c = float(np.dot(center, center) - length * length)
    disc = b * b - 4.0 * c

    if disc < 0.0:
        return None

    s = float(np.sqrt(max(0.0, disc)))
    roots = [(-b - s) / 2.0, (-b + s) / 2.0]
    roots = [t for t in roots if t > 0.05]

    if not roots:
        return None

    if preferred_depth > 0.05:
        t = min(roots, key=lambda x: abs(x - preferred_depth))
    else:
        t = min(roots, key=lambda x: abs(x - float(center[2])))

    return ray * t


def rs_to_pub_xyz(xyz_rs):
    # RealSense camera: X right, Y down, Z forward.
    # Published pseudo-MediaPipe compatible coordinates:
    # p.x = -X_rs, p.y = Y_rs, p.z = -Z_rs.
    return [-float(xyz_rs[0]), float(xyz_rs[1]), -float(xyz_rs[2])]


def draw_label(img, text, xy, color=(0, 255, 0)):
    cv2.putText(
        img,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--udp_host", default="127.0.0.1")
    parser.add_argument("--udp_port", type=int, default=50060)
    parser.add_argument("--window", default="RealSense RGB + Depth + MediaPipe")
    parser.add_argument("--depth_radius", type=int, default=2)
    parser.add_argument("--shoulder_depth_radius", type=int, default=4)
    parser.add_argument("--elbow_depth_radius", type=int, default=4)
    parser.add_argument("--wrist_depth_radius", type=int, default=3)
    parser.add_argument("--model_complexity", type=int, default=1)
    parser.add_argument("--mirror", action="store_true", default=True)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)

    print("============================================================")
    print(" REALSENSE + MEDIAPIPE -> UDP POSE LANDMARKS")
    print("============================================================")
    print(f"profile:     depth z16 + color rgb8 {args.width}x{args.height}@{args.fps}")
    print(f"udp:         {args.udp_host}:{args.udp_port}")
    print("============================================================")

    profile = pipeline.start(config)

    dev = profile.get_device()
    print("Device:", dev.get_info(rs.camera_info.name))
    print("Serial:", dev.get_info(rs.camera_info.serial_number))
    try:
        print("USB:", dev.get_info(rs.camera_info.usb_type_descriptor))
    except Exception:
        pass

    align = rs.align(rs.stream.color)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_i = 0
    t0 = time.time()

    # Calibrated human upper-arm lengths in RealSense meters.
    # They are estimated online from stable raw depth first, then used to reconstruct
    # elbows from 2D ray + fixed shoulder-elbow length.
    upper_len_calib = {
        "left": None,
        "right": None,
    }
    upper_len_samples = {
        "left": [],
        "right": [],
    }
    max_len_samples = 90

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            color_rgb = np.asanyarray(color_frame.get_data())
            color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)

            depth = np.asanyarray(depth_frame.get_data())

            intr = depth_frame.profile.as_video_stream_profile().intrinsics

            result = pose.process(color_rgb)

            names = []
            xs = []
            ys = []
            zs = []
            visibility = []

            valid = False

            if result.pose_landmarks:
                valid = True

                mp_drawing.draw_landmarks(
                    color_bgr,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )

                h, w = color_bgr.shape[:2]

                # First pass: collect pixels, raw depth points, and rays.
                px = {}
                raw_rs = {}
                raw_depth = {}
                rays = {}
                lm_visibility = {}

                for name, idx in LANDMARKS:
                    lm = result.pose_landmarks.landmark[idx]
                    u = int(lm.x * w)
                    v = int(lm.y * h)

                    lm_visibility[name] = float(lm.visibility)
                    px[name] = (u, v)

                    xyz_rs = np.array([0.0, 0.0, 0.0], dtype=float)
                    d_m = 0.0

                    if 0 <= u < w and 0 <= v < h:
                        if "shoulder" in name:
                            radius = args.shoulder_depth_radius
                        elif "elbow" in name:
                            radius = args.elbow_depth_radius
                        elif "wrist" in name:
                            radius = args.wrist_depth_radius
                        else:
                            radius = args.depth_radius

                        d_m = median_depth_m(depth_frame, u, v, radius=radius)

                        r = ray_from_pixel(intr, u, v)
                        if r is not None:
                            rays[name] = r

                        if d_m > 0.0:
                            xyz_rs = np.array(
                                rs.rs2_deproject_pixel_to_point(
                                    intr,
                                    [float(u), float(v)],
                                    float(d_m),
                                ),
                                dtype=float,
                            )

                    raw_rs[name] = xyz_rs
                    raw_depth[name] = d_m

                # Calibrate upper arm length from raw depth when both shoulder and elbow are valid.
                for side in ("left", "right"):
                    sh_name = f"{side}_shoulder"
                    el_name = f"{side}_elbow"

                    if raw_depth.get(sh_name, 0.0) > 0.0 and raw_depth.get(el_name, 0.0) > 0.0:
                        L = float(np.linalg.norm(raw_rs[el_name] - raw_rs[sh_name]))

                        # Human upper-arm length plausible range.
                        if 0.12 < L < 0.55:
                            samples = upper_len_samples[side]
                            samples.append(L)
                            if len(samples) > max_len_samples:
                                samples.pop(0)
                            if len(samples) >= 20:
                                upper_len_calib[side] = float(np.median(samples))

                # Second pass: publish corrected 3D.
                corrected_rs = dict(raw_rs)

                for side in ("left", "right"):
                    sh_name = f"{side}_shoulder"
                    el_name = f"{side}_elbow"

                    L = upper_len_calib.get(side)
                    sh = raw_rs.get(sh_name)
                    ray_el = rays.get(el_name)
                    d_pref = raw_depth.get(el_name, 0.0)

                    if L is not None and sh is not None and ray_el is not None:
                        el_corr = point_on_ray_with_distance_from_center(
                            ray_el,
                            sh,
                            L,
                            preferred_depth=d_pref,
                        )
                        if el_corr is not None:
                            corrected_rs[el_name] = el_corr

                for name, idx in LANDMARKS:
                    u, v = px.get(name, (-1, -1))
                    xyz_rs = corrected_rs.get(name, np.array([0.0, 0.0, 0.0], dtype=float))
                    d_m = raw_depth.get(name, 0.0)

                    x_pub, y_pub, z_pub = rs_to_pub_xyz(xyz_rs)

                    names.append(name)
                    xs.append(x_pub)
                    ys.append(y_pub)
                    zs.append(z_pub)
                    visibility.append(float(lm_visibility.get(name, 0.0)) if d_m > 0.0 else 0.0)

                    if name in (
                        "left_shoulder", "right_shoulder",
                        "left_elbow", "right_elbow",
                        "left_wrist", "right_wrist",
                    ) and 0 <= u < w and 0 <= v < h:
                        color = (0, 255, 0)
                        label_extra = f"{d_m:.2f}m"

                        if "elbow" in name:
                            side = "left" if name.startswith("left") else "right"
                            if upper_len_calib.get(side) is not None:
                                color = (0, 255, 255)
                                label_extra = f"L={upper_len_calib[side]:.2f}"

                        cv2.circle(color_bgr, (u, v), 5, color, -1)
                        draw_label(color_bgr, f"{name}: {label_extra}", (u + 5, v - 5), color)

                payload = {
                    "stamp": time.time(),
                    "names": names,
                    "x": xs,
                    "y": ys,
                    "z": zs,
                    "visibility": visibility,
                    "valid": valid,
                }

                data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                sock.sendto(data, (args.udp_host, args.udp_port))

            frame_i += 1
            fps = frame_i / max(1e-6, time.time() - t0)

            draw_label(color_bgr, f"FPS: {fps:.1f}", (10, 25), (0, 255, 255))
            draw_label(color_bgr, f"UDP: {args.udp_host}:{args.udp_port}", (10, 50), (0, 255, 255))

            show = cv2.resize(color_bgr, (640, 480))

            # Mirror only the displayed image for operator convenience.
            # Important: published 3D coordinates are NOT mirrored.
            if args.mirror:
                show = cv2.flip(show, 1)

            cv2.imshow(args.window, show)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
