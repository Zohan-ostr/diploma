#!/usr/bin/env python3
import argparse
import time

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except Exception as e:
    raise SystemExit(f"pyrealsense2 import failed: {e}")

try:
    import mediapipe as mp
    HAS_MP = True
except Exception as e:
    print("WARNING: mediapipe import failed:", e)
    HAS_MP = False


def median_depth_m(depth_frame, u, v, radius=2):
    vals = []
    w = depth_frame.get_width()
    h = depth_frame.get_height()

    for yy in range(max(0, v - radius), min(h, v + radius + 1)):
        for xx in range(max(0, u - radius), min(w, u + radius + 1)):
            d = depth_frame.get_distance(xx, yy)
            if d > 0.05:
                vals.append(d)

    if not vals:
        return 0.0

    return float(np.median(vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--color_format", default="rgb8", choices=["rgb8", "bgr8", "yuyv"])
    parser.add_argument("--no_mediapipe", action="store_true")
    args = parser.parse_args()

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    color_format_map = {
        "rgb8": rs.format.rgb8,
        "bgr8": rs.format.bgr8,
        "yuyv": rs.format.yuyv,
    }
    color_format = color_format_map[args.color_format]

    config.enable_stream(rs.stream.color, args.width, args.height, color_format, args.fps)

    print("Starting RealSense pipeline...")
    profile = pipeline.start(config)

    dev = profile.get_device()
    print("Device:", dev.get_info(rs.camera_info.name))
    print("Serial:", dev.get_info(rs.camera_info.serial_number))
    try:
        print("USB:", dev.get_info(rs.camera_info.usb_type_descriptor))
    except Exception:
        pass

    align = rs.align(rs.stream.color)

    pose = None
    mp_pose = None
    mp_drawing = None

    if HAS_MP and not args.no_mediapipe:
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print("MediaPipe Pose enabled")
    else:
        print("MediaPipe disabled")

    names = []
    if pose is not None and mp_pose is not None:
        names = [
            ("left_shoulder", mp_pose.PoseLandmark.LEFT_SHOULDER),
            ("right_shoulder", mp_pose.PoseLandmark.RIGHT_SHOULDER),
            ("left_elbow", mp_pose.PoseLandmark.LEFT_ELBOW),
            ("right_elbow", mp_pose.PoseLandmark.RIGHT_ELBOW),
            ("left_wrist", mp_pose.PoseLandmark.LEFT_WRIST),
            ("right_wrist", mp_pose.PoseLandmark.RIGHT_WRIST),
        ]

    frame_i = 0
    t0 = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            color_raw = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())

            if args.color_format == "rgb8":
                color = cv2.cvtColor(color_raw, cv2.COLOR_RGB2BGR)
            elif args.color_format == "bgr8":
                color = color_raw
            elif args.color_format == "yuyv":
                color = cv2.cvtColor(color_raw, cv2.COLOR_YUV2BGR_YUY2)
            else:
                color = color_raw

            depth_vis = cv2.convertScaleAbs(depth, alpha=0.03)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

            result = None
            if pose is not None:
                rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)

                if result.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        color,
                        result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )

                    h, w = color.shape[:2]

                    if frame_i % 30 == 0:
                        print()
                        print("===== landmarks with RealSense depth =====")

                    for name, idx in names:
                        lm = result.pose_landmarks.landmark[idx]
                        u = int(lm.x * w)
                        v = int(lm.y * h)

                        if 0 <= u < w and 0 <= v < h:
                            d_m = median_depth_m(depth_frame, u, v, radius=2)

                            if d_m > 0.0:
                                xyz = rs.rs2_deproject_pixel_to_point(
                                    depth_intrin,
                                    [float(u), float(v)],
                                    float(d_m),
                                )
                            else:
                                xyz = [0.0, 0.0, 0.0]

                            cv2.circle(color, (u, v), 5, (0, 255, 0), -1)
                            cv2.putText(
                                color,
                                f"{name}: {d_m:.2f}m",
                                (u + 5, v - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (0, 255, 0),
                                1,
                            )

                            if frame_i % 30 == 0:
                                print(
                                    f"{name:15s} "
                                    f"vis={lm.visibility:.2f} "
                                    f"px=({u:3d},{v:3d}) "
                                    f"depth={d_m:.3f} m "
                                    f"XYZ=({xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f})"
                                )

            frame_i += 1
            fps = frame_i / max(1e-6, time.time() - t0)

            cv2.putText(
                color,
                f"FPS: {fps:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            combined = np.hstack([
                cv2.resize(color, (640, 480)),
                cv2.resize(depth_vis, (640, 480)),
            ])

            cv2.imshow("RealSense RGB + aligned depth + MediaPipe", combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
