#!/usr/bin/env python3
import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from upper_body_msgs.msg import PoseLandmarks3D

import mediapipe as mp
import pyrealsense2 as rs


Vec3 = Tuple[float, float, float]

MP_NAMES = {
    0: "nose",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
}

REQUIRED_NAMES = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
]

DRAW_NAMES = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul3(a: Vec3, k: float) -> Vec3:
    return (a[0] * k, a[1] * k, a[2] * k)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a: Vec3) -> float:
    return math.sqrt(max(1e-12, dot(a, a)))


def unit(a: Vec3, fallback: Vec3) -> Vec3:
    n = norm(a)
    if n < 1e-6:
        return fallback
    return (a[0] / n, a[1] / n, a[2] / n)


class RealSenseMediaPipeNode(Node):
    def __init__(self):
        super().__init__("realsense_mediapipe_node")

        self.width = int(self.declare_parameter("width", 640).value)
        self.height = int(self.declare_parameter("height", 480).value)
        self.fps = int(self.declare_parameter("fps", 30).value)
        self.preview = bool(self.declare_parameter("preview", True).value)
        self.preview_mirror = bool(self.declare_parameter("preview_mirror", True).value)
        self.depth_window = int(self.declare_parameter("depth_window", 5).value)
        self.min_depth_m = float(self.declare_parameter("min_depth_m", 0.15).value)
        self.max_depth_m = float(self.declare_parameter("max_depth_m", 6.0).value)

        self.pub = self.create_publisher(PoseLandmarks3D, "/pose/landmarks", 10)
        self.control_pub = self.create_publisher(String, "/teleop/control", 10)

        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)

        self.calibration_countdown_active = False
        self.calibration_deadline_sec = 0.0
        self.calibration_duration_sec = 3.0
        self.last_key_time_sec = 0.0
        self.key_debounce_sec = 0.5

        self.timer = self.create_timer(1.0 / max(1, self.fps), self.tick)

        self.get_logger().info("============================================================")
        self.get_logger().info("REALSENSE RGB-D MEDIAPIPE NODE")
        self.get_logger().info("Publishing /pose/landmarks in BODY frame")
        self.get_logger().info("origin: pelvis midpoint")
        self.get_logger().info("+Z: forward from body plane")
        self.get_logger().info("C: calibrate, R: reset, Q: close preview")
        self.get_logger().info("============================================================")

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def publish_control(self, command: str):
        msg = String()
        msg.data = command
        self.control_pub.publish(msg)
        self.get_logger().info(f"Published /teleop/control: {command}")

    def start_calibration_countdown(self):
        self.calibration_countdown_active = True
        self.calibration_deadline_sec = self.now_sec() + self.calibration_duration_sec

    def draw_countdown(self, frame):
        if not self.calibration_countdown_active:
            return

        remaining = self.calibration_deadline_sec - self.now_sec()
        if remaining <= 0:
            self.calibration_countdown_active = False
            self.publish_control("calibrate")
            cv2.putText(frame, "CALIBRATED", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 4)
            return

        shown = max(1, int(math.ceil(remaining)))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        cv2.putText(frame, "Calibration in", (120, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        cv2.putText(frame, str(shown), (290, 300), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 6)

    def depth_at(self, depth_frame, px: int, py: int) -> float:
        d0 = float(depth_frame.get_distance(px, py))
        if self.min_depth_m <= d0 <= self.max_depth_m:
            return d0

        vals = []
        w = self.depth_window
        for yy in range(max(0, py - w), min(self.height, py + w + 1)):
            for xx in range(max(0, px - w), min(self.width, px + w + 1)):
                d = float(depth_frame.get_distance(xx, yy))
                if self.min_depth_m <= d <= self.max_depth_m:
                    vals.append(d)

        if not vals:
            return 0.0
        return float(np.median(vals))

    def deproject_landmarks(self, image_landmarks, depth_frame):
        intr = depth_frame.profile.as_video_stream_profile().intrinsics

        cam_pts: Dict[str, Vec3] = {}
        vis: Dict[str, float] = {}
        pix: Dict[str, Tuple[int, int]] = {}

        for idx, name in MP_NAMES.items():
            lm = image_landmarks[idx]
            px = int(round(float(lm.x) * self.width))
            py = int(round(float(lm.y) * self.height))
            px = max(0, min(self.width - 1, px))
            py = max(0, min(self.height - 1, py))

            depth_m = self.depth_at(depth_frame, px, py)
            if depth_m <= 0:
                continue

            p = rs.rs2_deproject_pixel_to_point(intr, [float(px), float(py)], depth_m)
            cam_pts[name] = (float(p[0]), float(p[1]), float(p[2]))
            vis[name] = float(getattr(lm, "visibility", 1.0))
            pix[name] = (px, py)

        return cam_pts, vis, pix

    def to_body_frame(self, cam_pts: Dict[str, Vec3]) -> Optional[Dict[str, Vec3]]:
        for name in REQUIRED_NAMES:
            if name not in cam_pts:
                return None

        ls = cam_pts["left_shoulder"]
        rs_ = cam_pts["right_shoulder"]
        lh = cam_pts["left_hip"]
        rh = cam_pts["right_hip"]

        pelvis = mul3(add(lh, rh), 0.5)
        shoulders = mul3(add(ls, rs_), 0.5)

        # Внутренняя система, совместимая с текущим алгоритмом:
        # X: вправо от левого плеча к правому плечу
        # Y: от таза к плечам
        # Z: нормаль к плоскости тела, знак выбирается так, чтобы +Z был вперед.
        ex = unit(sub(rs_, ls), (1.0, 0.0, 0.0))
        ey = unit(sub(shoulders, pelvis), (0.0, -1.0, 0.0))

        ez = unit(cross(ex, ey), (0.0, 0.0, -1.0))

        if "nose" in cam_pts:
            nose_dir = sub(cam_pts["nose"], shoulders)
            if dot(nose_dir, ez) < 0:
                ez = mul3(ez, -1.0)
        else:
            # Для оператора лицом к камере RealSense вперед от тела обычно направлен к камере,
            # то есть примерно против camera-Z.
            if dot(ez, (0.0, 0.0, -1.0)) < 0:
                ez = mul3(ez, -1.0)

        # Ортогонализация X после выбора направления Z.
        ex = unit(cross(ey, ez), ex)

        body_pts: Dict[str, Vec3] = {}
        for name, p in cam_pts.items():
            rel = sub(p, pelvis)
            body_pts[name] = (
                dot(rel, ex),
                dot(rel, ey),
                dot(rel, ez),
            )

        return body_pts

    def publish_pose(self, body_pts: Dict[str, Vec3], vis: Dict[str, float]):
        msg = PoseLandmarks3D()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "body_pelvis_z_forward"
        msg.valid = True

        for name in MP_NAMES.values():
            if name not in body_pts:
                continue
            p = body_pts[name]
            msg.names.append(name)
            msg.x.append(float(p[0]))
            msg.y.append(float(p[1]))
            msg.z.append(float(p[2]))
            msg.visibility.append(float(vis.get(name, 1.0)))

        self.pub.publish(msg)

    def publish_invalid(self):
        msg = PoseLandmarks3D()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "body_pelvis_z_forward"
        msg.valid = False
        self.pub.publish(msg)

    def draw_z(self, frame, body_pts, pix):
        for name in DRAW_NAMES:
            if name not in body_pts or name not in pix:
                continue
            x, y = pix[name]
            z = body_pts[name][2]
            text = f"{name.replace('_', ' ')} z={z:+.3f}"
            cv2.putText(frame, text, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)

    def tick(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        except Exception as exc:
            self.get_logger().warn(f"RealSense frame wait failed: {exc}", throttle_duration_sec=2.0)
            return

        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            self.publish_invalid()
            return

        color = np.asanyarray(color_frame.get_data())
        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        frame = color.copy()

        if result.pose_landmarks:
            self.mp_draw.draw_landmarks(frame, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            cam_pts, vis, pix = self.deproject_landmarks(result.pose_landmarks.landmark, depth_frame)
            body_pts = self.to_body_frame(cam_pts)

            if body_pts is not None:
                self.publish_pose(body_pts, vis)
                self.draw_z(frame, body_pts, pix)
                cv2.putText(frame, "RGB-D body frame: origin=pelvis, +Z=forward",
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            else:
                self.publish_invalid()
                cv2.putText(frame, "Invalid depth for required body points",
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        else:
            self.publish_invalid()
            cv2.putText(frame, "No pose", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        if self.preview:
            self.draw_countdown(frame)
            shown = cv2.flip(frame, 1) if self.preview_mirror else frame
            cv2.imshow("RealSense RGB-D + MediaPipe body Z", shown)

            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                now = self.now_sec()
                if now - self.last_key_time_sec > self.key_debounce_sec:
                    self.last_key_time_sec = now
                    if key in {ord("c"), ord("C"), 241, 209}:
                        self.start_calibration_countdown()
                    elif key in {ord("r"), ord("R"), 234, 202}:
                        self.calibration_countdown_active = False
                        self.publish_control("reset")
                    elif key in {ord("q"), ord("Q"), 233, 201}:
                        self.preview = False
                        self.calibration_countdown_active = False
                        cv2.destroyWindow("RealSense RGB-D + MediaPipe body Z")

    def destroy_node(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RealSenseMediaPipeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
