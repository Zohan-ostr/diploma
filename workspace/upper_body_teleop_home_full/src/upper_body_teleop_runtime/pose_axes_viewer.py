#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Dict, Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from upper_body_msgs.msg import PoseLandmarks3D


def unit(v, eps=1e-9):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < eps:
        return None
    return v / n


def project_perp(v, axis):
    au = unit(axis)
    if au is None:
        return None
    p = np.asarray(v, dtype=float) - float(np.dot(v, au)) * au
    return unit(p)


class PoseAxesViewer(Node):
    MP_INDEX_TO_NAME = {
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

    def __init__(self):
        super().__init__("pose_axes_viewer")

        self.declare_parameter("input_topic", "/pose/landmarks")
        self.declare_parameter("depth_gain", 1.5)
        self.declare_parameter("standard_z_sign", 1.0)

        self.input_topic = self.get_parameter("input_topic").value
        self.depth_gain = float(self.get_parameter("depth_gain").value)
        self.standard_z_sign = float(self.get_parameter("standard_z_sign").value)

        self.last_raw = None
        self.last_body = None
        self.frame_count = 0
        self.last_input_frame_id = ""

        self.sub = self.create_subscription(PoseLandmarks3D, self.input_topic, self.on_msg, 10)
        self.timer = self.create_timer(1.0 / 20.0, self.on_timer)

        self.get_logger().info("============================================================")
        self.get_logger().info("POSE AXES VIEWER")
        self.get_logger().info(f"input_topic: {self.input_topic}")
        self.get_logger().info(f"depth_gain:  {self.depth_gain}")
        self.get_logger().info("left panel:  body X-Y")
        self.get_logger().info("right panel: body Z-Y")
        self.get_logger().info("ESC closes window")
        self.get_logger().info("============================================================")

    def parse_points(self, msg: PoseLandmarks3D) -> Dict[str, np.ndarray]:
        raw = {}

        names = None
        for field in ("names", "landmark_names", "joint_names", "name"):
            if hasattr(msg, field):
                val = getattr(msg, field)
                if val:
                    names = [str(x) for x in val]
                    break

        xs = list(getattr(msg, "x", []))
        ys = list(getattr(msg, "y", []))
        zs = list(getattr(msg, "z", []))

        if names and len(names) == len(xs):
            for i, name in enumerate(names):
                if i < len(xs) and i < len(ys) and i < len(zs):
                    p = np.array([xs[i], ys[i], zs[i]], dtype=float)
                    if np.all(np.isfinite(p)):
                        raw[name] = p
        else:
            for idx, name in self.MP_INDEX_TO_NAME.items():
                if idx < len(xs) and idx < len(ys) and idx < len(zs):
                    p = np.array([xs[idx], ys[idx], zs[idx]], dtype=float)
                    if np.all(np.isfinite(p)):
                        raw[name] = p

        return raw


    def apply_camera_depth_z_to_body_points(self, raw, body_pts):
        """
        X/Y — из локального базиса тела.
        Z — строго из карты глубины камеры относительно плеча.
        """
        fixed = {k: np.array(v, dtype=float).copy() for k, v in body_pts.items()}

        for side in ("left", "right"):
            sh = f"{side}_shoulder"
            el = f"{side}_elbow"
            wr = f"{side}_wrist"

            if sh not in raw or sh not in fixed:
                continue

            z0 = float(raw[sh][2])
            fixed[sh][2] = 0.0

            if el in raw and el in fixed:
                fixed[el][2] = (float(raw[el][2]) - z0) * self.depth_gain

            if wr in raw and wr in fixed:
                fixed[wr][2] = (float(raw[wr][2]) - z0) * self.depth_gain

        return fixed


    def make_depth_relative_to_side_shoulder(self, raw):
        fixed = {k: np.array(v, dtype=float).copy() for k, v in raw.items()}

        for side in ("left", "right"):
            sh = f"{side}_shoulder"
            el = f"{side}_elbow"
            wr = f"{side}_wrist"

            if sh not in fixed:
                continue

            z0 = float(fixed[sh][2])
            fixed[sh][2] = 0.0

            if el in fixed:
                fixed[el][2] = (float(fixed[el][2]) - z0) * self.depth_gain
            if wr in fixed:
                fixed[wr][2] = (float(fixed[wr][2]) - z0) * self.depth_gain

        return fixed

    def to_body_frame(self, raw):
        if "left_shoulder" not in raw or "right_shoulder" not in raw:
            return {}

        ls = raw["left_shoulder"]
        rs = raw["right_shoulder"]
        origin = 0.5 * (ls + rs)

        x_axis = unit(rs - ls)
        if x_axis is None:
            return {}

        if "left_hip" in raw and "right_hip" in raw:
            mid_hip = 0.5 * (raw["left_hip"] + raw["right_hip"])
            y0 = origin - mid_hip
        else:
            y0 = np.array([0.0, 1.0, 0.0], dtype=float)

        y_axis = project_perp(y0, x_axis)
        if y_axis is None:
            y_axis = project_perp(np.array([0.0, 1.0, 0.0], dtype=float), x_axis)
        if y_axis is None:
            return {}

        z_axis = unit(np.cross(x_axis, y_axis))
        if z_axis is None:
            return {}

        z_axis = self.standard_z_sign * z_axis

        if "nose" in raw:
            nose_z = float(np.dot(raw["nose"] - origin, z_axis))
            if nose_z < 0.0:
                z_axis = -z_axis

        body = {}
        for name, p in raw.items():
            v = p - origin
            body[name] = np.array([
                float(np.dot(v, x_axis)),
                float(np.dot(v, y_axis)),
                float(np.dot(v, z_axis)),
            ], dtype=float)

        return body

    def on_msg(self, msg):
        if hasattr(msg, "valid") and not msg.valid:
            return

        self.last_input_frame_id = str(getattr(msg.header, "frame_id", ""))

        raw = self.parse_points(msg)

        # Если камера уже публикует X/Y в body frame и Z из RealSense depth,
        # используем точки напрямую, без повторного пересчёта и без повторного
        # применения depth_gain.
        if self.last_input_frame_id == "body_pelvis_xy_camera_depth_z":
            body = {k: np.array(v, dtype=float).copy() for k, v in raw.items()}
        else:
            body = self.to_body_frame(raw)
            body = self.apply_camera_depth_z_to_body_points(raw, body)

        self.last_raw = raw
        self.last_body = body
        self.frame_count += 1

    def draw_arm(self, img, pts, side, panel, color):
        sh = pts.get(f"{side}_shoulder")
        el = pts.get(f"{side}_elbow")
        wr = pts.get(f"{side}_wrist")

        if sh is None or el is None or wr is None:
            return

        if panel == "xy":
            cx, cy = 250, 360
            scale = 260
            def P(p):
                return int(cx + p[0] * scale), int(cy - p[1] * scale)
            title = "BODY X-Y: shoulder-elbow-wrist"
        else:
            cx, cy = 760, 360
            scale = 260
            def P(p):
                return int(cx + p[2] * scale), int(cy - p[1] * scale)
            title = "BODY Z-Y: depth/up view"

        ps = P(sh)
        pe = P(el)
        pw = P(wr)

        cv2.line(img, ps, pe, color, 3)
        cv2.line(img, pe, pw, color, 3)
        cv2.circle(img, ps, 6, color, -1)
        cv2.circle(img, pe, 6, color, -1)
        cv2.circle(img, pw, 6, color, -1)

        cv2.putText(img, side[0].upper() + "S", (ps[0] + 6, ps[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.putText(img, side[0].upper() + "E", (pe[0] + 6, pe[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.putText(img, side[0].upper() + "W", (pw[0] + 6, pw[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        v1 = el - sh
        v2 = wr - el
        txt_y = 585 if side == "left" else 625
        x0 = 20 if panel == "xy" else 530

        cv2.putText(
            img,
            f"{side}: SE=({v1[0]:+.2f},{v1[1]:+.2f},{v1[2]:+.2f})  EW=({v2[0]:+.2f},{v2[1]:+.2f},{v2[2]:+.2f})",
            (x0, txt_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            color,
            1,
            cv2.LINE_AA,
        )

    def on_timer(self):
        img = np.zeros((700, 1040, 3), dtype=np.uint8)

        cv2.putText(img, "Pose axes viewer: shoulder->elbow and elbow->wrist", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(img, f"depth_gain={self.depth_gain:.2f} | ESC to close", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.rectangle(img, (15, 95), (505, 560), (80, 80, 80), 1)
        cv2.rectangle(img, (525, 95), (1025, 560), (80, 80, 80), 1)

        cv2.putText(img, "BODY X-Y", (25, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        cv2.putText(img, "CAMERA DEPTH Z-Y  (right = larger relative depth)", (535, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

        # Axes
        cv2.arrowedLine(img, (250, 360), (370, 360), (120, 120, 120), 1)
        cv2.arrowedLine(img, (250, 360), (250, 240), (120, 120, 120), 1)
        cv2.putText(img, "+X", (375, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        cv2.putText(img, "+Y", (255, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        cv2.arrowedLine(img, (760, 360), (880, 360), (120, 120, 120), 1)
        cv2.arrowedLine(img, (760, 360), (760, 240), (120, 120, 120), 1)
        cv2.putText(img, "+Z forward", (885, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        cv2.putText(img, "+Y", (765, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        if self.last_body:
            self.draw_arm(img, self.last_body, "left", "xy", (0, 220, 255))
            self.draw_arm(img, self.last_body, "right", "xy", (0, 255, 120))
            self.draw_arm(img, self.last_body, "left", "zy", (0, 220, 255))
            self.draw_arm(img, self.last_body, "right", "zy", (0, 255, 120))

            cv2.putText(img, f"frames: {self.frame_count}", (840, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        else:
            cv2.putText(img, "Waiting for /pose/landmarks...", (330, 350),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2)

        cv2.imshow("H1 pose axes viewer", img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            rclpy.shutdown()


def main():
    rclpy.init()
    node = PoseAxesViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
