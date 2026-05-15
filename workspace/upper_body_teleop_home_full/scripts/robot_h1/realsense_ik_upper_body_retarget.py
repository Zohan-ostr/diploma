#!/usr/bin/env python3
import math
from typing import Dict, Optional

import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool
from upper_body_msgs.msg import PoseLandmarks3D, UpperBodyCommand


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def unit(v: np.ndarray, eps: float = 1e-9) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n < eps:
        return None
    return v / n


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    au = unit(a)
    bu = unit(b)
    if au is None or bu is None:
        return 0.0
    return math.acos(clamp(float(np.dot(au, bu)), -1.0, 1.0))



def signed_angle_about_axis(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> float:
    au = unit(a)
    bu = unit(b)
    xu = unit(axis)

    if au is None or bu is None or xu is None:
        return 0.0

    sin_v = float(np.dot(xu, np.cross(au, bu)))
    cos_v = clamp(float(np.dot(au, bu)), -1.0, 1.0)

    return math.atan2(sin_v, cos_v)



def rotate_about_axis(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Rodrigues rotation formula.
    Rotates vector v around axis by angle.
    """
    au = unit(axis)
    if au is None:
        return v

    v = np.array(v, dtype=float)
    c = math.cos(angle)
    s = math.sin(angle)

    return (
        v * c
        + np.cross(au, v) * s
        + au * float(np.dot(au, v)) * (1.0 - c)
    )


def project_perp(v: np.ndarray, axis: np.ndarray) -> Optional[np.ndarray]:
    axis_u = unit(axis)
    if axis_u is None:
        return None
    p = v - float(np.dot(v, axis_u)) * axis_u
    return unit(p)


def lerp_dir(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    t = clamp(t, 0.0, 1.0)
    v = (1.0 - t) * a + t * b
    vu = unit(v)
    if vu is None:
        return a
    return vu


class ExpFilter:
    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.value = None

    def reset(self, value):
        self.value = np.array(value, dtype=float)

    def update(self, value):
        value = np.array(value, dtype=float)
        if self.value is None:
            self.value = value
        else:
            self.value = self.alpha * value + (1.0 - self.alpha) * self.value
        return self.value.copy()


class RealSenseIKUpperBodyRetarget(Node):
    JOINT_NAMES = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
    ]

    BASE_Q = np.array([
        -0.10, -0.015,  0.017, 1.34,
        -0.10, -0.015, -0.050, 1.32,
    ], dtype=float)

    TPOSE_Q = np.array([
        -0.10,  1.57,  1.74, 1.57,
        -0.10, -1.57, -1.74, 1.57,
    ], dtype=float)

    LOWER = np.array([
        -2.87, -0.34, -1.30, -1.25,
        -2.87, -3.11, -4.45, -1.25,
    ], dtype=float)

    UPPER = np.array([
         2.87,  3.11,  4.45,  2.61,
         2.87,  0.34,  1.30,  2.61,
    ], dtype=float)

    def __init__(self):
        super().__init__("realsense_ik_upper_body_retarget")

        self.declare_parameter("input_topic", "/pose/landmarks")
        self.declare_parameter("output_topic", "/upper_body/command_geom")
        self.declare_parameter("calibration_topic", "/upper_body/start_calibration")

        self.declare_parameter("calibration_frames", 45)

        self.declare_parameter("landmark_alpha", 0.35)
        self.declare_parameter("joint_alpha", 0.45)

        self.declare_parameter("max_joint_step", 0.090)
        self.declare_parameter("yaw_max_step", 0.300)
        self.declare_parameter("elbow_max_step", 0.180)

        self.declare_parameter("pitch_gain", 1.00)
        self.declare_parameter("shoulder_pitch_grid", 61)
        self.declare_parameter("shoulder_roll_grid", 91)
        self.declare_parameter("shoulder_continuity_weight", 0.03)

        # H1 shoulder_pitch has opposite depth effect in the upper half.
        # Below shoulder level, pitch forward is normal.
        # Above shoulder level, pitch forward/backward must be inverted in FK model.
        self.declare_parameter("invert_pitch_forward_above_shoulder", True)
        self.declare_parameter("invert_yaw_forward_above_shoulder", True)
        self.declare_parameter("invert_fabrik_target_forward_above_shoulder", True)
        self.declare_parameter("shoulder_pitch_search_window", 0.75)
        self.declare_parameter("shoulder_roll_search_window", 0.90)
        self.declare_parameter("pitch_gain_upper", 0.45)
        self.declare_parameter("pitch_upper_blend_width", 0.20)
        self.declare_parameter("mirror_pitch_y_above_shoulder", True)
        self.declare_parameter("pitch_y_min_den", 0.20)
        self.declare_parameter("elbow_gain", 1.35)
        self.declare_parameter("elbow_bend_deadzone", 0.08)
        self.declare_parameter("use_calibrated_elbow_bias", True)
        self.declare_parameter("left_elbow_straight", 1.57)
        self.declare_parameter("right_elbow_straight", 1.57)
        self.declare_parameter("min_length_ratio", 0.45)
        self.declare_parameter("max_length_ratio", 1.80)
        self.declare_parameter("min_upper_length_ratio", 0.60)
        self.declare_parameter("max_upper_length_ratio", 1.50)
        self.declare_parameter("min_fore_length_ratio", 0.30)
        self.declare_parameter("max_fore_length_ratio", 2.40)
        self.declare_parameter("elbow_deadzone_rad", 0.20)

        # Рабочие границы yaw:
        # верх уже уменьшен на 0.15 рад: 1.89 -> 1.74
        self.declare_parameter("left_yaw_up", 1.74)
        self.declare_parameter("right_yaw_up", -1.74)
        self.declare_parameter("left_yaw_down", -1.30)
        self.declare_parameter("right_yaw_down", 1.30)

        # Веса поиска yaw.
        self.declare_parameter("yaw_grid_count", 81)
        self.declare_parameter("yaw_continuity_weight", 0.08)

        # Vector yaw mode:
        # shoulder_yaw is calculated as a signed angle between:
        #   yaw_ref vector
        #   projected elbow->wrist vector
        # around the shoulder->elbow axis.
        self.declare_parameter("use_vector_yaw", True)
        self.declare_parameter("yaw_vector_gain", 1.0)
        self.declare_parameter("left_yaw_angle_sign", 1.0)
        self.declare_parameter("right_yaw_angle_sign", -1.0)
        self.declare_parameter("yaw_vector_continuity", 0.15)

        # Robot arm geometry for FABRIK-style wrist matching.
        # These are effective lengths used only for target scaling.
        self.declare_parameter("robot_upper_len", 0.31)
        self.declare_parameter("robot_fore_len", 0.31)
        self.declare_parameter("fabrik_wrist_weight", 1.0)
        self.declare_parameter("fabrik_position_weight", 1.0)

        self.declare_parameter("min_visibility", 0.01)

        # Debug: lock shoulder_yaw to verify elbow placement only.
        self.declare_parameter("lock_shoulder_yaw", True)
        self.declare_parameter("locked_left_yaw", 1.74)
        self.declare_parameter("locked_right_yaw", -1.74)

        # Standard body coordinate system:
        #   X: left shoulder -> right shoulder
        #   Y: bottom -> top
        #   Z: body-plane normal, forward positive
        #
        # If forward/backward is inverted for the current camera source,
        # change standard_z_sign to -1.0 in the launch script.
        self.declare_parameter("use_standard_body_frame", True)
        self.declare_parameter("standard_z_sign", 1.0)

        # Debug mode: isolate elbow placement.
        # Only shoulder_pitch and shoulder_roll are controlled from shoulder->elbow.
        # shoulder_yaw and elbow_pitch are kept fixed.
        self.declare_parameter("elbow_only_debug", True)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.calibration_topic = self.get_parameter("calibration_topic").value

        self.calibration_frames = int(self.get_parameter("calibration_frames").value)

        self.landmark_alpha = float(self.get_parameter("landmark_alpha").value)
        self.joint_alpha = float(self.get_parameter("joint_alpha").value)

        self.max_joint_step = float(self.get_parameter("max_joint_step").value)
        self.yaw_max_step = float(self.get_parameter("yaw_max_step").value)
        self.elbow_max_step = float(self.get_parameter("elbow_max_step").value)

        self.pitch_gain = float(self.get_parameter("pitch_gain").value)
        self.shoulder_pitch_grid = int(self.get_parameter("shoulder_pitch_grid").value)
        self.shoulder_roll_grid = int(self.get_parameter("shoulder_roll_grid").value)
        self.shoulder_continuity_weight = float(self.get_parameter("shoulder_continuity_weight").value)
        self.invert_pitch_forward_above_shoulder = bool(
            self.get_parameter("invert_pitch_forward_above_shoulder").value
        )
        self.invert_yaw_forward_above_shoulder = bool(
            self.get_parameter("invert_yaw_forward_above_shoulder").value
        )
        self.invert_fabrik_target_forward_above_shoulder = bool(
            self.get_parameter("invert_fabrik_target_forward_above_shoulder").value
        )
        self.shoulder_pitch_search_window = float(self.get_parameter("shoulder_pitch_search_window").value)
        self.shoulder_roll_search_window = float(self.get_parameter("shoulder_roll_search_window").value)
        self.pitch_gain_upper = float(self.get_parameter("pitch_gain_upper").value)
        self.pitch_upper_blend_width = float(self.get_parameter("pitch_upper_blend_width").value)
        self.mirror_pitch_y_above_shoulder = bool(self.get_parameter("mirror_pitch_y_above_shoulder").value)
        self.pitch_y_min_den = float(self.get_parameter("pitch_y_min_den").value)
        self.elbow_gain = float(self.get_parameter("elbow_gain").value)
        self.elbow_bend_deadzone = float(self.get_parameter("elbow_bend_deadzone").value)
        self.use_calibrated_elbow_bias = bool(self.get_parameter("use_calibrated_elbow_bias").value)

        self.left_elbow_straight = float(self.get_parameter("left_elbow_straight").value)
        self.right_elbow_straight = float(self.get_parameter("right_elbow_straight").value)
        self.min_length_ratio = float(self.get_parameter("min_length_ratio").value)
        self.max_length_ratio = float(self.get_parameter("max_length_ratio").value)
        self.min_upper_length_ratio = float(self.get_parameter("min_upper_length_ratio").value)
        self.max_upper_length_ratio = float(self.get_parameter("max_upper_length_ratio").value)
        self.min_fore_length_ratio = float(self.get_parameter("min_fore_length_ratio").value)
        self.max_fore_length_ratio = float(self.get_parameter("max_fore_length_ratio").value)
        self.left_elbow_straight = float(self.get_parameter("left_elbow_straight").value)
        self.right_elbow_straight = float(self.get_parameter("right_elbow_straight").value)
        self.elbow_deadzone_rad = float(self.get_parameter("elbow_deadzone_rad").value)

        self.left_yaw_up = float(self.get_parameter("left_yaw_up").value)
        self.right_yaw_up = float(self.get_parameter("right_yaw_up").value)
        self.left_yaw_down = float(self.get_parameter("left_yaw_down").value)
        self.right_yaw_down = float(self.get_parameter("right_yaw_down").value)

        self.yaw_grid_count = int(self.get_parameter("yaw_grid_count").value)
        self.yaw_continuity_weight = float(self.get_parameter("yaw_continuity_weight").value)

        self.use_vector_yaw = bool(self.get_parameter("use_vector_yaw").value)
        self.yaw_vector_gain = float(self.get_parameter("yaw_vector_gain").value)
        self.left_yaw_angle_sign = float(self.get_parameter("left_yaw_angle_sign").value)
        self.right_yaw_angle_sign = float(self.get_parameter("right_yaw_angle_sign").value)
        self.yaw_vector_continuity = float(self.get_parameter("yaw_vector_continuity").value)

        self.robot_upper_len = float(self.get_parameter("robot_upper_len").value)
        self.robot_fore_len = float(self.get_parameter("robot_fore_len").value)
        self.fabrik_wrist_weight = float(self.get_parameter("fabrik_wrist_weight").value)
        self.fabrik_position_weight = float(self.get_parameter("fabrik_position_weight").value)

        self.min_visibility = float(self.get_parameter("min_visibility").value)

        self.lock_shoulder_yaw = bool(self.get_parameter("lock_shoulder_yaw").value)
        self.locked_left_yaw = float(self.get_parameter("locked_left_yaw").value)
        self.locked_right_yaw = float(self.get_parameter("locked_right_yaw").value)
        self.use_standard_body_frame = bool(self.get_parameter("use_standard_body_frame").value)
        self.standard_z_sign = float(self.get_parameter("standard_z_sign").value)
        self.elbow_only_debug = bool(self.get_parameter("elbow_only_debug").value)

        self.pub = self.create_publisher(UpperBodyCommand, self.output_topic, 10)
        self.sub = self.create_subscription(PoseLandmarks3D, self.input_topic, self.on_landmarks, 10)
        self.calib_sub = self.create_subscription(Bool, self.calibration_topic, self.on_calibration, 10)

        self.filters: Dict[str, ExpFilter] = {}

        self.q_filter = ExpFilter(self.joint_alpha)
        self.q_filter.reset(self.TPOSE_Q)
        self.last_q = self.TPOSE_Q.copy()

        self.prev_left_yaw = self.left_yaw_up
        self.prev_right_yaw = self.right_yaw_up

        self.last_pitch_gain_by_side = {
            "left": self.pitch_gain,
            "right": self.pitch_gain,
        }

        self.calibration_requested = False
        self.calibrating = False
        self.rest_ready = False
        self.calibration_count = 0

        self.calib_lengths = {
            "left_upper": [],
            "left_fore": [],
            "right_upper": [],
            "right_fore": [],
        }
        self.human_lengths = {
            "left_upper": None,
            "left_fore": None,
            "right_upper": None,
            "right_fore": None,
        }

        self.calib_bend_angles = {
            "left": [],
            "right": [],
        }
        self.elbow_bend_bias = {
            "left": 0.0,
            "right": 0.0,
        }

        self.last_pts = None
        self.msg_count = 0

        self.timer = self.create_timer(1.0 / 30.0, self.on_timer)

        self.get_logger().info("============================================================")
        self.get_logger().info("REALSENSE TWO-STAGE IK UPPER BODY RETARGET")
        self.get_logger().info(f"input_topic:   {self.input_topic}")
        self.get_logger().info(f"output_topic:  {self.output_topic}")
        self.get_logger().info(f"yaw up L/R:    {self.left_yaw_up} / {self.right_yaw_up}")
        self.get_logger().info(f"yaw down L/R:  {self.left_yaw_down} / {self.right_yaw_down}")
        self.get_logger().info(f"elbow straight L/R: {self.left_elbow_straight} / {self.right_elbow_straight}")
        self.get_logger().info(f"elbow deadzone: {self.elbow_deadzone_rad}")
        self.get_logger().info(f"elbow_only_debug: {self.elbow_only_debug}")
        self.get_logger().info("IK stage 1: shoulder->elbow => shoulder_pitch, shoulder_roll")
        self.get_logger().info("IK stage 2: vector angle elbow_pitch, no wrist IK influence")
        self.get_logger().info("IK stage 3: FABRIK wrist matching => shoulder_yaw only")
        self.get_logger().info("Yaw search is constrained to forward-bending half-plane.")
        self.get_logger().info("============================================================")

    def on_calibration(self, msg: Bool):
        if not msg.data:
            return

        self.calibration_requested = True
        self.calibrating = True
        self.rest_ready = False
        self.calibration_count = 0

        for k in self.calib_lengths:
            self.calib_lengths[k] = []
        for k in self.human_lengths:
            self.human_lengths[k] = None
        for k in self.calib_bend_angles:
            self.calib_bend_angles[k] = []
        for k in self.elbow_bend_bias:
            self.elbow_bend_bias[k] = 0.0

        self.prev_left_yaw = self.left_yaw_up
        self.prev_right_yaw = self.right_yaw_up

        self.last_pitch_gain_by_side = {
            "left": self.pitch_gain,
            "right": self.pitch_gain,
        }

        self.get_logger().info("Manual calibration requested. Holding T-pose while collecting neutral samples...")

    def get_points(self, msg: PoseLandmarks3D):
        raw = {}
        vis = {}

        for i, name in enumerate(msg.names):
            if i < len(msg.x) and i < len(msg.y) and i < len(msg.z):
                raw[name] = np.array([msg.x[i], msg.y[i], msg.z[i]], dtype=float)
                vis[name] = float(msg.visibility[i]) if i < len(msg.visibility) else 1.0

        return raw, vis

    def filtered_point(self, name: str, p: np.ndarray) -> np.ndarray:
        if name not in self.filters:
            self.filters[name] = ExpFilter(self.landmark_alpha)
        return self.filters[name].update(p)

    def mp_to_body(self, p_mp: np.ndarray, origin_mp: np.ndarray) -> np.ndarray:
        p = p_mp - origin_mp

        # Совместимость с нашим RealSense publisher:
        # после преобразования:
        # x = forward
        # y = left
        # z = up
        x_forward = -p[2]
        y_left = p[0]
        z_up = -p[1]

        return np.array([x_forward, y_left, z_up], dtype=float)

    def standardize_points_to_body_frame(self, raw):
        """
        Convert any landmark source to one standard coordinate frame.

        This is source-independent:
          - RealSense depth landmarks
          - webcam MediaPipe world landmarks

        Standard frame:
          origin = middle of shoulders

          X axis:
            left_shoulder -> right_shoulder

          Y axis:
            torso bottom -> top, preferably mid_hips -> mid_shoulders,
            projected perpendicular to X

          Z axis:
            normal to the body plane.
            Z = 0 is the body plane.
            Z > 0 is forward.
            Z < 0 is backward.

        standard_z_sign can be changed to -1 if camera/source gives inverted forward.
        """
        required = ["left_shoulder", "right_shoulder"]
        if not all(k in raw for k in required):
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

        # Make Z forward-positive automatically when face landmarks are available.
        # Nose should be in front of the body plane.
        if "nose" in raw:
            nose_forward = float(np.dot(raw["nose"] - origin, z_axis))
            if nose_forward < 0.0:
                z_axis = -z_axis

        pts = {}
        for name, p in raw.items():
            v = p - origin
            pts[name] = np.array([
                float(np.dot(v, x_axis)),  # X: left shoulder -> right shoulder
                float(np.dot(v, y_axis)),  # Y: bottom -> top
                float(np.dot(v, z_axis)),  # Z: forward/backward from body plane
            ], dtype=float)

        return pts

    def make_body_points(self, raw):
        if self.use_standard_body_frame:
            pts_raw = self.standardize_points_to_body_frame(raw)
        else:
            shoulder_mid = 0.5 * (raw["left_shoulder"] + raw["right_shoulder"])
            pts_raw = {}
            for name, p in raw.items():
                pts_raw[name] = self.mp_to_body(p, shoulder_mid)

        pts = {}
        for name, p in pts_raw.items():
            pts[name] = self.filtered_point(name, p)

        return pts

    def visible_enough(self, vis):
        required = [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
        ]
        return all(vis.get(name, 0.0) >= self.min_visibility for name in required)

    def publish_q(self, q: np.ndarray, frame_id: str):
        msg = UpperBodyCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.joint_names = list(self.JOINT_NAMES)
        msg.position = [float(v) for v in q]
        msg.confidence = [1.0 for _ in q]
        msg.valid = True
        self.pub.publish(msg)

    def smooth_publish_tpose(self):
        target = self.TPOSE_Q.copy()

        delta = target - self.last_q
        step_limits = np.array([
            self.max_joint_step,
            self.max_joint_step,
            self.yaw_max_step,
            self.elbow_max_step,
            self.max_joint_step,
            self.max_joint_step,
            self.yaw_max_step,
            self.elbow_max_step,
        ], dtype=float)

        q = self.last_q + np.clip(delta, -step_limits, step_limits)
        q = self.q_filter.update(q)
        q = np.clip(q, self.LOWER, self.UPPER)

        self.last_q = q.copy()
        self.publish_q(q, "realsense_ik_tpose")

    def on_timer(self):
        if not self.rest_ready:
            self.smooth_publish_tpose()

    def collect_length_sample(self, pts):
        for side in ("left", "right"):
            sh = pts[f"{side}_shoulder"]
            el = pts[f"{side}_elbow"]
            wr = pts[f"{side}_wrist"]

            upper = el - sh
            fore = wr - el

            upper_len = float(np.linalg.norm(upper))
            fore_len = float(np.linalg.norm(fore))

            if upper_len > 1e-4:
                self.calib_lengths[f"{side}_upper"].append(upper_len)
            if fore_len > 1e-4:
                self.calib_lengths[f"{side}_fore"].append(fore_len)

            upper_u = unit(upper)
            fore_u = unit(fore)
            if upper_u is not None and fore_u is not None:
                bend = angle_between(upper_u, fore_u)
                if 0.0 <= bend <= math.pi:
                    self.calib_bend_angles[side].append(float(bend))

    def finish_length_calibration(self):
        for k, vals in self.calib_lengths.items():
            if vals:
                self.human_lengths[k] = float(np.median(vals))

        for side in ("left", "right"):
            vals = self.calib_bend_angles.get(side, [])
            if vals and self.use_calibrated_elbow_bias:
                self.elbow_bend_bias[side] = float(np.median(vals))
            else:
                self.elbow_bend_bias[side] = 0.0

        self.get_logger().info(
            "Human arm length calibration: "
            f"L upper/fore={self.human_lengths['left_upper']:.3f}/"
            f"{self.human_lengths['left_fore']:.3f}, "
            f"R upper/fore={self.human_lengths['right_upper']:.3f}/"
            f"{self.human_lengths['right_fore']:.3f}"
        )
        self.get_logger().info(
            "Elbow bend bias calibration: "
            f"L={self.elbow_bend_bias['left']:.3f}, "
            f"R={self.elbow_bend_bias['right']:.3f}, "
            f"deadzone={self.elbow_bend_deadzone:.3f}"
        )

    def length_geometry_status(self, side: str, upper: np.ndarray, fore: np.ndarray):
        """
        Upper arm is primary: shoulder->elbow must be reliable.
        Forearm is secondary: if elbow->wrist is noisy, keep previous yaw/elbow,
        but do not break shoulder_pitch/roll.
        """
        upper_ref = self.human_lengths.get(f"{side}_upper")
        fore_ref = self.human_lengths.get(f"{side}_fore")

        upper_ok = True
        fore_ok = True
        upper_ratio = 1.0
        fore_ratio = 1.0

        upper_len = float(np.linalg.norm(upper))
        fore_len = float(np.linalg.norm(fore))

        if upper_ref is not None and upper_ref > 1e-6:
            upper_ratio = upper_len / upper_ref
            upper_ok = self.min_upper_length_ratio <= upper_ratio <= self.max_upper_length_ratio

        if fore_ref is not None and fore_ref > 1e-6:
            fore_ratio = fore_len / fore_ref
            fore_ok = self.min_fore_length_ratio <= fore_ratio <= self.max_fore_length_ratio

        if self.msg_count % 60 == 0:
            self.get_logger().info(
                f"{side} length ratio: upper={upper_ratio:.2f} ok={int(upper_ok)}, "
                f"fore={fore_ratio:.2f} ok={int(fore_ok)}"
            )

        return upper_ok, fore_ok

    def compute_shoulder_basis(self, pts):
        """
        Standard dynamic torso basis.

        x:
          left_shoulder -> right_shoulder

        y:
          bottom -> top, preferably mid_hips -> mid_shoulders,
          projected perpendicular to x

        z:
          normal to the torso plane, approximately forward

        Internal body coordinates remain:
          x_body = forward
          y_body = left/right
          z_body = up

        This function returns:
          sx = right direction by shoulders
          sy = up direction by torso
          sz = forward normal
        """
        ls = pts.get("left_shoulder")
        rs = pts.get("right_shoulder")

        if ls is None or rs is None:
            return None

        # User standard:
        # vector x goes from detected left shoulder to detected right shoulder.
        sx = unit(rs - ls)
        if sx is None:
            return None

        if "left_hip" in pts and "right_hip" in pts:
            mid_sh = 0.5 * (ls + rs)
            mid_hip = 0.5 * (pts["left_hip"] + pts["right_hip"])
            sy0 = mid_sh - mid_hip
        else:
            sy0 = np.array([0.0, 0.0, 1.0], dtype=float)

        sy = project_perp(sy0, sx)
        if sy is None:
            sy = project_perp(np.array([0.0, 0.0, 1.0], dtype=float), sx)

        if sy is None:
            return None

        # x right, y up => z is normal.
        # Choose sign so z roughly matches positive body-forward.
        sz = unit(np.cross(sx, sy))
        if sz is None:
            return None

        if float(np.dot(sz, np.array([1.0, 0.0, 0.0], dtype=float))) < 0.0:
            sz = -sz

        return sx, sy, sz

    def effective_pitch_gain(self, vertical: float) -> float:
        """
        Below shoulder level, keep normal pitch gain.
        Above shoulder level, reduce pitch gain so arms do not rotate backward too much.

        vertical = dot(shoulder->elbow, shoulders_y)
          vertical < 0: elbow below shoulders
          vertical > 0: elbow above shoulders
        """
        width = max(1e-6, self.pitch_upper_blend_width)

        # Smooth blend:
        # vertical <= 0       -> t=0
        # vertical >= width   -> t=1
        t = clamp(vertical / width, 0.0, 1.0)

        return (1.0 - t) * self.pitch_gain + t * self.pitch_gain_upper

    def model_upper_dir_from_pitch_roll(self, side: str, pitch_q: float, roll_q: float, pts):
        """
        Forward model for the robot upper-arm direction.

        We use this model for numerical elbow IK:
          candidate shoulder_pitch/roll -> predicted shoulder->elbow direction.

        Basis:
          sx = shoulders_x, right shoulder -> left shoulder
          sy = shoulders_y, torso up
          sz = shoulders_z, torso forward

        roll_mag:
          0      -> arm down
          pi/2   -> arm to side
          pi     -> arm up

        pitch_geom:
          forward/backward rotation in the torso depth direction.
        """
        basis = self.compute_shoulder_basis(pts)
        if basis is None:
            return None

        sx, sy, sz = basis
        side_sign = 1.0 if side == "left" else -1.0

        if side == "left":
            pitch_geom = (pitch_q - self.BASE_Q[0]) / max(1e-6, self.pitch_gain)
        else:
            pitch_geom = (pitch_q - self.BASE_Q[4]) / max(1e-6, self.pitch_gain)

        roll_mag = abs(float(roll_q))

        lateral_out = math.sin(roll_mag) * math.cos(pitch_geom)
        vertical = -math.cos(roll_mag) * math.cos(pitch_geom)
        forward = math.sin(pitch_geom)

        # Upper-half branch correction.
        #
        # Below shoulder level the FK branch works directly.
        # Above shoulder level H1 behaves as if both depth and lateral branch
        # are mirrored. If we invert only forward, pitch improves but roll/yaw
        # still choose the wrong upper branch.
        #
        # Therefore, for roll_mag > pi/2 we mirror both:
        #   forward     -> -forward
        #   lateral_out -> -lateral_out
        #
        # vertical is left unchanged because roll_mag already moves it from
        # down -> side -> up.
        if self.invert_pitch_forward_above_shoulder and roll_mag > (math.pi / 2.0):
            forward = -forward
            lateral_out = -lateral_out

        d = (
            side_sign * lateral_out * sx
            + vertical * sy
            + forward * sz
        )

        return unit(d)

    def model_upper_dir_from_pitch_roll(self, side: str, pitch_q: float, roll_q: float, pts):
        """
        Forward model for the robot upper-arm direction.

        We use this model for numerical elbow IK:
          candidate shoulder_pitch/roll -> predicted shoulder->elbow direction.

        Basis:
          sx = shoulders_x, right shoulder -> left shoulder
          sy = shoulders_y, torso up
          sz = shoulders_z, torso forward

        roll_mag:
          0      -> arm down
          pi/2   -> arm to side
          pi     -> arm up

        pitch_geom:
          forward/backward rotation in the torso depth direction.
        """
        basis = self.compute_shoulder_basis(pts)
        if basis is None:
            return None

        sx, sy, sz = basis
        side_sign = 1.0 if side == "left" else -1.0

        if side == "left":
            pitch_geom = (pitch_q - self.BASE_Q[0]) / max(1e-6, self.pitch_gain)
        else:
            pitch_geom = (pitch_q - self.BASE_Q[4]) / max(1e-6, self.pitch_gain)

        roll_mag = abs(float(roll_q))

        lateral_out = math.sin(roll_mag) * math.cos(pitch_geom)
        vertical = -math.cos(roll_mag) * math.cos(pitch_geom)
        forward = math.sin(pitch_geom)

        # Upper-half branch correction.
        #
        # Below shoulder level the FK branch works directly.
        # Above shoulder level H1 behaves as if both depth and lateral branch
        # are mirrored. If we invert only forward, pitch improves but roll/yaw
        # still choose the wrong upper branch.
        #
        # Therefore, for roll_mag > pi/2 we mirror both:
        #   forward     -> -forward
        #   lateral_out -> -lateral_out
        #
        # vertical is left unchanged because roll_mag already moves it from
        # down -> side -> up.
        if self.invert_pitch_forward_above_shoulder and roll_mag > (math.pi / 2.0):
            forward = -forward
            lateral_out = -lateral_out

        d = (
            side_sign * lateral_out * sx
            + vertical * sy
            + forward * sz
        )

        return unit(d)

    def model_upper_dir_from_pitch_roll(self, side: str, pitch_q: float, roll_q: float, pts):
        """
        Forward model for the robot upper-arm direction.

        We use this model for numerical elbow IK:
          candidate shoulder_pitch/roll -> predicted shoulder->elbow direction.

        Basis:
          sx = shoulders_x, right shoulder -> left shoulder
          sy = shoulders_y, torso up
          sz = shoulders_z, torso forward

        roll_mag:
          0      -> arm down
          pi/2   -> arm to side
          pi     -> arm up

        pitch_geom:
          forward/backward rotation in the torso depth direction.
        """
        basis = self.compute_shoulder_basis(pts)
        if basis is None:
            return None

        sx, sy, sz = basis
        side_sign = 1.0 if side == "left" else -1.0

        if side == "left":
            pitch_geom = (pitch_q - self.BASE_Q[0]) / max(1e-6, self.pitch_gain)
        else:
            pitch_geom = (pitch_q - self.BASE_Q[4]) / max(1e-6, self.pitch_gain)

        roll_mag = abs(float(roll_q))

        lateral_out = math.sin(roll_mag) * math.cos(pitch_geom)
        vertical = -math.cos(roll_mag) * math.cos(pitch_geom)
        forward = math.sin(pitch_geom)

        # Upper-half branch correction.
        #
        # Below shoulder level the FK branch works directly.
        # Above shoulder level H1 behaves as if both depth and lateral branch
        # are mirrored. If we invert only forward, pitch improves but roll/yaw
        # still choose the wrong upper branch.
        #
        # Therefore, for roll_mag > pi/2 we mirror both:
        #   forward     -> -forward
        #   lateral_out -> -lateral_out
        #
        # vertical is left unchanged because roll_mag already moves it from
        # down -> side -> up.
        if self.invert_pitch_forward_above_shoulder and roll_mag > (math.pi / 2.0):
            forward = -forward
            lateral_out = -lateral_out

        d = (
            side_sign * lateral_out * sx
            + vertical * sy
            + forward * sz
        )

        return unit(d)

    def solve_shoulder_pitch_roll_from_basis(self, side: str, upper_u: np.ndarray, pts):
        """
        Analytic elbow placement from the dynamic shoulder basis.

        This is the previous faster solution:
          shoulder->elbow direction -> shoulder_pitch + shoulder_roll

        Fix kept from the later experiments:
          in the upper hemisphere the shoulder branch is mirrored.
          Therefore for vertical > 0 we invert both:
            forward component
            lateral component

        This avoids the case:
          human forward-up -> robot backward-up.
        """
        basis = self.compute_shoulder_basis(pts)
        if basis is None:
            return self.solve_shoulder_pitch_roll(side, upper_u)

        sx, sy, sz = basis

        lateral = float(np.dot(upper_u, sx))
        vertical = float(np.dot(upper_u, sy))
        forward = float(np.dot(upper_u, sz))

        # sx is standardized as:
        #   left_shoulder -> right_shoulder
        #
        # Therefore outward direction is:
        #   left arm  -> -sx
        #   right arm -> +sx
        if side == "left":
            outward_sign = -1.0
        else:
            outward_sign = 1.0

        upper_half = (
            self.invert_pitch_forward_above_shoulder
            and vertical > 0.0
        )

        if upper_half:
            # Upper hemisphere correction:
            # depth/forward must be mirrored for shoulder_pitch.
            #
            # Important:
            # do NOT mirror lateral for shoulder_roll here.
            # After standardizing X as left_shoulder -> right_shoulder,
            # outward_sign already gives correct outward component:
            #   left  outward = -X
            #   right outward = +X
            #
            # If we also invert lateral in the upper half, lateral_out becomes
            # negative and is clipped to zero, so roll stops working above shoulders.
            forward_for_pitch = -forward
        else:
            forward_for_pitch = forward

        # Pitch is the first shoulder joint.
        # It must be computed before roll compensation uses pitch_raw.
        pitch_raw = math.atan2(
            forward_for_pitch,
            max(0.10, abs(vertical)),
        )

        # Roll must be calculated AFTER removing pitch influence.
        #
        # Reason:
        #   shoulder_pitch is the first joint in the chain.
        #   If we calculate roll directly from raw 3D upper vector, then Z/depth
        #   contaminates roll. This is why bringing the elbow toward the camera
        #   incorrectly changed shoulder_roll.
        #
        # Correct order:
        #   1) compute pitch_raw in Y-Z plane
        #   2) rotate upper vector back around X by pitch_raw
        #   3) compute roll from the resulting X-Y vector
        #
        # With our convention:
        #   sx = body X, left_shoulder -> right_shoulder
        #   sy = body Y, bottom -> top
        #   sz = body Z, forward
        upper_no_pitch = rotate_about_axis(upper_u, sx, pitch_raw)
        upper_no_pitch_u = unit(upper_no_pitch)

        if upper_no_pitch_u is None:
            lateral_roll = 0.0
            vertical_roll = vertical
        else:
            lateral_roll = float(np.dot(upper_no_pitch_u, sx))
            vertical_roll = float(np.dot(upper_no_pitch_u, sy))

        lateral_out = outward_sign * lateral_roll
        lateral_out = max(0.0, lateral_out)

        # Roll:
        #   arm down -> 0
        #   T-pose   -> +/-pi/2
        #   arm up   -> +/-pi
        #
        # Now this is not raw X-Y projection anymore.
        # It is X-Y after pitch compensation.
        roll_mag = math.atan2(lateral_out, -vertical_roll)
        roll_mag = clamp(roll_mag, 0.0, 3.05)

        # Joint sign:
        #   left shoulder_roll  positive outward/up
        #   right shoulder_roll negative outward/up
        joint_roll_sign = 1.0 if side == "left" else -1.0
        roll = joint_roll_sign * roll_mag

        # Pitch:
        #   only torso depth / vertical plane.
        #   lateral is deliberately not used in pitch.

        if side == "left":
            pitch = self.BASE_Q[0] + self.pitch_gain * pitch_raw
            pitch = clamp(pitch, self.LOWER[0], self.UPPER[0])
            roll = clamp(roll, self.LOWER[1], self.UPPER[1])
        else:
            pitch = self.BASE_Q[4] + self.pitch_gain * pitch_raw
            pitch = clamp(pitch, self.LOWER[4], self.UPPER[4])
            roll = clamp(roll, self.LOWER[5], self.UPPER[5])

        if self.msg_count % 60 == 0:
            self.get_logger().info(
                f"{side} analytic elbow: "
                f"upper={int(upper_half)}, "
                f"lat={lateral:.3f}, fwd={forward:.3f}, lat_no_pitch={lateral_roll:.3f}, vert_no_pitch={vertical_roll:.3f}, lateral_out={lateral_out:.3f}, "
                f"vert={vertical:.3f}, "
                f"fwd={forward:.3f}, fwd_pitch={forward_for_pitch:.3f}, "
                f"pitch={pitch:.3f}, roll={roll:.3f}"
            )

        return pitch, roll

    def solve_shoulder_pitch_roll(self, side: str, upper_u: np.ndarray):
        side_sign = 1.0 if side == "left" else -1.0

        forward = float(upper_u[0])
        lateral_out = float(side_sign * upper_u[1])
        vertical = float(upper_u[2])

        lateral_out = max(0.0, lateral_out)

        # down -> 0, T-pose -> pi/2, arm up -> pi
        roll_mag = math.atan2(lateral_out, -vertical)
        roll_mag = clamp(roll_mag, 0.0, 3.05)

        roll = side_sign * roll_mag

        pitch_raw = math.atan2(
            forward,
            max(0.10, math.sqrt(lateral_out * lateral_out + vertical * vertical)),
        )

        # Important correction for elbow placement:
        # below shoulder level, depth/forward pitch works normally;
        # above shoulder level, the same forward elbow motion must rotate
        # shoulder_pitch in the opposite direction.
        #
        # vertical is dot(shoulder->elbow, shoulders_y):
        #   vertical < 0  => elbow below shoulders
        #   vertical >= 0 => elbow above shoulders
        if vertical >= 0.0:
            pitch_raw = -pitch_raw

        pitch_gain_eff = self.effective_pitch_gain(vertical)
        self.last_pitch_gain_by_side[side] = pitch_gain_eff

        if side == "left":
            pitch = self.BASE_Q[0] + pitch_gain_eff * pitch_raw
            pitch = clamp(pitch, self.LOWER[0], self.UPPER[0])
            roll = clamp(roll, self.LOWER[1], self.UPPER[1])
        else:
            pitch = self.BASE_Q[4] + pitch_gain_eff * pitch_raw
            pitch = clamp(pitch, self.LOWER[4], self.UPPER[4])
            roll = clamp(roll, self.LOWER[5], self.UPPER[5])

        return pitch, roll

    def yaw_model_dir(self, side: str, upper_u: np.ndarray, yaw_q: float, pts=None):
        """
        Model forearm bending direction controlled by shoulder_yaw.

        Important:
          In the upper hemisphere, H1 shoulder yaw has mirrored depth behavior.
          Therefore the forward axis used for yaw must also be inverted above
          the shoulder line, the same way we fixed pitch/roll.
        """
        if pts is not None:
            basis = self.compute_shoulder_basis(pts)
        else:
            basis = None

        if basis is not None:
            sx, sy, sz = basis
            body_up = sy
            body_forward = sz

            vertical = float(np.dot(upper_u, sy))
            upper_half = vertical > 0.0

            if self.invert_yaw_forward_above_shoulder and upper_half:
                body_forward = -body_forward
        else:
            body_up = np.array([0.0, 0.0, 1.0], dtype=float)
            body_forward = np.array([1.0, 0.0, 0.0], dtype=float)

        up_dir = project_perp(body_up, upper_u)
        forward_dir = project_perp(body_forward, upper_u)

        if up_dir is None and forward_dir is None:
            return None

        if up_dir is None:
            up_dir = forward_dir
        if forward_dir is None:
            forward_dir = up_dir

        down_dir = -up_dir

        if side == "left":
            yaw_down = self.left_yaw_down
            yaw_up = self.left_yaw_up
        else:
            yaw_down = self.right_yaw_down
            yaw_up = self.right_yaw_up

        denom = yaw_up - yaw_down
        if abs(denom) < 1e-6:
            t = 0.5
        else:
            t = clamp((yaw_q - yaw_down) / denom, 0.0, 1.0)

        # t=0   => elbow/forearm down
        # t=0.5 => elbow/forearm forward
        # t=1   => elbow/forearm up
        if t < 0.5:
            return lerp_dir(down_dir, forward_dir, t / 0.5)
        else:
            return lerp_dir(forward_dir, up_dir, (t - 0.5) / 0.5)

    def fk_upper_dir_from_pitch_roll(self, side: str, shoulder_pitch: float, shoulder_roll: float, pts):
        """
        Forward kinematics wrapper after shoulder_pitch/shoulder_roll.

        solve_arm() uses this to get the robot's actual upper-arm direction
        after the current shoulder_pitch and shoulder_roll values.

        FABRIK/yaw must start from this robot upper-arm direction, not from
        the raw human shoulder->elbow vector.
        """
        upper_u = self.model_upper_dir_from_pitch_roll(
            side,
            shoulder_pitch,
            shoulder_roll,
            pts,
        )

        if upper_u is None:
            return None, None

        robot_elbow = upper_u * max(1e-6, self.robot_upper_len)
        return upper_u, robot_elbow

    def correct_forearm_for_upper_yaw_branch(self, side: str, robot_upper_u: np.ndarray, fore_u: np.ndarray, pts):
        """
        Correct human elbow->wrist direction before FABRIK/yaw matching.

        We already found experimentally that H1 upper hemisphere is mirrored.
        For shoulder_yaw this means the target forearm depth component must be
        mirrored too, otherwise yaw becomes inverted above shoulders.

        Correction is applied only when robot upper arm is above the shoulder plane.
        """
        fu = unit(fore_u)
        if fu is None:
            return None

        if not self.invert_fabrik_target_forward_above_shoulder:
            return fu

        basis = self.compute_shoulder_basis(pts) if pts is not None else None
        if basis is None or robot_upper_u is None:
            return fu

        sx, sy, sz = basis

        vertical_upper = float(np.dot(robot_upper_u, sy))
        upper_half = vertical_upper > 0.0

        if not upper_half:
            return fu

        lateral = float(np.dot(fu, sx))
        vertical = float(np.dot(fu, sy))
        forward = float(np.dot(fu, sz))

        # Main fix: in upper hemisphere invert depth/forward target.
        forward = -forward

        corrected = lateral * sx + vertical * sy + forward * sz
        cu = unit(corrected)

        if cu is None:
            return fu

        return cu

    def scaled_robot_wrist_target(self, side: str, robot_elbow: np.ndarray, fore_u: np.ndarray):
        """
        Build wrist target in robot-scaled geometry.

        fore_u must already be corrected for the current H1 yaw branch.

        wrist_target = robot_elbow + normalize(corrected_fore_u) * robot_fore_len
        """
        fu = unit(fore_u)
        if fu is None:
            return None

        return robot_elbow + fu * max(1e-6, self.robot_fore_len)

    def model_forearm_dir_from_yaw(self, side: str, robot_upper_u: np.ndarray, yaw_q: float, pts=None):
        """
        Direction of the robot forearm caused by shoulder_yaw.

        This gives the bend-plane direction. elbow_pitch determines bend angle,
        but yaw determines the plane in which the forearm bends.
        """
        return self.yaw_model_dir(side, robot_upper_u, yaw_q, pts)

    def fabrik_forearm_target_dir(self, side: str, upper_u: np.ndarray, fore_u: np.ndarray):
        """
        FABRIK-style wrist target direction relative to the robot elbow.

        elbow_pitch is NOT solved here.

        The wrist target is built from the measured 3D vector elbow->wrist,
        but its length is scaled to robot_fore_len. This prevents different
        human/robot arm lengths from forcing artificial elbow bending.

        Returned vector is a unit direction in the same body/shoulder basis.
        """
        fu = unit(fore_u)
        if fu is None:
            return None

        wrist_target = fu * max(1e-6, self.robot_fore_len)
        target_dir = unit(wrist_target)

        return target_dir

    def solve_elbow_pitch_from_angle(self, side: str, upper_u: np.ndarray, fore_u: np.ndarray):
        """
        elbow_pitch is direct 3D bend angle, with MediaPipe straight-arm bias removed.

        raw_bend:
          angle between shoulder->elbow and elbow->wrist

        effective_bend:
          raw_bend - calibrated_straight_arm_bias - deadzone

        This prevents MediaPipe's small visible bend on a straight human arm from
        permanently bending the robot elbow.
        """
        raw_bend = angle_between(upper_u, fore_u)

        bias = self.elbow_bend_bias.get(side, 0.0) if self.use_calibrated_elbow_bias else 0.0
        effective_bend = max(0.0, raw_bend - bias - self.elbow_bend_deadzone)

        if side == "left":
            elbow_q = self.left_elbow_straight - self.elbow_gain * effective_bend
            elbow_q = clamp(elbow_q, self.LOWER[3], self.UPPER[3])
        else:
            elbow_q = self.right_elbow_straight - self.elbow_gain * effective_bend
            elbow_q = clamp(elbow_q, self.LOWER[7], self.UPPER[7])

        if self.msg_count % 60 == 0:
            self.get_logger().info(
                f"{side} elbow bend: raw={raw_bend:.3f}, "
                f"bias={bias:.3f}, eff={effective_bend:.3f}, q={elbow_q:.3f}"
            )

        return elbow_q, effective_bend

    def solve_shoulder_yaw_by_vectors(self, side: str, upper_u: np.ndarray, fore_u: np.ndarray, pts):
        """
        Pure vector-based shoulder_yaw.

        Definitions:
          upper = shoulder -> elbow
          fore  = elbow -> wrist

          sx = left_shoulder -> right_shoulder
          sy = bottom -> top
          sz = body forward normal

        yaw_ref:
          passes through elbow conceptually,
          points outward:
            left arm  -> left
            right arm -> right,
          and is perpendicular to upper.

        target:
          projection of elbow->wrist onto the plane perpendicular to upper.

        yaw angle:
          signed angle between yaw_ref and target around upper.
        """
        axis = unit(upper_u)
        target_raw = unit(fore_u)

        if axis is None or target_raw is None:
            return self.prev_left_yaw if side == "left" else self.prev_right_yaw

        basis = self.compute_shoulder_basis(pts)
        if basis is None:
            return self.prev_left_yaw if side == "left" else self.prev_right_yaw

        sx, sy, sz = basis

        # sx is left_shoulder -> right_shoulder.
        # Outward:
        #   left arm  = -sx
        #   right arm = +sx
        if side == "left":
            outward = -sx
            yaw_down = self.left_yaw_down
            yaw_up = self.left_yaw_up
            prev_yaw = self.prev_left_yaw
            sign = self.left_yaw_angle_sign
        else:
            outward = sx
            yaw_down = self.right_yaw_down
            yaw_up = self.right_yaw_up
            prev_yaw = self.prev_right_yaw
            sign = self.right_yaw_angle_sign

        yaw_ref = project_perp(outward, axis)

        # Singularity: if arm is almost exactly outward, outward is parallel to upper.
        # Then use torso-forward as fallback reference.
        if yaw_ref is None or float(np.linalg.norm(yaw_ref)) < 1e-6:
            yaw_ref = project_perp(sz, axis)

        # Second fallback.
        if yaw_ref is None:
            yaw_ref = project_perp(sy, axis)

        target = project_perp(target_raw, axis)

        if yaw_ref is None or target is None:
            return prev_yaw

        angle = signed_angle_about_axis(yaw_ref, target, axis)

        # Map angle to joint range around middle value.
        yaw_mid = 0.5 * (yaw_down + yaw_up)
        yaw_target = yaw_mid + sign * self.yaw_vector_gain * angle

        yaw_target = clamp(yaw_target, min(yaw_down, yaw_up), max(yaw_down, yaw_up))

        yaw = (
            (1.0 - self.yaw_vector_continuity) * yaw_target
            + self.yaw_vector_continuity * prev_yaw
        )

        if self.msg_count % 60 == 0:
            self.get_logger().info(
                f"{side} vector yaw: "
                f"angle={angle:.3f}, "
                f"yaw_target={yaw_target:.3f}, "
                f"yaw={yaw:.3f}"
            )

        return yaw

    def solve_shoulder_yaw_from_forearm(
        self,
        side: str,
        robot_upper_u: np.ndarray,
        fore_u: np.ndarray,
        pts=None,
        robot_elbow=None,
        bend_angle: float = 0.0,
    ):
        """
        Stage 3:
          scaled wrist/FABRIK matching by shoulder_yaw only.

        elbow_pitch is already fixed by bend_angle.
        This function MUST NOT change elbow_pitch.

        We build a robot-scaled wrist target:
          wrist_target = robot_elbow + normalize(human elbow->wrist) * robot_fore_len

        For each yaw candidate, we model robot wrist position:
          candidate_wrist =
              robot_elbow
              + robot_fore_len * (
                    cos(bend_angle) * robot_upper_u
                    + sin(bend_angle) * yaw_bend_dir
                )

        Then choose yaw with minimum wrist position error.
        """
        if robot_elbow is None:
            robot_elbow = robot_upper_u * max(1e-6, self.robot_upper_len)

        corrected_fore_u = self.correct_forearm_for_upper_yaw_branch(
            side,
            robot_upper_u,
            fore_u,
            pts,
        )

        wrist_target = self.scaled_robot_wrist_target(side, robot_elbow, corrected_fore_u)
        if wrist_target is None:
            if side == "left":
                return self.prev_left_yaw
            return self.prev_right_yaw

        target_vec = wrist_target - robot_elbow
        target_dir = unit(target_vec)

        if side == "left":
            yaw_down = self.left_yaw_down
            yaw_up = self.left_yaw_up
            prev_yaw = self.prev_left_yaw
        else:
            yaw_down = self.right_yaw_down
            yaw_up = self.right_yaw_up
            prev_yaw = self.prev_right_yaw

        if target_dir is None:
            return prev_yaw

        best_yaw = prev_yaw
        best_cost = 1e9
        best_pos_err = 1e9

        lo = min(yaw_down, yaw_up)
        hi = max(yaw_down, yaw_up)

        bend_angle = clamp(float(bend_angle), 0.0, math.pi)
        c = math.cos(bend_angle)
        s = math.sin(bend_angle)

        for yaw_q in np.linspace(lo, hi, max(9, self.yaw_grid_count)):
            bend_dir = self.model_forearm_dir_from_yaw(side, robot_upper_u, float(yaw_q), pts)
            if bend_dir is None:
                continue

            # Candidate forearm direction with fixed elbow angle.
            candidate_fore = c * robot_upper_u + s * bend_dir
            candidate_fore = unit(candidate_fore)
            if candidate_fore is None:
                continue

            candidate_wrist = robot_elbow + candidate_fore * max(1e-6, self.robot_fore_len)

            pos_err = float(np.linalg.norm(candidate_wrist - wrist_target))
            dir_err = 1.0 - clamp(float(np.dot(candidate_fore, target_dir)), -1.0, 1.0)

            continuity = self.yaw_continuity_weight * (float(yaw_q) - prev_yaw) ** 2

            cost = (
                self.fabrik_position_weight * pos_err
                + self.fabrik_wrist_weight * dir_err
                + continuity
            )

            if cost < best_cost:
                best_cost = cost
                best_pos_err = pos_err
                best_yaw = float(yaw_q)

        if self.msg_count % 60 == 0:
            self.get_logger().info(
                f"{side} FABRIK wrist scale: "
                f"yaw={best_yaw:.3f}, "
                f"bend={bend_angle:.3f}, "
                f"pos_err={best_pos_err:.3f}"
            )

        return best_yaw

    def solve_arm(self, pts, side: str):
        sh = pts[f"{side}_shoulder"]
        el = pts[f"{side}_elbow"]
        wr = pts[f"{side}_wrist"]

        upper = el - sh          # human shoulder -> elbow
        fore = wr - el           # human elbow -> wrist

        upper_ok, fore_ok = self.length_geometry_status(side, upper, fore)

        if not upper_ok:
            return None

        upper_u = unit(upper)
        fore_u = unit(fore)

        if upper_u is None:
            return None

        # ============================================================
        # 1) ELBOW POSITION PRIORITY
        # ============================================================
        # shoulder_pitch and shoulder_roll are solved ONLY from
        # shoulder -> elbow.
        #
        # This is the highest priority part of the algorithm.
        shoulder_pitch, shoulder_roll = self.solve_shoulder_pitch_roll_from_basis(
            side,
            upper_u,
            pts,
        )

        # Calculate actual robot upper-arm direction after applying
        # shoulder_pitch and shoulder_roll.
        #
        # FABRIK / yaw must use this robot FK direction, not raw human upper_u.
        robot_upper_u, robot_elbow = self.fk_upper_dir_from_pitch_roll(
            side,
            shoulder_pitch,
            shoulder_roll,
            pts,
        )

        if robot_upper_u is None:
            robot_upper_u = upper_u

        # ============================================================
        # 2) ELBOW PITCH PRIORITY
        # ============================================================
        # elbow_pitch is calculated ONLY as the 3D angle between:
        #
        #   human shoulder -> elbow
        #   human elbow    -> wrist
        #
        # FABRIK is not allowed to change elbow_pitch.
        if fore_u is None or not fore_ok:
            if side == "left":
                elbow_q = float(self.last_q[3])
            else:
                elbow_q = float(self.last_q[7])
            bend_angle = 0.0
        else:
            elbow_q, bend_angle = self.solve_elbow_pitch_from_angle(
                side,
                upper_u,
                fore_u,
            )

        # ============================================================
        # 3) WRIST / FABRIK STAGE
        # ============================================================
        # FABRIK-style wrist matching affects ONLY shoulder_yaw.
        #
        # It tries to rotate the forearm plane around robot_upper_u so that
        # the robot wrist direction best matches the human elbow->wrist
        # direction scaled to robot_fore_len.
        #
        # It must never modify elbow_pitch.
        if self.lock_shoulder_yaw:
            if side == "left":
                shoulder_yaw = self.locked_left_yaw
            else:
                shoulder_yaw = self.locked_right_yaw

        elif self.elbow_only_debug or fore_u is None or not fore_ok:
            if side == "left":
                shoulder_yaw = self.left_yaw_up
            else:
                shoulder_yaw = self.right_yaw_up
        else:
            if self.use_vector_yaw:
                shoulder_yaw = self.solve_shoulder_yaw_by_vectors(
                    side,
                    robot_upper_u,
                    fore_u,
                    pts,
                )
            else:
                shoulder_yaw = self.solve_shoulder_yaw_from_forearm(
                    side,
                    robot_upper_u,
                    fore_u,
                    pts,
                    robot_elbow=robot_elbow,
                    bend_angle=bend_angle,
                )

        if self.msg_count % 60 == 0:
            self.get_logger().info(
                f"{side}: "
                f"pitch={shoulder_pitch:.3f}, "
                f"roll={shoulder_roll:.3f}, "
                f"yaw={shoulder_yaw:.3f}, "
                f"elbow_q={elbow_q:.3f}, "
                f"bend_angle={bend_angle:.3f}"
            )

        return shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_q

    def compute_q(self, pts):
        lsol = self.solve_arm(pts, "left")
        rsol = self.solve_arm(pts, "right")

        if lsol is None or rsol is None:
            return None

        q = self.last_q.copy()

        q[0], q[1], q[2], q[3] = lsol
        q[4], q[5], q[6], q[7] = rsol

        q = np.clip(q, self.LOWER, self.UPPER)

        delta = q - self.last_q
        step_limits = np.array([
            self.max_joint_step,
            self.max_joint_step,
            self.yaw_max_step,
            self.elbow_max_step,
            self.max_joint_step,
            self.max_joint_step,
            self.yaw_max_step,
            self.elbow_max_step,
        ], dtype=float)

        q_limited = self.last_q + np.clip(delta, -step_limits, step_limits)
        q_filtered = self.q_filter.update(q_limited)
        q_filtered = np.clip(q_filtered, self.LOWER, self.UPPER)

        self.last_q = q_filtered.copy()

        self.prev_left_yaw = float(q_filtered[2])
        self.prev_right_yaw = float(q_filtered[6])

        return q_filtered

    def on_landmarks(self, msg: PoseLandmarks3D):
        self.msg_count += 1

        if not msg.valid:
            return

        raw, vis = self.get_points(msg)

        required = [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
        ]

        if not all(name in raw for name in required):
            return

        pts = self.make_body_points(raw)
        self.last_pts = pts

        if not self.rest_ready:
            if not self.calibration_requested:
                return

            if not self.visible_enough(vis):
                if self.msg_count % 30 == 0:
                    self.get_logger().warn("Waiting for visible RealSense landmarks...")
                return

            self.collect_length_sample(pts)
            self.calibration_count += 1

            if self.calibration_count < self.calibration_frames:
                if self.calibration_count % 15 == 0:
                    self.get_logger().info(
                        f"Calibrating arm scale... {self.calibration_count}/{self.calibration_frames}"
                    )
                return

            self.finish_length_calibration()

            self.calibrating = False
            self.rest_ready = True
            self.calibration_requested = False

            self.get_logger().info("Calibration complete. Teleoperation enabled.")

        q = self.compute_q(pts)

        if q is None:
            return

        self.publish_q(q, "realsense_two_stage_ik")

        if self.msg_count % 30 == 0:
            self.get_logger().info(
                "q=" + np.array2string(q, precision=3, suppress_small=True)
                + f" yaw L/R={q[2]:.3f}/{q[6]:.3f}"
            )


def main():
    rclpy.init()
    node = RealSenseIKUpperBodyRetarget()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
