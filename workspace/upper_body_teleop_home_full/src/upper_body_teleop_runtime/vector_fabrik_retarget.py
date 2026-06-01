#!/usr/bin/env python3
# ============================================================
# VECTOR FABRIK RETARGET
# ============================================================
#
# Назначение файла:
#   преобразовать 3D-точки оператора из /pose/landmarks
#   в команды верхней части тела робота:
#
#   left/right shoulder_pitch
#   left/right shoulder_roll
#   left/right shoulder_yaw
#   left/right elbow
#
# Основная логика:
#   1. Выделяем базис тела оператора X/Y/Z.
#   2. Формируем векторы руки:
#        shoulder -> elbow
#        elbow    -> wrist
#   3. Через FABRIK-подобный сегментный подбор находим pitch/roll,
#      чтобы поставить локоть.
#   4. Elbow считаем как угол между двумя сегментами руки.
#   5. Yaw подбираем сегментным поиском, чтобы позиционировать кисть.
#
# Что студентам можно менять в симуляции:
#   - размеры grid-поиска;
#   - веса ошибок;
#   - коэффициенты сглаживания;
#   - окна поиска pitch/roll/yaw;
#   - параметры deadzone/gain для elbow.
#
# Что нельзя менять без понимания:
#   - порядок суставов JOINT_NAMES;
#   - знаки левой/правой руки;
#   - структуру базиса X/Y/Z;
#   - формирование UpperBodyCommand.
#
# Для реального робота:
#   этот файл только рассчитывает целевые углы.
#   Настройки PD-регуляторов реального H1 находятся НЕ здесь.
# ============================================================

import math
import time
from typing import Dict, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool
from upper_body_msgs.msg import PoseLandmarks3D, UpperBodyCommand


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def unit(v: np.ndarray, eps: float = 1e-9) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n < eps:
        return None
    return v / n


def project_perp(v: np.ndarray, axis: np.ndarray) -> Optional[np.ndarray]:
    au = unit(axis)
    if au is None:
        return None
    p = v - float(np.dot(v, au)) * au
    return unit(p)


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    au = unit(a)
    bu = unit(b)
    if au is None or bu is None:
        return 0.0
    return math.acos(clamp(float(np.dot(au, bu)), -1.0, 1.0))


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


class VectorFabrikRetarget(Node):
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
        -0.10,  0.00,  0.00, 1.57,
        -0.10,  0.00,  0.00, 1.57,
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
        super().__init__("vector_fabrik_retarget")

        self.declare_parameter("input_topic", "/pose/landmarks")
        self.declare_parameter("output_topic", "/upper_body/command_geom")
        self.declare_parameter("calibration_topic", "/upper_body/start_calibration")

        self.declare_parameter("calibration_frames", 90)
        self.declare_parameter("calibration_duration_sec", 3.0)
        self.declare_parameter("min_visibility", 0.01)

        self.declare_parameter("landmark_alpha", 0.35)
        self.declare_parameter("joint_alpha", 0.45)

        self.declare_parameter("robot_upper_len", 0.31)
        self.declare_parameter("robot_fore_len", 0.31)

        self.declare_parameter("pitch_grid", 21)
        self.declare_parameter("roll_grid", 31)
        self.declare_parameter("yaw_grid", 81)

        self.declare_parameter("pitch_window", 1.30)
        self.declare_parameter("roll_window", 1.60)

        self.declare_parameter("upper_direction_weight", 1.0)
        self.declare_parameter("upper_continuity_weight", 0.015)
        self.declare_parameter("yaw_direction_weight", 1.0)
        self.declare_parameter("yaw_wrist_position_weight", 4.0)
        self.declare_parameter("yaw_refine_window", 0.18)
        self.declare_parameter("yaw_refine_grid", 17)
        self.declare_parameter("yaw_continuity_weight", 0.05)

        self.declare_parameter("pitch_geom_gain", 1.0)

        self.declare_parameter("left_yaw_down", -1.30)
        self.declare_parameter("left_yaw_up", 1.74)
        self.declare_parameter("right_yaw_down", 1.30)
        self.declare_parameter("right_yaw_up", -1.74)

        self.declare_parameter("left_elbow_straight", 1.57)
        self.declare_parameter("right_elbow_straight", 1.57)
        self.declare_parameter("elbow_gain", 1.35)
        self.declare_parameter("elbow_bend_deadzone", 0.015)
        self.declare_parameter("elbow_bend_response_gain", 1.75)
        self.declare_parameter("use_calibrated_elbow_bias", True)

        self.declare_parameter("max_joint_step", 0.090)
        self.declare_parameter("yaw_max_step", 0.300)
        self.declare_parameter("elbow_max_step", 0.180)

        self.declare_parameter("standard_z_sign", 1.0)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.calibration_topic = self.get_parameter("calibration_topic").value

        self.calibration_frames = int(self.get_parameter("calibration_frames").value)
        self.calibration_duration_sec = float(self.get_parameter("calibration_duration_sec").value)
        self.min_visibility = float(self.get_parameter("min_visibility").value)

        self.landmark_alpha = float(self.get_parameter("landmark_alpha").value)
        self.joint_alpha = float(self.get_parameter("joint_alpha").value)

        self.robot_upper_len = float(self.get_parameter("robot_upper_len").value)
        self.robot_fore_len = float(self.get_parameter("robot_fore_len").value)

        self.pitch_grid = int(self.get_parameter("pitch_grid").value)
        self.roll_grid = int(self.get_parameter("roll_grid").value)
        self.yaw_grid = int(self.get_parameter("yaw_grid").value)

        self.pitch_window = float(self.get_parameter("pitch_window").value)
        self.roll_window = float(self.get_parameter("roll_window").value)

        self.upper_direction_weight = float(self.get_parameter("upper_direction_weight").value)
        self.upper_continuity_weight = float(self.get_parameter("upper_continuity_weight").value)
        self.yaw_direction_weight = float(self.get_parameter("yaw_direction_weight").value)
        self.yaw_wrist_position_weight = float(self.get_parameter("yaw_wrist_position_weight").value)
        self.yaw_refine_window = float(self.get_parameter("yaw_refine_window").value)
        self.yaw_refine_grid = int(self.get_parameter("yaw_refine_grid").value)
        self.yaw_continuity_weight = float(self.get_parameter("yaw_continuity_weight").value)

        self.pitch_geom_gain = float(self.get_parameter("pitch_geom_gain").value)

        self.left_yaw_down = float(self.get_parameter("left_yaw_down").value)
        self.left_yaw_up = float(self.get_parameter("left_yaw_up").value)
        self.right_yaw_down = float(self.get_parameter("right_yaw_down").value)
        self.right_yaw_up = float(self.get_parameter("right_yaw_up").value)

        self.left_elbow_straight = float(self.get_parameter("left_elbow_straight").value)
        self.right_elbow_straight = float(self.get_parameter("right_elbow_straight").value)
        self.elbow_gain = float(self.get_parameter("elbow_gain").value)
        self.elbow_bend_deadzone = float(self.get_parameter("elbow_bend_deadzone").value)
        self.elbow_bend_response_gain = float(self.get_parameter("elbow_bend_response_gain").value)
        self.use_calibrated_elbow_bias = bool(self.get_parameter("use_calibrated_elbow_bias").value)

        self.max_joint_step = float(self.get_parameter("max_joint_step").value)
        self.yaw_max_step = float(self.get_parameter("yaw_max_step").value)
        self.elbow_max_step = float(self.get_parameter("elbow_max_step").value)

        self.standard_z_sign = float(self.get_parameter("standard_z_sign").value)

        self.filters: Dict[str, ExpFilter] = {}
        self.q_filter = ExpFilter(self.joint_alpha)
        self.q_filter.reset(self.TPOSE_Q)

        self.last_q = self.TPOSE_Q.copy()

        self.rest_ready = False
        self.calibration_requested = False
        self.calibration_count = 0
        self.calibration_start_time = None

        self.calib_bend_angles = {
            "left": [],
            "right": [],
        }
        self.elbow_bend_bias = {
            "left": 0.0,
            "right": 0.0,
        }

        self.msg_count = 0

        self.pub = self.create_publisher(UpperBodyCommand, self.output_topic, 10)
        self.sub = self.create_subscription(PoseLandmarks3D, self.input_topic, self.on_landmarks, 10)
        self.calib_sub = self.create_subscription(Bool, self.calibration_topic, self.on_calibration, 10)
        self.timer = self.create_timer(1.0 / 30.0, self.on_timer)

        self.get_logger().info("============================================================")
        self.get_logger().info("VECTOR FABRIK RETARGET")
        self.get_logger().info(f"input_topic:   {self.input_topic}")
        self.get_logger().info(f"output_topic:  {self.output_topic}")
        self.get_logger().info("stage 1: FABRIK elbow => shoulder_pitch + shoulder_roll")
        self.get_logger().info("stage 2: elbow_pitch = angle(shoulder->elbow, elbow->wrist)")
        self.get_logger().info("stage 3: segmented wrist search => shoulder_yaw only")
        self.get_logger().info("T-pose is held until calibration")
        self.get_logger().info("============================================================")

    def on_calibration(self, msg: Bool):
        if not msg.data:
            return

        self.calibration_requested = True
        self.rest_ready = False
        self.calibration_count = 0
        self.calibration_start_time = time.monotonic()

        for side in ("left", "right"):
            self.calib_bend_angles[side] = []
            self.elbow_bend_bias[side] = 0.0

        self.get_logger().info(f"Manual calibration requested. Hold straight T-pose for {self.calibration_duration_sec:.1f} sec. Collecting false elbow bend bias...")

    def filtered_point(self, name: str, p: np.ndarray) -> np.ndarray:
        if name not in self.filters:
            self.filters[name] = ExpFilter(self.landmark_alpha)
        return self.filters[name].update(p)

    def get_raw_points(self, msg: PoseLandmarks3D):
        raw = {}
        vis = {}

        for i, name in enumerate(msg.names):
            if i < len(msg.x) and i < len(msg.y) and i < len(msg.z):
                raw[name] = np.array([msg.x[i], msg.y[i], msg.z[i]], dtype=float)
                vis[name] = float(msg.visibility[i]) if i < len(msg.visibility) else 1.0

        return raw, vis

    def standardize_points_to_body_frame(self, raw):
        required = ["left_shoulder", "right_shoulder"]
        if not all(k in raw for k in required):
            return {}

        ls = raw["left_shoulder"]
        rs = raw["right_shoulder"]
        origin = 0.5 * (ls + rs)

        x_axis = unit(rs - ls)  # X: left shoulder -> right shoulder
        if x_axis is None:
            return {}

        if "left_hip" in raw and "right_hip" in raw:
            mid_hip = 0.5 * (raw["left_hip"] + raw["right_hip"])
            y0 = origin - mid_hip
        else:
            y0 = np.array([0.0, 1.0, 0.0], dtype=float)

        y_axis = project_perp(y0, x_axis)
        if y_axis is None:
            y_axis = project_perp(np.array([0.0, 1.0, 0.0], dtype=float))
        if y_axis is None:
            return {}

        z_axis = unit(np.cross(x_axis, y_axis))
        if z_axis is None:
            return {}

        z_axis = self.standard_z_sign * z_axis

        # Auto-forward: nose should be in front of body plane.
        if "nose" in raw:
            nose_z = float(np.dot(raw["nose"] - origin, z_axis))
            if nose_z < 0.0:
                z_axis = -z_axis

        pts = {}
        for name, p in raw.items():
            v = p - origin
            pts[name] = np.array([
                float(np.dot(v, x_axis)),
                float(np.dot(v, y_axis)),
                float(np.dot(v, z_axis)),
            ], dtype=float)

        return pts

    def make_points(self, raw):
        pts_raw = self.standardize_points_to_body_frame(raw)
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
        self.publish_smoothed(self.TPOSE_Q, "vector_fabrik_tpose")

    def publish_smoothed(self, target_q: np.ndarray, frame_id: str):
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

        delta = target_q - self.last_q
        q = self.last_q + np.clip(delta, -step_limits, step_limits)
        q = self.q_filter.update(q)
        q = np.clip(q, self.LOWER, self.UPPER)

        self.last_q = q.copy()
        self.publish_q(q, frame_id)

    def on_timer(self):
        if not self.rest_ready:
            self.smooth_publish_tpose()

    def model_upper_dir(self, side: str, pitch_q: float, roll_q: float) -> Optional[np.ndarray]:
        """
        Robot upper-arm FK model in standardized body frame.

        X = left -> right
        Y = bottom -> top
        Z = forward

        left outward  = -X
        right outward = +X
        """
        if side == "left":
            base_pitch = self.BASE_Q[0]
            outward_sign = -1.0
        else:
            base_pitch = self.BASE_Q[4]
            outward_sign = 1.0

        pitch_geom = (float(pitch_q) - base_pitch) / max(1e-6, self.pitch_geom_gain)
        roll_mag = abs(float(roll_q))

        lateral_out = math.sin(roll_mag) * math.cos(pitch_geom)
        vertical = -math.cos(roll_mag) * math.cos(pitch_geom)
        forward = math.sin(pitch_geom)

        d = np.array([
            outward_sign * lateral_out,
            vertical,
            forward,
        ], dtype=float)

        return unit(d)

    def search_pitch_roll_for_elbow(self, side: str, desired_upper_u: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        FABRIK-like elbow stage.

        We choose pitch+roll whose FK upper-arm direction best matches
        the measured shoulder->elbow direction.
        """
        desired = unit(desired_upper_u)
        if desired is None:
            desired = np.array([0.0, -1.0, 0.0], dtype=float)

        if side == "left":
            pitch_idx = 0
            roll_idx = 1
            pitch_lo_abs, pitch_hi_abs = self.LOWER[0], self.UPPER[0]
            roll_lo_abs, roll_hi_abs = 0.0, self.UPPER[1]
        else:
            pitch_idx = 4
            roll_idx = 5
            pitch_lo_abs, pitch_hi_abs = self.LOWER[4], self.UPPER[4]
            roll_lo_abs, roll_hi_abs = self.LOWER[5], 0.0

        prev_pitch = float(self.last_q[pitch_idx])
        prev_roll = float(self.last_q[roll_idx])

        pitch_lo = clamp(prev_pitch - self.pitch_window, pitch_lo_abs, pitch_hi_abs)
        pitch_hi = clamp(prev_pitch + self.pitch_window, pitch_lo_abs, pitch_hi_abs)

        roll_lo = clamp(prev_roll - self.roll_window, roll_lo_abs, roll_hi_abs)
        roll_hi = clamp(prev_roll + self.roll_window, roll_lo_abs, roll_hi_abs)

        pitch_values = np.linspace(pitch_lo, pitch_hi, max(5, self.pitch_grid))
        roll_values = np.linspace(roll_lo, roll_hi, max(5, self.roll_grid))

        best_pitch = prev_pitch
        best_roll = prev_roll
        best_dir = desired
        best_cost = 1e9
        best_dot = -1.0

        for pitch_q in pitch_values:
            for roll_q in roll_values:
                model = self.model_upper_dir(side, float(pitch_q), float(roll_q))
                if model is None:
                    continue

                dot_v = clamp(float(np.dot(model, desired)), -1.0, 1.0)
                direction_error = 1.0 - dot_v

                continuity = self.upper_continuity_weight * (
                    (float(pitch_q) - prev_pitch) ** 2
                    + (float(roll_q) - prev_roll) ** 2
                )

                cost = self.upper_direction_weight * direction_error + continuity

                if cost < best_cost:
                    best_cost = cost
                    best_dot = dot_v
                    best_pitch = float(pitch_q)
                    best_roll = float(roll_q)
                    best_dir = model

        if self.msg_count % 60 == 0:
            self.get_logger().info(
                f"{side} elbow FABRIK: pitch={best_pitch:.3f}, roll={best_roll:.3f}, dot={best_dot:.3f}"
            )

        return best_pitch, best_roll, best_dir

    def solve_elbow_pitch(self, side: str, upper_u: np.ndarray, fore_u: np.ndarray) -> Tuple[float, float]:
        raw_bend = angle_between(upper_u, fore_u)

        bias = self.elbow_bend_bias.get(side, 0.0) if self.use_calibrated_elbow_bias else 0.0
        # Keep straight-arm calibration, but do not make the elbow insensitive.
        #
        # raw_bend - bias:
        #   removes MediaPipe's visible bend on a straight human arm.
        #
        # elbow_bend_deadzone:
        #   only a very small noise zone.
        #
        # elbow_bend_response_gain:
        #   amplifies real bending after the calibrated straight-arm point,
        #   so small real bends are visible and full bends can be reached.
        effective_bend = max(0.0, raw_bend - bias - self.elbow_bend_deadzone)
        effective_bend *= self.elbow_bend_response_gain
        effective_bend = clamp(effective_bend, 0.0, math.pi)

        if side == "left":
            q = self.left_elbow_straight - self.elbow_gain * effective_bend
            q = clamp(q, self.LOWER[3], self.UPPER[3])
        else:
            q = self.right_elbow_straight - self.elbow_gain * effective_bend
            q = clamp(q, self.LOWER[7], self.UPPER[7])

        if self.msg_count % 60 == 0:
            self.get_logger().info(
                f"{side} elbow pitch: raw={raw_bend:.3f}, bias={bias:.3f}, eff={effective_bend:.3f}, q={q:.3f}"
            )

        return q, effective_bend

    def yaw_bend_dir(self, side: str, upper_u: np.ndarray, yaw_q: float) -> Optional[np.ndarray]:
        """
        Direction of forearm bend plane for a yaw candidate.
        """
        up = np.array([0.0, 1.0, 0.0], dtype=float)
        forward = np.array([0.0, 0.0, 1.0], dtype=float)

        up_dir = project_perp(up, upper_u)
        forward_dir = project_perp(forward, upper_u)

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

        if t < 0.5:
            d = (1.0 - t / 0.5) * down_dir + (t / 0.5) * forward_dir
        else:
            a = (t - 0.5) / 0.5
            d = (1.0 - a) * forward_dir + a * up_dir

        return unit(d)

    def search_yaw_for_wrist(
        self,
        side: str,
        robot_upper_u: np.ndarray,
        human_fore_u: np.ndarray,
        bend_angle: float,
    ) -> float:
        """
        Segmented FABRIK-style wrist stage.

        Fixed before this function:
          shoulder_pitch
          shoulder_roll
          elbow_pitch / bend_angle

        Variable here:
          shoulder_yaw only

        Goal:
          choose yaw that puts the robot wrist as close as possible to the
          scaled human wrist target.

        Segment logic:
          yaw_down  -> yaw_mid: forearm bends down -> forward
          yaw_mid   -> yaw_up:  forearm bends forward -> up

        This is not full IK over all joints. It is a one-joint segmented search
        for the wrist position.
        """
        target_fore_u = unit(human_fore_u)
        robot_upper_u = unit(robot_upper_u)

        if target_fore_u is None or robot_upper_u is None:
            return self.last_q[2] if side == "left" else self.last_q[6]

        if side == "left":
            prev_yaw = float(self.last_q[2])
            yaw_down = self.left_yaw_down
            yaw_up = self.left_yaw_up
        else:
            prev_yaw = float(self.last_q[6])
            yaw_down = self.right_yaw_down
            yaw_up = self.right_yaw_up

        yaw_lo = min(yaw_down, yaw_up)
        yaw_hi = max(yaw_down, yaw_up)

        robot_elbow = robot_upper_u * max(1e-6, self.robot_upper_len)
        target_wrist = robot_elbow + target_fore_u * max(1e-6, self.robot_fore_len)

        bend = clamp(float(bend_angle), 0.0, math.pi)
        c = math.cos(bend)
        s = math.sin(bend)

        def eval_yaw(yaw_q: float):
            bend_dir = self.yaw_bend_dir(side, robot_upper_u, float(yaw_q))
            if bend_dir is None:
                return None

            # elbow_pitch is fixed by bend_angle.
            # yaw only rotates the bend plane.
            candidate_fore = unit(c * robot_upper_u + s * bend_dir)
            if candidate_fore is None:
                return None

            candidate_wrist = robot_elbow + candidate_fore * max(1e-6, self.robot_fore_len)

            pos_err = float(np.linalg.norm(candidate_wrist - target_wrist))
            dir_err = 1.0 - clamp(float(np.dot(candidate_fore, target_fore_u)), -1.0, 1.0)

            continuity = self.yaw_continuity_weight * (float(yaw_q) - prev_yaw) ** 2

            cost = (
                self.yaw_wrist_position_weight * pos_err
                + self.yaw_direction_weight * dir_err
                + continuity
            )

            return cost, pos_err, dir_err, candidate_fore

        # First pass: full segmented yaw range.
        yaw_values = np.linspace(yaw_lo, yaw_hi, max(9, self.yaw_grid))

        best_yaw = prev_yaw
        best_cost = 1e9
        best_pos_err = 1e9
        best_dir_err = 1e9

        for yaw_q in yaw_values:
            r = eval_yaw(float(yaw_q))
            if r is None:
                continue

            cost, pos_err, dir_err, _ = r

            if cost < best_cost:
                best_cost = cost
                best_pos_err = pos_err
                best_dir_err = dir_err
                best_yaw = float(yaw_q)

        # Second pass: local refinement around best yaw.
        refine_lo = clamp(best_yaw - self.yaw_refine_window, yaw_lo, yaw_hi)
        refine_hi = clamp(best_yaw + self.yaw_refine_window, yaw_lo, yaw_hi)

        for yaw_q in np.linspace(refine_lo, refine_hi, max(5, self.yaw_refine_grid)):
            r = eval_yaw(float(yaw_q))
            if r is None:
                continue

            cost, pos_err, dir_err, _ = r

            if cost < best_cost:
                best_cost = cost
                best_pos_err = pos_err
                best_dir_err = dir_err
                best_yaw = float(yaw_q)

        if self.msg_count % 60 == 0:
            self.get_logger().info(
                f"{side} wrist segmented yaw: "
                f"yaw={best_yaw:.3f}, "
                f"pos_err={best_pos_err:.3f}, "
                f"dir_err={best_dir_err:.3f}, "
                f"bend={bend:.3f}"
            )

        return best_yaw

    def collect_calibration_sample(self, pts):
        for side in ("left", "right"):
            sh = pts[f"{side}_shoulder"]
            el = pts[f"{side}_elbow"]
            wr = pts[f"{side}_wrist"]

            upper_u = unit(el - sh)
            fore_u = unit(wr - el)

            if upper_u is not None and fore_u is not None:
                bend = angle_between(upper_u, fore_u)
                self.calib_bend_angles[side].append(float(bend))

    def finish_calibration(self):
        for side in ("left", "right"):
            vals = self.calib_bend_angles.get(side, [])
            if vals and self.use_calibrated_elbow_bias:
                # Ложный угол сгиба прямой руки.
                # Берём медиану за 3 секунды, чтобы не реагировать на единичные выбросы.
                self.elbow_bend_bias[side] = float(np.median(vals))
            else:
                self.elbow_bend_bias[side] = 0.0

        l_bias = self.elbow_bend_bias["left"]
        r_bias = self.elbow_bend_bias["right"]

        self.get_logger().info(
            "False elbow bend bias calibrated: "
            f"L={l_bias:.4f} rad ({math.degrees(l_bias):.2f} deg), "
            f"R={r_bias:.4f} rad ({math.degrees(r_bias):.2f} deg), "
            f"deadzone={self.elbow_bend_deadzone:.4f}, "
            f"response={self.elbow_bend_response_gain:.2f}"
        )
        self.get_logger().info("Calibration complete. Teleoperation enabled.")

    def solve_arm(self, pts, side: str):
        sh = pts[f"{side}_shoulder"]
        el = pts[f"{side}_elbow"]
        wr = pts[f"{side}_wrist"]

        upper = el - sh
        fore = wr - el

        upper_u = unit(upper)
        fore_u = unit(fore)

        if upper_u is None or fore_u is None:
            return None

        # 1) FABRIK for elbow: pitch + roll.
        pitch_q, roll_q, robot_upper_u = self.search_pitch_roll_for_elbow(side, upper_u)

        # 2) Elbow pitch rule: angle between human vectors.
        elbow_q, bend_angle = self.solve_elbow_pitch(side, upper_u, fore_u)

        # 3) FABRIK for wrist: yaw only.
        yaw_q = self.search_yaw_for_wrist(side, robot_upper_u, fore_u, bend_angle)

        return pitch_q, roll_q, yaw_q, elbow_q

    def compute_q(self, pts):
        lsol = self.solve_arm(pts, "left")
        rsol = self.solve_arm(pts, "right")

        if lsol is None or rsol is None:
            return None

        q = self.last_q.copy()
        q[0], q[1], q[2], q[3] = lsol
        q[4], q[5], q[6], q[7] = rsol

        return np.clip(q, self.LOWER, self.UPPER)

    def on_landmarks(self, msg: PoseLandmarks3D):
        self.msg_count += 1

        if not msg.valid:
            return

        raw, vis = self.get_raw_points(msg)

        required = [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
        ]

        if not all(name in raw for name in required):
            return

        if not self.visible_enough(vis):
            return

        pts = self.make_points(raw)

        if not all(name in pts for name in required):
            return

        if not self.rest_ready:
            if not self.calibration_requested:
                return

            self.collect_calibration_sample(pts)
            self.calibration_count += 1

            if self.calibration_start_time is None:
                self.calibration_start_time = time.monotonic()

            elapsed = time.monotonic() - self.calibration_start_time
            enough_time = elapsed >= self.calibration_duration_sec
            enough_frames = self.calibration_count >= self.calibration_frames

            # Основное условие — 3 секунды выдержки.
            # calibration_frames оставлен как дополнительная страховка для старых запусков.
            if not enough_time:
                if self.calibration_count % 15 == 0:
                    self.get_logger().info(
                        f"Calibrating false elbow bend... "
                        f"{elapsed:.1f}/{self.calibration_duration_sec:.1f} sec, "
                        f"samples={self.calibration_count}"
                    )
                return

            self.finish_calibration()
            self.rest_ready = True
            self.calibration_requested = False
            self.calibration_start_time = None

        q = self.compute_q(pts)
        if q is None:
            return

        self.publish_smoothed(q, "vector_fabrik")

    def destroy_node(self):
        super().destroy_node()


def main():
    rclpy.init()
    node = VectorFabrikRetarget()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
