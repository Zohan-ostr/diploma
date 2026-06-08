#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TEMPLATE POSE RETARGET 2 FOR UNITREE H1

Демонстрационный retarget для показательных фото/скриншотов.

Для каждой руки выбирается ближайшая из 6 заготовок:
  1) рука вверх;
  2) рука вниз;
  3) рука в сторону;
  4) рука вперёд;
  5) рука под 45 градусов вниз;
  6) рука под 45 градусов вверх.

Pitch/Roll берутся из таблицы заготовок.
Elbow считается по углу между shoulder->elbow и elbow->wrist.
Yaw считается широким сегментным поиском.
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool
from upper_body_msgs.msg import PoseLandmarks3D, UpperBodyCommand


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


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


class TemplatePoseRetarget2(Node):
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

    # left:  pitch, roll, yaw, elbow
    # right: pitch, roll, yaw, elbow
    TPOSE_Q = np.array([
        -0.10,  1.57,  1.74, 1.57,
        -0.10, -1.57, -1.74, 1.57,
    ], dtype=float)

    BASE_Q = np.array([
        -0.10, 0.00, 0.00, 1.57,
        -0.10, 0.00, 0.00, 1.57,
    ], dtype=float)

    # Старые широкие ограничения H1 из рабочего retarget.
    LOWER = np.array([
        -2.87, -0.34, -1.30, -1.25,
        -2.87, -3.11, -4.45, -1.25,
    ], dtype=float)

    UPPER = np.array([
         2.87,  3.11,  4.45,  2.61,
         2.87,  0.34,  1.30,  2.61,
    ], dtype=float)

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
        super().__init__("template_pose_retarget_2")

        self.declare_parameter("input_topic", "/pose/landmarks")
        self.declare_parameter("output_topic", "/upper_body/command_geom")
        self.declare_parameter("calibration_topic", "/upper_body/start_calibration")

        self.declare_parameter("calibration_frames", 30)
        self.declare_parameter("min_visibility", 0.01)

        self.declare_parameter("landmark_alpha", 0.45)
        self.declare_parameter("joint_alpha", 0.55)

        self.declare_parameter("left_yaw_down", -1.30)
        self.declare_parameter("left_yaw_up", 1.89)
        self.declare_parameter("right_yaw_down", 1.30)
        self.declare_parameter("right_yaw_up", -1.89)

        self.declare_parameter("yaw_grid", 121)
        self.declare_parameter("yaw_direction_weight", 1.0)
        self.declare_parameter("yaw_continuity_weight", 0.025)

        # Запрет ухода руки назад.
        # В локальном базисе Z направлена вперёд. Если кандидат yaw даёт
        # candidate_fore.z < 0, добавляем большой штраф.
        self.declare_parameter("no_back_yaw_penalty", 12.0)
        self.declare_parameter("no_back_z_margin", 0.02)

        self.declare_parameter("left_elbow_straight", 1.57)
        self.declare_parameter("right_elbow_straight", 1.57)
        self.declare_parameter("elbow_gain", 1.35)
        self.declare_parameter("elbow_bend_deadzone", 0.015)
        self.declare_parameter("elbow_bend_response_gain", 1.75)
        self.declare_parameter("use_calibrated_elbow_bias", True)

        self.declare_parameter("max_joint_step", 0.160)
        self.declare_parameter("yaw_max_step", 0.260)
        self.declare_parameter("elbow_max_step", 0.180)

        self.declare_parameter("standard_z_sign", 1.0)

        # Усиление глубины руки относительно плеча.
        # elbow.z = (elbow.z - shoulder.z) * depth_gain
        # wrist.z = (wrist.z - shoulder.z) * depth_gain
        self.declare_parameter("depth_gain", 1.5)

        # Масштаб калибровочного вычета для локтя.
        # 1.0 — полностью вычитать bias, 0.0 — не вычитать.
        # Для демонстрации лучше 0.35, чтобы локоть сгибался заметнее.
        self.declare_parameter("elbow_bias_scale", 0.0)

        # Гистерезис выбора заготовки: чтобы рука не щёлкала между двумя позами.
        self.declare_parameter("template_switch_margin", 0.08)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.calibration_topic = self.get_parameter("calibration_topic").value

        self.calibration_frames = int(self.get_parameter("calibration_frames").value)
        self.min_visibility = float(self.get_parameter("min_visibility").value)

        self.landmark_alpha = float(self.get_parameter("landmark_alpha").value)
        self.joint_alpha = float(self.get_parameter("joint_alpha").value)

        self.left_yaw_down = float(self.get_parameter("left_yaw_down").value)
        self.left_yaw_up = float(self.get_parameter("left_yaw_up").value)
        self.right_yaw_down = float(self.get_parameter("right_yaw_down").value)
        self.right_yaw_up = float(self.get_parameter("right_yaw_up").value)

        self.yaw_grid = int(self.get_parameter("yaw_grid").value)
        self.yaw_direction_weight = float(self.get_parameter("yaw_direction_weight").value)
        self.yaw_continuity_weight = float(self.get_parameter("yaw_continuity_weight").value)
        self.no_back_yaw_penalty = float(self.get_parameter("no_back_yaw_penalty").value)
        self.no_back_z_margin = float(self.get_parameter("no_back_z_margin").value)

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
        self.depth_gain = float(self.get_parameter("depth_gain").value)
        self.elbow_bias_scale = float(self.get_parameter("elbow_bias_scale").value)
        self.template_switch_margin = float(self.get_parameter("template_switch_margin").value)

        self.filters: Dict[str, ExpFilter] = {}

        self.q_filter = ExpFilter(self.joint_alpha)
        self.q_filter.reset(self.TPOSE_Q)
        self.last_q = self.TPOSE_Q.copy()

        self.rest_ready = False
        self.calibration_requested = False
        self.calibration_count = 0

        self.calib_bend_angles = {"left": [], "right": []}
        self.elbow_bend_bias = {"left": 0.0, "right": 0.0}

        self.last_template = {"left": None, "right": None}
        self.msg_count = 0
        self.last_input_frame_id = ""

        self.pub = self.create_publisher(UpperBodyCommand, self.output_topic, 10)
        self.sub = self.create_subscription(PoseLandmarks3D, self.input_topic, self.on_landmarks, 10)
        self.calib_sub = self.create_subscription(Bool, self.calibration_topic, self.on_calibration, 10)
        self.timer = self.create_timer(1.0 / 30.0, self.on_timer)

        self.get_logger().info("============================================================")
        self.get_logger().info("TEMPLATE POSE RETARGET 2")
        self.get_logger().info(f"input_topic:   {self.input_topic}")
        self.get_logger().info(f"output_topic:  {self.output_topic}")
        self.get_logger().info("pitch/roll: nearest of 6 predefined templates per arm")
        self.get_logger().info("elbow: angle(shoulder->elbow, elbow->wrist)")
        self.get_logger().info("yaw: full segmented search")
        self.get_logger().info("T-pose is held until calibration")
        self.get_logger().info("============================================================")

    # ------------------------------------------------------------
    # Templates
    # ------------------------------------------------------------

    def template_table(self, side: str):
        """
        Возвращает список:
          name, desired_direction, shoulder_pitch, shoulder_roll

        desired_direction используется только для выбора ближайшей позы
        по руке оператора.

        pitch/roll — готовая поза робота для демонстрации.
        """
        if side == "left":
            outward = np.array([-1.0, 0.0, 0.0], dtype=float)
            roll_side = 1.57
            roll_45 = 0.80
            roll_45_up = 0.45
            roll_down = 0.05
        else:
            outward = np.array([1.0, 0.0, 0.0], dtype=float)
            roll_side = -1.57
            roll_45 = -0.80
            roll_45_up = -0.45
            roll_down = -0.05

        up = np.array([0.0, 1.0, 0.0], dtype=float)
        down = np.array([0.0, -1.0, 0.0], dtype=float)
        forward = np.array([0.0, 0.0, 1.0], dtype=float)

        return [
            # Для реального H1 положительный большой pitch уводил руку назад,
            # поэтому верхние демонстрационные позы задаём отрицательным pitch.
            ("up", unit(up), 2.55, roll_down),
            ("down", unit(down), -0.10, roll_down),
            ("side", unit(outward), -0.10, roll_side),
            ("forward", unit(forward), 1.47, roll_down),
            ("down_45", unit(outward + down), -0.10, roll_45),
            ("up_45", unit(outward + up), 2.20, roll_45_up),
        ]

    def select_template(self, side: str, upper_u: np.ndarray) -> Tuple[str, float, float, float]:
        """
        Выбор ближайшей заготовки по максимальному dot(direction_template, upper_u).
        Возвращает:
          template_name, pitch, roll, best_dot.
        """
        u = unit(upper_u)
        if u is None:
            u = np.array([0.0, -1.0, 0.0], dtype=float)

        table = self.template_table(side)

        scores = []
        for name, d, pitch, roll in table:
            if d is None:
                continue
            dot_v = clamp(float(np.dot(u, d)), -1.0, 1.0)
            scores.append((dot_v, name, pitch, roll))

        scores.sort(reverse=True, key=lambda x: x[0])
        best_dot, best_name, best_pitch, best_roll = scores[0]

        # Гистерезис: если старая поза почти такая же близкая,
        # оставляем её, чтобы не было переключения туда-сюда.
        prev_name = self.last_template.get(side)
        if prev_name is not None:
            prev_score = None
            prev_pitch = None
            prev_roll = None
            for dot_v, name, pitch, roll in scores:
                if name == prev_name:
                    prev_score = dot_v
                    prev_pitch = pitch
                    prev_roll = roll
                    break

            if prev_score is not None and (best_dot - prev_score) < self.template_switch_margin:
                best_name = prev_name
                best_dot = prev_score
                best_pitch = prev_pitch
                best_roll = prev_roll

        self.last_template[side] = best_name

        if self.msg_count % 30 == 0:
            self.get_logger().info(
                f"{side} template={best_name}, dot={best_dot:.3f}, "
                f"pitch={best_pitch:.3f}, roll={best_roll:.3f}, "
                f"upper=({u[0]:.2f}, {u[1]:.2f}, {u[2]:.2f})"
            )

        return best_name, float(best_pitch), float(best_roll), float(best_dot)

    # ------------------------------------------------------------
    # Message parsing
    # ------------------------------------------------------------

    def on_calibration(self, msg: Bool):
        if not msg.data:
            return

        if self.calibration_requested and self.calibration_count > 0:
            return

        self.calibration_requested = True
        self.rest_ready = False
        self.calibration_count = 0

        self.calib_bend_angles = {"left": [], "right": []}
        self.elbow_bend_bias = {"left": 0.0, "right": 0.0}

        self.get_logger().info("Manual calibration requested. Collecting straight-arm bias...")

    def get_raw_points(self, msg: PoseLandmarks3D):
        raw = {}
        vis = {}

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
        visibility = list(getattr(msg, "visibility", []))

        if names and len(names) == len(xs):
            for i, name in enumerate(names):
                if i < len(xs) and i < len(ys) and i < len(zs):
                    p = np.array([xs[i], ys[i], zs[i]], dtype=float)
                    if np.all(np.isfinite(p)):
                        raw[name] = p
                        vis[name] = float(visibility[i]) if i < len(visibility) else 1.0
        else:
            for idx, name in self.MP_INDEX_TO_NAME.items():
                if idx < len(xs) and idx < len(ys) and idx < len(zs):
                    p = np.array([xs[idx], ys[idx], zs[idx]], dtype=float)
                    if np.all(np.isfinite(p)):
                        raw[name] = p
                        vis[name] = float(visibility[idx]) if idx < len(visibility) else 1.0

        return raw, vis


    def apply_camera_depth_z_to_body_points(self, raw, body_pts):
        """
        Z-координата руки берётся строго из камеры / карты глубины.

        X/Y остаются после пересчёта в локальный базис тела.
        Z НЕ берётся из body frame, потому что body frame может инвертироваться
        при ошибках ориентации тела.

        Для каждой руки:
          shoulder.z = 0
          elbow.z    = (raw_elbow.z - raw_shoulder.z) * depth_gain
          wrist.z    = (raw_wrist.z - raw_shoulder.z) * depth_gain
        """
        fixed = {k: np.array(v, dtype=float).copy() for k, v in body_pts.items()}

        for side in ("left", "right"):
            shoulder = f"{side}_shoulder"
            elbow = f"{side}_elbow"
            wrist = f"{side}_wrist"

            if shoulder not in raw or shoulder not in fixed:
                continue

            shoulder_depth = float(raw[shoulder][2])
            fixed[shoulder][2] = 0.0

            if elbow in raw and elbow in fixed:
                fixed[elbow][2] = (float(raw[elbow][2]) - shoulder_depth) * self.depth_gain

            if wrist in raw and wrist in fixed:
                fixed[wrist][2] = (float(raw[wrist][2]) - shoulder_depth) * self.depth_gain

        if self.msg_count % 30 == 0:
            try:
                self.get_logger().info(
                    "CAMERA depth z x%.2f: "
                    "L_elbow_z=%.3f L_wrist_z=%.3f R_elbow_z=%.3f R_wrist_z=%.3f"
                    % (
                        self.depth_gain,
                        fixed.get("left_elbow", [0, 0, 0])[2],
                        fixed.get("left_wrist", [0, 0, 0])[2],
                        fixed.get("right_elbow", [0, 0, 0])[2],
                        fixed.get("right_wrist", [0, 0, 0])[2],
                    )
                )
            except Exception:
                pass

        return fixed


    def make_depth_relative_to_side_shoulder(self, raw):
        fixed = {k: np.array(v, dtype=float).copy() for k, v in raw.items()}

        for side in ("left", "right"):
            shoulder = f"{side}_shoulder"
            elbow = f"{side}_elbow"
            wrist = f"{side}_wrist"

            if shoulder not in fixed:
                continue

            shoulder_z = float(fixed[shoulder][2])
            fixed[shoulder][2] = 0.0

            if elbow in fixed:
                fixed[elbow][2] = (float(fixed[elbow][2]) - shoulder_z) * self.depth_gain
            if wrist in fixed:
                fixed[wrist][2] = (float(fixed[wrist][2]) - shoulder_z) * self.depth_gain

        if self.msg_count % 30 == 0:
            try:
                self.get_logger().info(
                    "relative depth x%.2f: "
                    "L_elbow_z=%.3f L_wrist_z=%.3f R_elbow_z=%.3f R_wrist_z=%.3f"
                    % (
                        self.depth_gain,
                        fixed.get("left_elbow", [0, 0, 0])[2],
                        fixed.get("left_wrist", [0, 0, 0])[2],
                        fixed.get("right_elbow", [0, 0, 0])[2],
                        fixed.get("right_wrist", [0, 0, 0])[2],
                    )
                )
            except Exception:
                pass

        return fixed

    def visible_enough(self, vis):
        required = [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
        ]
        return all(vis.get(name, 0.0) >= self.min_visibility for name in required)

    def filtered_point(self, name: str, p: np.ndarray) -> np.ndarray:
        if name not in self.filters:
            self.filters[name] = ExpFilter(self.landmark_alpha)
        return self.filters[name].update(p)

    def standardize_points_to_body_frame(self, raw):
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
        # Новый camera node уже публикует:
        #   X/Y — body frame,
        #   Z   — camera depth относительно плеча.
        # Поэтому для frame_id body_pelvis_xy_camera_depth_z используем
        # координаты напрямую и не делаем повторный пересчёт.
        if self.last_input_frame_id == "body_pelvis_xy_camera_depth_z":
            pts_raw = {k: np.array(v, dtype=float).copy() for k, v in raw.items()}
        else:
            pts_raw = self.standardize_points_to_body_frame(raw)
            pts_raw = self.apply_camera_depth_z_to_body_points(raw, pts_raw)

        pts = {}
        for name, p in pts_raw.items():
            pts[name] = self.filtered_point(name, p)
        return pts

    # ------------------------------------------------------------
    # Elbow and yaw
    # ------------------------------------------------------------

    def solve_elbow_pitch(self, side: str, upper_u: np.ndarray, fore_u: np.ndarray) -> Tuple[float, float]:
        raw_bend = angle_between(upper_u, fore_u)

        bias = self.elbow_bend_bias.get(side, 0.0) if self.use_calibrated_elbow_bias else 0.0
        bias *= self.elbow_bias_scale

        effective_bend = max(0.0, raw_bend - bias - self.elbow_bend_deadzone)
        effective_bend *= self.elbow_bend_response_gain
        effective_bend = clamp(effective_bend, 0.0, math.pi)

        if side == "left":
            q = self.left_elbow_straight - self.elbow_gain * effective_bend
            q = clamp(q, self.LOWER[3], self.UPPER[3])
        else:
            q = self.right_elbow_straight - self.elbow_gain * effective_bend
            q = clamp(q, self.LOWER[7], self.UPPER[7])

        return q, effective_bend

    def model_upper_dir(self, side: str, pitch_q: float, roll_q: float) -> Optional[np.ndarray]:
        if side == "left":
            base_pitch = self.BASE_Q[0]
            outward_sign = -1.0
        else:
            base_pitch = self.BASE_Q[4]
            outward_sign = 1.0

        pitch_geom = float(pitch_q) - base_pitch
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

    def yaw_bend_dir(self, side: str, upper_u: np.ndarray, yaw_q: float) -> Optional[np.ndarray]:
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
        target = unit(human_fore_u)
        if target is None:
            return self.last_q[2] if side == "left" else self.last_q[6]

        if side == "left":
            prev_yaw = float(self.last_q[2])
            yaw_lo = min(self.left_yaw_down, self.left_yaw_up)
            yaw_hi = max(self.left_yaw_down, self.left_yaw_up)
        else:
            prev_yaw = float(self.last_q[6])
            yaw_lo = min(self.right_yaw_down, self.right_yaw_up)
            yaw_hi = max(self.right_yaw_down, self.right_yaw_up)

        yaw_values = np.linspace(yaw_lo, yaw_hi, max(9, self.yaw_grid))

        bend = clamp(float(bend_angle), 0.0, math.pi)
        c = math.cos(bend)
        s = math.sin(bend)

        best_yaw = prev_yaw
        best_cost = 1e9

        for yaw_q in yaw_values:
            bend_dir = self.yaw_bend_dir(side, robot_upper_u, float(yaw_q))
            if bend_dir is None:
                continue

            candidate_fore = unit(c * robot_upper_u + s * bend_dir)
            if candidate_fore is None:
                continue

            dot_v = clamp(float(np.dot(candidate_fore, target)), -1.0, 1.0)
            direction_error = 1.0 - dot_v
            continuity = self.yaw_continuity_weight * (float(yaw_q) - prev_yaw) ** 2

            # candidate_fore[2] < 0 означает, что предплечье уходит назад.
            # Для демонстрационного режима такие yaw-кандидаты запрещаем
            # через большой штраф, а не через жёсткий обрыв поиска.
            back_amount = max(0.0, -float(candidate_fore[2]) - self.no_back_z_margin)
            backward_penalty = self.no_back_yaw_penalty * back_amount

            cost = self.yaw_direction_weight * direction_error + continuity + backward_penalty

            if cost < best_cost:
                best_cost = cost
                best_yaw = float(yaw_q)

        return best_yaw

    # ------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------

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
                self.elbow_bend_bias[side] = float(np.median(vals))
            else:
                self.elbow_bend_bias[side] = 0.0

        self.get_logger().info(
            "Elbow bend bias calibration: "
            f"L={self.elbow_bend_bias['left']:.3f}, "
            f"R={self.elbow_bend_bias['right']:.3f}"
        )
        self.get_logger().info("Calibration complete. Template teleoperation enabled.")

    # ------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------

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

        template_name, pitch_q, roll_q, _ = self.select_template(side, upper_u)

        elbow_q, bend_angle = self.solve_elbow_pitch(side, upper_u, fore_u)

        robot_upper_u = self.model_upper_dir(side, pitch_q, roll_q)
        if robot_upper_u is None:
            robot_upper_u = upper_u

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

    # ------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------

    def publish_q(self, q: np.ndarray, frame_id: str):
        msg = UpperBodyCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.joint_names = list(self.JOINT_NAMES)
        msg.position = [float(v) for v in q]
        msg.confidence = [1.0 for _ in q]
        msg.valid = True
        self.pub.publish(msg)

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

    def smooth_publish_tpose(self):
        self.publish_smoothed(self.TPOSE_Q, "template_retarget_2_tpose")

    def on_timer(self):
        if not self.rest_ready:
            self.smooth_publish_tpose()

    def on_landmarks(self, msg: PoseLandmarks3D):
        self.msg_count += 1
        self.last_input_frame_id = str(getattr(msg.header, "frame_id", ""))

        if hasattr(msg, "valid") and not msg.valid:
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

            if self.calibration_count < self.calibration_frames:
                if self.calibration_count % 10 == 0:
                    self.get_logger().info(
                        f"Calibrating straight-arm bias... {self.calibration_count}/{self.calibration_frames}"
                    )
                return

            self.finish_calibration()
            self.rest_ready = True
            self.calibration_requested = False

        q = self.compute_q(pts)
        if q is None:
            return

        self.publish_smoothed(q, "template_retarget_2")


def main():
    rclpy.init()
    node = TemplatePoseRetarget2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
