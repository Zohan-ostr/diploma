#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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



def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if abs(edge1 - edge0) < 1e-9:
        return 1.0 if x >= edge1 else 0.0
    t = clamp((float(x) - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


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
        super().__init__("vector_projection_retarget_sim")

        self.declare_parameter("input_topic", "/pose/landmarks")
        self.declare_parameter("output_topic", "/upper_body/command_geom")
        self.declare_parameter("calibration_topic", "/upper_body/start_calibration")

        self.declare_parameter("calibration_frames", 45)
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

        # Двухэтапный сегментный поиск pitch/roll.
        # 1 этап: вся допустимая область pitch/roll делится на 10x10 сегментов.
        # 2 этап: область вокруг лучшего результата +- половина грубого шага
        #         снова делится на 10x10 сегментов.
        # Итого: 200 проверок ПЗК на одну руку.
        self.declare_parameter("segment_pr_segments", 10)
        self.declare_parameter("segment_pr_continuity_weight", 0.002)

        # Параметры нового проекционного расчёта pitch/roll.
        self.declare_parameter("projection_pitch_min_norm", 0.12)
        self.declare_parameter("projection_pitch_full_norm", 0.35)
        self.declare_parameter("projection_pitch_gain", 1.0)
        self.declare_parameter("projection_roll_gain", 1.0)

        # Инверсия оси Z на входе нового проекционного алгоритма.
        # Важно: инвертируем все точки, чтобы forward/back одинаково
        # исправились для pitch, elbow и yaw.
        self.declare_parameter("projection_input_z_sign", -1.0)

        # Жёсткие зоны для положений "рука ровно вниз" и "рука ровно вверх".
        # Если upper-arm почти совпадает с осью Y, roll принудительно 0,
        # чтобы рука не уходила внутрь корпуса из-за шума X.
        self.declare_parameter("projection_vertical_snap_y", 0.92)
        self.declare_parameter("projection_roll_x_deadzone", 0.08)

        # Если оператор поднимает руку вверх или текущая модель руки
        # плохо совпадает с направлением shoulder->elbow, локального окна
        # вокруг предыдущего положения может не хватить. В этом случае
        # включается расширенный поиск по всему допустимому диапазону
        # pitch/roll, чтобы робот мог дотянуться до верхней позы.
        self.declare_parameter("enable_global_upper_search", True)
        self.declare_parameter("upper_global_search_dot_threshold", 0.35)
        self.declare_parameter("upper_arm_up_y_threshold", 0.20)

        self.declare_parameter("upper_direction_weight", 1.0)
        self.declare_parameter("upper_continuity_weight", 0.015)
        self.declare_parameter("yaw_direction_weight", 1.0)
        self.declare_parameter("yaw_continuity_weight", 0.05)

        # Двухэтапный сегментный поиск shoulder_yaw.
        # Грубо: 10 проверок по всей зоне yaw.
        # Уточнение: 10 проверок вокруг лучшего значения.
        # Итого: 20 проверок yaw на одну руку.
        self.declare_parameter("segment_yaw_segments", 10)
        self.declare_parameter("segment_yaw_continuity_weight", 0.01)

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

        # Совместимость со старым launcher-скриптом.
        # Эти параметры были в старом sim_geometric_retarget_old_with_tpose_yaw.py.
        # В данном варианте они не используются напрямую, но их объявление
        # позволяет безопасно оставлять старые команды запуска.
        self.declare_parameter("map_x_from_z", 1.0)
        self.declare_parameter("pitch_gain", 0.45)
        self.declare_parameter("roll_gain", 1.0)
        self.declare_parameter("yaw_hysteresis", 0.045)
        self.declare_parameter("forward_yaw_threshold", 1.50)
        self.declare_parameter("forward_yaw_blend_width", 0.20)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.calibration_topic = self.get_parameter("calibration_topic").value

        self.calibration_frames = int(self.get_parameter("calibration_frames").value)
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

        self.segment_pr_segments = int(self.get_parameter("segment_pr_segments").value)
        self.segment_pr_continuity_weight = float(self.get_parameter("segment_pr_continuity_weight").value)

        self.projection_pitch_min_norm = float(self.get_parameter("projection_pitch_min_norm").value)
        self.projection_pitch_full_norm = float(self.get_parameter("projection_pitch_full_norm").value)
        self.projection_pitch_gain = float(self.get_parameter("projection_pitch_gain").value)
        self.projection_roll_gain = float(self.get_parameter("projection_roll_gain").value)
        self.projection_input_z_sign = float(self.get_parameter("projection_input_z_sign").value)
        self.projection_vertical_snap_y = float(self.get_parameter("projection_vertical_snap_y").value)
        self.projection_roll_x_deadzone = float(self.get_parameter("projection_roll_x_deadzone").value)

        self.enable_global_upper_search = bool(self.get_parameter("enable_global_upper_search").value)
        self.upper_global_search_dot_threshold = float(
            self.get_parameter("upper_global_search_dot_threshold").value
        )
        self.upper_arm_up_y_threshold = float(
            self.get_parameter("upper_arm_up_y_threshold").value
        )

        self.upper_direction_weight = float(self.get_parameter("upper_direction_weight").value)
        self.upper_continuity_weight = float(self.get_parameter("upper_continuity_weight").value)
        self.yaw_direction_weight = float(self.get_parameter("yaw_direction_weight").value)
        self.yaw_continuity_weight = float(self.get_parameter("yaw_continuity_weight").value)

        self.segment_yaw_segments = int(self.get_parameter("segment_yaw_segments").value)
        self.segment_yaw_continuity_weight = float(self.get_parameter("segment_yaw_continuity_weight").value)

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
        self.get_logger().info("VECTOR PROJECTION RETARGET SIM")
        self.get_logger().info(f"input_topic:   {self.input_topic}")
        self.get_logger().info(f"output_topic:  {self.output_topic}")
        self.get_logger().info("stage 1: two-stage segmented FK search => shoulder_pitch + shoulder_roll")
        self.get_logger().info("stage 2: elbow_pitch = angle(shoulder->elbow, elbow->wrist)")
        self.get_logger().info("stage 3: two-stage segmented FK search => shoulder_yaw")
        self.get_logger().info("T-pose is held until calibration")
        self.get_logger().info("============================================================")

    def on_calibration(self, msg: Bool):
        if not msg.data:
            return

        # Защита от многократного сброса калибровки, потому что
        # h1_5_start_calibration.sh публикует Bool несколько раз подряд.
        if self.calibration_requested and self.calibration_count > 0:
            return

        self.calibration_requested = True
        self.rest_ready = False
        self.calibration_count = 0

        for side in ("left", "right"):
            self.calib_bend_angles[side] = []
            self.elbow_bend_bias[side] = 0.0

        self.get_logger().info("Manual calibration requested. Collecting straight-arm bias...")

    def filtered_point(self, name: str, p: np.ndarray) -> np.ndarray:
        if name not in self.filters:
            self.filters[name] = ExpFilter(self.landmark_alpha)
        return self.filters[name].update(p)

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
            # Исправление старой версии: project_perp требует ось.
            y_axis = project_perp(np.array([0.0, 1.0, 0.0], dtype=float), x_axis)
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

        # Новый алгоритм pitch/roll/yaw должен видеть одинаковую ось Z.
        # Поэтому forward/back исправляем здесь, до построения векторов руки.
        if self.projection_input_z_sign != 1.0:
            for name, p in list(pts_raw.items()):
                v = np.array(p, dtype=float).copy()
                v[2] = self.projection_input_z_sign * float(v[2])
                pts_raw[name] = v

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
        Двухэтапный сегментный поиск shoulder_pitch и shoulder_roll.

        Цель:
          подобрать pitch/roll так, чтобы локоть робота оказался как можно
          ближе к целевому положению локтя оператора.

        Этап 1:
          вся допустимая область pitch/roll делится на 10x10 сегментов;
          в центре каждого сегмента выполняется ПЗК и оценивается ошибка локтя.

        Этап 2:
          вокруг лучшего результата берётся область +- половина грубого шага;
          эта область снова делится на 10x10 сегментов;
          снова выполняется ПЗК и выбирается лучший результат.

        На одну руку:
          100 проверок на грубом этапе + 100 проверок на уточнении.
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

        nseg = max(2, int(self.segment_pr_segments))

        target_elbow = desired * max(1e-6, self.robot_upper_len)

        def centers(lo: float, hi: float, n: int):
            lo = float(lo)
            hi = float(hi)
            if hi < lo:
                lo, hi = hi, lo
            step = (hi - lo) / max(1, n)
            values = [lo + (i + 0.5) * step for i in range(n)]
            return values, step

        def eval_candidate(pitch_q: float, roll_q: float):
            model_dir = self.model_upper_dir(side, float(pitch_q), float(roll_q))
            if model_dir is None:
                return None

            # ПЗК для локтя:
            # shoulder считается началом координат,
            # elbow = direction(pitch, roll) * длина верхнего звена робота.
            model_elbow = model_dir * max(1e-6, self.robot_upper_len)

            dist_err = float(np.linalg.norm(model_elbow - target_elbow))

            # Небольшой штраф за резкий уход от прошлого положения.
            # Он не заменяет основной критерий расстояния, а только стабилизирует
            # выбор при близких по качеству решениях.
            continuity = self.segment_pr_continuity_weight * (
                (float(pitch_q) - prev_pitch) ** 2
                + (float(roll_q) - prev_roll) ** 2
            )

            cost = dist_err + continuity
            return cost, dist_err, model_dir

        def search_area(pitch_lo, pitch_hi, roll_lo, roll_hi):
            pitch_values, pitch_step = centers(pitch_lo, pitch_hi, nseg)
            roll_values, roll_step = centers(roll_lo, roll_hi, nseg)

            best = None

            for pitch_q in pitch_values:
                pitch_q = clamp(float(pitch_q), pitch_lo_abs, pitch_hi_abs)

                for roll_q in roll_values:
                    roll_q = clamp(float(roll_q), roll_lo_abs, roll_hi_abs)

                    r = eval_candidate(pitch_q, roll_q)
                    if r is None:
                        continue

                    cost, dist_err, model_dir = r

                    if best is None or cost < best["cost"]:
                        best = {
                            "pitch": pitch_q,
                            "roll": roll_q,
                            "dir": model_dir,
                            "cost": cost,
                            "dist": dist_err,
                            "pitch_step": pitch_step,
                            "roll_step": roll_step,
                        }

            return best

        # ------------------------------------------------------------
        # Этап 1. Грубый поиск по всей допустимой области.
        # ------------------------------------------------------------
        coarse = search_area(
            pitch_lo_abs,
            pitch_hi_abs,
            roll_lo_abs,
            roll_hi_abs,
        )

        if coarse is None:
            return prev_pitch, prev_roll, desired

        # ------------------------------------------------------------
        # Этап 2. Уточнение вокруг лучшего результата.
        # Берём +- половина шага грубого этапа.
        # ------------------------------------------------------------
        refine_pitch_lo = clamp(
            coarse["pitch"] - 0.5 * coarse["pitch_step"],
            pitch_lo_abs,
            pitch_hi_abs,
        )
        refine_pitch_hi = clamp(
            coarse["pitch"] + 0.5 * coarse["pitch_step"],
            pitch_lo_abs,
            pitch_hi_abs,
        )

        refine_roll_lo = clamp(
            coarse["roll"] - 0.5 * coarse["roll_step"],
            roll_lo_abs,
            roll_hi_abs,
        )
        refine_roll_hi = clamp(
            coarse["roll"] + 0.5 * coarse["roll_step"],
            roll_lo_abs,
            roll_hi_abs,
        )

        refined = search_area(
            refine_pitch_lo,
            refine_pitch_hi,
            refine_roll_lo,
            refine_roll_hi,
        )

        best = refined if refined is not None else coarse

        best_pitch = float(best["pitch"])
        best_roll = float(best["roll"])
        best_dir = best["dir"]

        if self.msg_count % 30 == 0:
            dot_v = clamp(float(np.dot(best_dir, desired)), -1.0, 1.0)
            self.get_logger().info(
                f"{side} two-stage pitch/roll search: "
                f"pitch={best_pitch:+.3f}, roll={best_roll:+.3f}, "
                f"elbow_dist={best['dist']:.4f}, dot={dot_v:.3f}, "
                f"coarse_dist={coarse['dist']:.4f}, "
                f"iters={nseg*nseg*2}"
            )

        return best_pitch, best_roll, best_dir

    def solve_elbow_pitch(self, side: str, upper_u: np.ndarray, fore_u: np.ndarray) -> Tuple[float, float]:
        raw_bend = angle_between(upper_u, fore_u)

        bias = self.elbow_bend_bias.get(side, 0.0) if self.use_calibrated_elbow_bias else 0.0

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
        desired_upper_u: np.ndarray,
        desired_fore_u: np.ndarray,
        bend_angle: float,
    ) -> float:
        """
        Двухэтапный поиск shoulder_yaw по реальной геометрии локтя и кисти.

        Важно:
          robot_upper_u — уже рассчитанное направление плеча робота после pitch/roll.
          desired_upper_u — направление shoulder->elbow оператора.
          desired_fore_u — направление elbow->wrist оператора.

        Цель yaw:
          не просто повторить направление предплечья,
          а из уже найденного локтя робота попасть кистью в целевую точку.

        Геометрия:
          S = (0, 0, 0)
          E_target = desired_upper_u * robot_upper_len
          W_target = E_target + desired_fore_u * robot_fore_len

          E_robot = robot_upper_u * robot_upper_len
          W_candidate = E_robot + fore_candidate(yaw) * robot_fore_len

        Поиск:
          10 грубых проверок по всей зоне yaw;
          10 уточняющих проверок вокруг лучшего yaw.
        """
        def vec3(x, fallback):
            try:
                a = np.asarray(x, dtype=float).reshape(-1)
                if a.size >= 3:
                    v = a[:3].astype(float)
                    if np.all(np.isfinite(v)):
                        return v
            except Exception:
                pass
            return np.array(fallback, dtype=float)

        robot_upper_u = unit(vec3(robot_upper_u, [0.0, -1.0, 0.0]))
        desired_upper_u = unit(vec3(desired_upper_u, [0.0, -1.0, 0.0]))
        desired_fore_u = unit(vec3(desired_fore_u, [0.0, -1.0, 0.0]))

        if robot_upper_u is None:
            robot_upper_u = np.array([0.0, -1.0, 0.0], dtype=float)
        if desired_upper_u is None:
            desired_upper_u = np.array([0.0, -1.0, 0.0], dtype=float)
        if desired_fore_u is None:
            desired_fore_u = np.array([0.0, -1.0, 0.0], dtype=float)

        if side == "left":
            yaw_idx = 2
            yaw_a = float(self.left_yaw_down)
            yaw_b = float(self.left_yaw_up)
            yaw_lo_abs = float(self.LOWER[2])
            yaw_hi_abs = float(self.UPPER[2])
        else:
            yaw_idx = 6
            yaw_a = float(self.right_yaw_down)
            yaw_b = float(self.right_yaw_up)
            yaw_lo_abs = float(self.LOWER[6])
            yaw_hi_abs = float(self.UPPER[6])

        yaw_lo = max(min(yaw_a, yaw_b), yaw_lo_abs)
        yaw_hi = min(max(yaw_a, yaw_b), yaw_hi_abs)

        if yaw_hi < yaw_lo:
            yaw_lo, yaw_hi = yaw_lo_abs, yaw_hi_abs

        prev_yaw = float(self.last_q[yaw_idx])

        try:
            nseg = max(2, int(self.segment_yaw_segments))
        except Exception:
            nseg = 10

        try:
            continuity_weight = float(self.segment_yaw_continuity_weight)
        except Exception:
            continuity_weight = 0.01

        # ------------------------------------------------------------
        # Главная правка:
        # target_wrist считается не от robot_upper_u, а от целевой
        # геометрии руки оператора.
        # ------------------------------------------------------------
        target_elbow = self.robot_upper_len * desired_upper_u
        target_wrist = target_elbow + self.robot_fore_len * desired_fore_u

        robot_elbow = self.robot_upper_len * robot_upper_u

        # Нормальная длина до цели, чтобы лог был понятнее.
        target_dist = float(np.linalg.norm(target_wrist))

        def centers(lo: float, hi: float, n: int):
            lo = float(lo)
            hi = float(hi)
            if hi < lo:
                lo, hi = hi, lo
            step = (hi - lo) / max(1, n)
            values = [lo + (i + 0.5) * step for i in range(n)]
            return values, step

        def make_fore_candidate(yaw_q: float):
            """
            Строим направление предплечья при данном yaw.

            robot_upper_u — ось верхнего звена.
            В плоскости, перпендикулярной robot_upper_u, yaw выбирает
            направление сгиба. bend_angle задаёт величину сгиба локтя.
            """
            axis = unit(robot_upper_u)
            if axis is None:
                axis = np.array([0.0, -1.0, 0.0], dtype=float)

            refs = [
                np.array([0.0, 1.0, 0.0], dtype=float),
                np.array([0.0, 0.0, 1.0], dtype=float),
                np.array([1.0, 0.0, 0.0], dtype=float),
            ]

            e1 = None
            for ref in refs:
                cand = project_perp(ref, axis)
                cand = unit(cand)
                if cand is not None:
                    e1 = cand
                    break

            if e1 is None:
                return axis

            e2 = unit(np.cross(axis, e1))
            if e2 is None:
                return axis

            if side == "right":
                e2 = -e2

            bend_dir = (
                math.cos(float(yaw_q)) * e1
                + math.sin(float(yaw_q)) * e2
            )
            bend_dir = unit(bend_dir)

            if bend_dir is None:
                return axis

            fore = (
                math.cos(float(bend_angle)) * axis
                + math.sin(float(bend_angle)) * bend_dir
            )
            fore = unit(fore)

            if fore is None:
                return axis

            return fore

        def eval_yaw(yaw_q: float):
            yaw_q = clamp(float(yaw_q), yaw_lo_abs, yaw_hi_abs)

            fore = make_fore_candidate(yaw_q)
            if fore is None:
                return None

            candidate_wrist = robot_elbow + self.robot_fore_len * fore

            wrist_err = float(np.linalg.norm(candidate_wrist - target_wrist))

            # Дополнительная оценка направления оставляем только как слабую
            # стабилизацию, основная цель — именно точка W_target.
            dot_fore = clamp(float(np.dot(fore, desired_fore_u)), -1.0, 1.0)
            dir_err = 1.0 - dot_fore

            continuity = continuity_weight * (yaw_q - prev_yaw) ** 2

            cost = (
                1.0 * wrist_err
                + 0.10 * dir_err
                + continuity
            )

            return {
                "yaw": yaw_q,
                "fore": fore,
                "candidate_wrist": candidate_wrist,
                "cost": cost,
                "wrist_err": wrist_err,
                "dir_err": dir_err,
                "dot": dot_fore,
            }

        def search_area(lo: float, hi: float):
            values, step = centers(lo, hi, nseg)
            best = None

            for yaw_q in values:
                r = eval_yaw(yaw_q)
                if r is None:
                    continue

                if best is None or r["cost"] < best["cost"]:
                    best = r
                    best["step"] = step

            return best

        # Этап 1. Грубый поиск.
        coarse = search_area(yaw_lo, yaw_hi)

        if coarse is None:
            return float(prev_yaw)

        # Этап 2. Уточнение вокруг лучшего.
        half_step = 0.5 * float(coarse["step"])

        refine_lo = clamp(coarse["yaw"] - half_step, yaw_lo, yaw_hi)
        refine_hi = clamp(coarse["yaw"] + half_step, yaw_lo, yaw_hi)

        refined = search_area(refine_lo, refine_hi)
        best = refined if refined is not None else coarse

        if self.msg_count % 30 == 0:
            elbow_err = float(np.linalg.norm(robot_elbow - target_elbow))
            self.get_logger().info(
                f"{side} two-stage yaw real-target: "
                f"yaw={best['yaw']:+.3f}, "
                f"wrist_err={best['wrist_err']:.4f}, "
                f"elbow_err={elbow_err:.4f}, "
                f"target_dist={target_dist:.4f}, "
                f"dot_fore={best['dot']:.3f}, "
                f"coarse_wrist_err={coarse['wrist_err']:.4f}, "
                f"iters={nseg * 2}"
            )

        return float(best["yaw"])

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
            f"R={self.elbow_bend_bias['right']:.3f}, "
            f"deadzone={self.elbow_bend_deadzone:.3f}, response={self.elbow_bend_response_gain:.2f}"
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
        yaw_q = self.search_yaw_for_wrist(side, robot_upper_u, upper_u, fore_u, bend_angle)

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
                if self.calibration_count % 15 == 0:
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
