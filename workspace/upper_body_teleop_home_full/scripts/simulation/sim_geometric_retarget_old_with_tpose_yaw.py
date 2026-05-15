#!/usr/bin/env python3
import math
from typing import Dict, Optional

import numpy as np
import rclpy
from rclpy.node import Node

from upper_body_msgs.msg import PoseLandmarks3D, UpperBodyCommand
from std_msgs.msg import String
from std_msgs.msg import Bool


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


class H1OldRetargetWithTPoseYaw(Node):
    # ВАЖНО: порядок и имена как у старого bridge.
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

    # Осторожная T-поза: плечи в стороны, локти слегка согнуты.
    TPOSE_Q = np.array([
        -0.10,  1.57,  0.017, 1.30,
        -0.10, -1.57, -0.050, 1.30,
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
        super().__init__("sim_geometric_retarget_tpose_yaw")

        self.declare_parameter("input_topic", "/pose/landmarks")
        self.declare_parameter("output_topic", "/upper_body/command_geom")
        self.declare_parameter("calibration_topic", "/upper_body/start_calibration")

        self.declare_parameter("tpose_hold_sec", 4.0)
        self.declare_parameter("calibration_frames", 45)

        self.declare_parameter("min_shoulder_visibility", 0.10)
        self.declare_parameter("min_elbow_visibility", 0.03)
        self.declare_parameter("min_wrist_visibility", 0.01)

        self.declare_parameter("landmark_alpha", 0.25)
        self.declare_parameter("joint_alpha", 0.45)

        # Меньше, чем раньше, чтобы не было резкого рывка в T-позу.
        self.declare_parameter("max_joint_step", 0.090)
        self.declare_parameter("elbow_max_step", 0.180)
        self.declare_parameter("yaw_max_step", 0.300)

        self.declare_parameter("map_x_from_z", 1.0)

        self.declare_parameter("pitch_gain", 0.45)
        self.declare_parameter("roll_gain", 1.00)
        self.declare_parameter("elbow_gain", 1.35)

        # Yaw вниз. Yaw вверх = старый BASE yaw.
        self.declare_parameter("left_yaw_up", 1.74)
        self.declare_parameter("left_yaw_down", -1.30)
        self.declare_parameter("right_yaw_up", -1.74)
        self.declare_parameter("right_yaw_down", 1.30)

        # Гистерезис перехода кисти относительно линии плечо->локоть.
        self.declare_parameter("yaw_hysteresis", 0.045)
        self.declare_parameter("forward_yaw_threshold", 1.50)
        self.declare_parameter("forward_yaw_blend_width", 0.20)

        self.declare_parameter("left_pitch_sign", -1.0)
        self.declare_parameter("right_pitch_sign", -1.0)

        self.declare_parameter("left_roll_sign", 1.0)
        self.declare_parameter("right_roll_sign", -1.0)

        # Из старой рабочей версии: локоть инвертирован.
        self.declare_parameter("left_elbow_sign", -1.0)
        self.declare_parameter("right_elbow_sign", -1.0)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.calibration_topic = self.get_parameter("calibration_topic").value

        self.tpose_hold_sec = float(self.get_parameter("tpose_hold_sec").value)
        self.calibration_frames = int(self.get_parameter("calibration_frames").value)

        self.min_sh_vis = float(self.get_parameter("min_shoulder_visibility").value)
        self.min_el_vis = float(self.get_parameter("min_elbow_visibility").value)
        self.min_wr_vis = float(self.get_parameter("min_wrist_visibility").value)

        self.landmark_alpha = float(self.get_parameter("landmark_alpha").value)
        self.joint_alpha = float(self.get_parameter("joint_alpha").value)
        self.max_joint_step = float(self.get_parameter("max_joint_step").value)
        self.elbow_max_step = float(self.get_parameter("elbow_max_step").value)
        self.yaw_max_step = float(self.get_parameter("yaw_max_step").value)

        self.map_x_from_z = float(self.get_parameter("map_x_from_z").value)

        self.pitch_gain = float(self.get_parameter("pitch_gain").value)
        self.roll_gain = float(self.get_parameter("roll_gain").value)
        self.elbow_gain = float(self.get_parameter("elbow_gain").value)

        self.left_yaw_up = float(self.get_parameter("left_yaw_up").value)
        self.left_yaw_down = float(self.get_parameter("left_yaw_down").value)
        self.right_yaw_up = float(self.get_parameter("right_yaw_up").value)
        self.right_yaw_down = float(self.get_parameter("right_yaw_down").value)
        self.yaw_hysteresis = float(self.get_parameter("yaw_hysteresis").value)
        self.forward_yaw_threshold = float(self.get_parameter("forward_yaw_threshold").value)
        self.forward_yaw_blend_width = float(self.get_parameter("forward_yaw_blend_width").value)

        self.left_pitch_sign = float(self.get_parameter("left_pitch_sign").value)
        self.right_pitch_sign = float(self.get_parameter("right_pitch_sign").value)
        self.left_roll_sign = float(self.get_parameter("left_roll_sign").value)
        self.right_roll_sign = float(self.get_parameter("right_roll_sign").value)
        self.left_elbow_sign = float(self.get_parameter("left_elbow_sign").value)
        self.right_elbow_sign = float(self.get_parameter("right_elbow_sign").value)

        self.pub = self.create_publisher(UpperBodyCommand, self.output_topic, 10)
        self.sub = self.create_subscription(PoseLandmarks3D, self.input_topic, self.on_landmarks, 10)
        self.calib_sub = self.create_subscription(Bool, self.calibration_topic, self.on_calibration_request, 10)

        self.filters: Dict[str, ExpFilter] = {}

        self.q_filter = ExpFilter(self.joint_alpha)
        self.q_filter.reset(self.BASE_Q)

        # Стартуем не с T-позы, а с BASE_Q, чтобы первое движение было плавным.
        self.last_q = self.BASE_Q.copy()

        self.rest_samples = []
        self.rest = None

        # Изначально после калибровки локоть смотрит вверх.
        self.left_yaw_branch = 0
        self.right_yaw_branch = 0

        # Manual calibration gate:
        # До явной команды калибровки держим T-позу бесконечно.
        self.calibration_requested = False
        self.calibrating = False

        self.msg_count = 0
        self.start_time = self.get_clock().now()

        # Таймер нужен, чтобы T-поза публиковалась сразу, даже до первой нормальной позы.
        self.timer = self.create_timer(1.0 / 30.0, self.on_timer)

        self.get_logger().info("===== H1 RETARGET OLD ALGO + TPOSE + LINE-YAW =====")
        self.get_logger().info(f"input_topic:     {self.input_topic}")
        self.get_logger().info(f"output_topic:    {self.output_topic}")
        self.get_logger().info(f"calib_topic:     {self.calibration_topic}")
        self.get_logger().info(f"tpose_hold_sec:  {self.tpose_hold_sec}")
        self.get_logger().info(f"BASE_Q:          {np.array2string(self.BASE_Q, precision=3)}")
        self.get_logger().info(f"TPOSE_Q:         {np.array2string(self.TPOSE_Q, precision=3)}")
        self.get_logger().info(f"yaw_up L/R:      {self.left_yaw_up} / {self.right_yaw_up}")
        self.get_logger().info(f"yaw_down L/R:    {self.left_yaw_down} / {self.right_yaw_down}")
        self.get_logger().info(f"yaw_hysteresis:  {self.yaw_hysteresis}")

    def elapsed(self) -> float:
        return (self.get_clock().now() - self.start_time).nanoseconds * 1e-9

    def on_calibration_request(self, msg: Bool):
        if not msg.data:
            return

        self.get_logger().info("Manual calibration requested. Holding T-pose while collecting neutral samples...")

        self.calibration_requested = True
        self.calibrating = True
        self.rest = None
        self.rest_samples = []

        # После калибровки начальное состояние yaw снова считаем elbow-up.
        self.left_yaw_branch = 0
        self.right_yaw_branch = 0

    def on_teleop_control(self, msg: String):
        command = msg.data.strip().lower()

        if command == "calibrate":
            self.get_logger().info("Camera requested calibration by key C. Holding T-pose while collecting neutral samples...")

            self.calibration_requested = True
            self.calibrating = True
            self.rest = None
            self.rest_samples = []

            # После калибровки начальное yaw-состояние считаем elbow-up.
            self.left_yaw_branch = 0
            self.right_yaw_branch = 0

        elif command == "reset":
            self.get_logger().info("Camera requested reset by key R. Back to T-pose and wait for calibration.")

            self.calibration_requested = False
            self.calibrating = False
            self.rest = None
            self.rest_samples = []
            self.left_yaw_branch = 0
            self.right_yaw_branch = 0

    def on_start_calibration(self, msg: Bool):
        if not msg.data:
            return

        self.get_logger().info("Manual calibration command received from terminal 5.")
        self.get_logger().info("Holding T-pose while collecting neutral samples...")

        self.calibration_requested = True
        self.calibrating = True
        self.rest = None
        self.rest_samples = []

        # После калибровки начальное yaw-состояние считаем elbow-up.
        self.left_yaw_branch = 0
        self.right_yaw_branch = 0

    def get_points(self, msg: PoseLandmarks3D):



        raw = {}
        vis = {}
        for i, name in enumerate(msg.names):
            if i < len(msg.x) and i < len(msg.y) and i < len(msg.z):
                raw[name] = np.array([msg.x[i], msg.y[i], msg.z[i]], dtype=float)
                vis[name] = float(msg.visibility[i]) if i < len(msg.visibility) else 1.0
        return raw, vis

    def mp_to_body(self, p_mp: np.ndarray, origin_mp: np.ndarray) -> np.ndarray:
        """
        Старое рабочее преобразование:
        x = forward
        y = left
        z = up
        """
        p = p_mp - origin_mp
        x_forward = -self.map_x_from_z * p[2]
        y_left = p[0]
        z_up = -p[1]
        return np.array([x_forward, y_left, z_up], dtype=float)

    def filtered_point(self, name: str, p: np.ndarray) -> np.ndarray:
        if name not in self.filters:
            self.filters[name] = ExpFilter(self.landmark_alpha)
        return self.filters[name].update(p)

    def landmarks_visible_enough(self, vis: Dict[str, float]) -> bool:
        checks = [
            ("left_shoulder", self.min_sh_vis),
            ("right_shoulder", self.min_sh_vis),
            ("left_elbow", self.min_el_vis),
            ("right_elbow", self.min_el_vis),
            ("left_wrist", self.min_wr_vis),
            ("right_wrist", self.min_wr_vis),
        ]
        return all(vis.get(name, 0.0) >= thr for name, thr in checks)

    def make_body_points(self, raw):
        shoulder_mid = 0.5 * (raw["left_shoulder"] + raw["right_shoulder"])
        origin = shoulder_mid

        pts = {}
        for name in [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
        ]:
            if name in raw:
                pts[name] = self.filtered_point(name, self.mp_to_body(raw[name], origin))
        return pts

    def wrist_line_residual_z(self, sh: np.ndarray, el: np.ndarray, wr: np.ndarray) -> float:
        """
        Новый yaw-критерий:
        строим линию shoulder->elbow, проектируем wrist на эту линию.
        Если wrist выше линии по z => elbow-up.
        Если ниже линии по z => elbow-down.
        """
        upper = el - sh
        sw = wr - sh
        denom = max(1e-9, float(np.dot(upper, upper)))
        t = clamp(float(np.dot(sw, upper) / denom), 0.0, 1.0)
        closest = sh + t * upper
        return float(wr[2] - closest[2])

    def update_yaw_branch(self, side: str, residual_z: float):
        if side == "left":
            old = self.left_yaw_branch
            if residual_z > self.yaw_hysteresis:
                self.left_yaw_branch = 0
            elif residual_z < -self.yaw_hysteresis:
                self.left_yaw_branch = 1
            if old != self.left_yaw_branch:
                self.get_logger().info(f"left yaw branch {old}->{self.left_yaw_branch}, residual_z={residual_z:.3f}")
        else:
            old = self.right_yaw_branch
            if residual_z > self.yaw_hysteresis:
                self.right_yaw_branch = 0
            elif residual_z < -self.yaw_hysteresis:
                self.right_yaw_branch = 1
            if old != self.right_yaw_branch:
                self.get_logger().info(f"right yaw branch {old}->{self.right_yaw_branch}, residual_z={residual_z:.3f}")

    def arm_features(self, pts: Dict[str, np.ndarray], side: str):
        sh = pts[f"{side}_shoulder"]
        el = pts[f"{side}_elbow"]
        wr = pts[f"{side}_wrist"]

        upper = el - sh
        fore = wr - el

        u = unit(upper)
        f = unit(fore)
        if u is None or f is None:
            return None

        horizontal = math.sqrt(u[0] * u[0] + u[1] * u[1])
        elevation = math.atan2(horizontal, max(1e-6, -u[2]))

        side_sign_for_measure = 1.0 if side == "left" else -1.0
        roll = math.atan2(side_sign_for_measure * u[1], max(1e-6, -u[2]))

        elbow = angle_between(upper, fore)

        residual_z = self.wrist_line_residual_z(sh, el, wr)

        # Depth / forward criterion:
        # body x-axis is forward. If wrist is farther forward from shoulder
        # than the upper-arm length, use middle yaw so elbow bends forward.
        upper_len = max(1e-6, float(np.linalg.norm(upper)))
        wrist_forward = float(wr[0] - sh[0])
        forward_ratio = wrist_forward / upper_len

        return {
            "elevation": elevation,
            "roll": roll,
            "elbow": elbow,
            "residual_z": residual_z,
            "upper": upper,
            "fore": fore,
            "forward_ratio": forward_ratio,
        }

    def calibrate_or_wait(self, pts, vis):
        if self.rest is not None:
            return True

        if not self.landmarks_visible_enough(vis):
            if self.msg_count % 30 == 0:
                self.get_logger().warn(
                    "Waiting for visible landmarks: "
                    f"LS={vis.get('left_shoulder',0):.2f}, "
                    f"LE={vis.get('left_elbow',0):.2f}, "
                    f"LW={vis.get('left_wrist',0):.2f}, "
                    f"RS={vis.get('right_shoulder',0):.2f}, "
                    f"RE={vis.get('right_elbow',0):.2f}, "
                    f"RW={vis.get('right_wrist',0):.2f}"
                )
            return False

        lf = self.arm_features(pts, "left")
        rf = self.arm_features(pts, "right")
        if lf is None or rf is None:
            return False

        self.rest_samples.append({"left": lf, "right": rf})

        if len(self.rest_samples) < self.calibration_frames:
            if self.msg_count % 15 == 0:
                self.get_logger().info(f"Calibrating neutral pose... {len(self.rest_samples)}/{self.calibration_frames}")
            return False

        self.rest = {
            "left": {
                "elevation": float(np.mean([s["left"]["elevation"] for s in self.rest_samples])),
                "roll": float(np.mean([s["left"]["roll"] for s in self.rest_samples])),
                "elbow": float(np.mean([s["left"]["elbow"] for s in self.rest_samples])),
            },
            "right": {
                "elevation": float(np.mean([s["right"]["elevation"] for s in self.rest_samples])),
                "roll": float(np.mean([s["right"]["roll"] for s in self.rest_samples])),
                "elbow": float(np.mean([s["right"]["elbow"] for s in self.rest_samples])),
            },
        }

        # После калибровки начальное состояние — elbow-up.
        self.left_yaw_branch = 0
        self.right_yaw_branch = 0

        self.calibrating = False
        self.calibration_requested = False

        self.get_logger().info("Calibration complete. Teleoperation enabled.")
        self.get_logger().info(f"rest left:  {self.rest['left']}")
        self.get_logger().info(f"rest right: {self.rest['right']}")
        return True

    def apply_forward_yaw_override(self, side: str, yaw: float, forward_ratio: float) -> float:
        """
        If the wrist is far forward, use middle yaw between up/down branches.
        This makes the elbow bend forward instead of only up/down.
        """
        if side == "left":
            yaw_mid = 0.5 * (self.left_yaw_up + self.left_yaw_down)
        else:
            yaw_mid = 0.5 * (self.right_yaw_up + self.right_yaw_down)

        if forward_ratio <= self.forward_yaw_threshold:
            return yaw

        width = max(1e-6, self.forward_yaw_blend_width)
        blend = clamp(
            (forward_ratio - self.forward_yaw_threshold) / width,
            0.0,
            1.0,
        )

        return (1.0 - blend) * yaw + blend * yaw_mid

    def compute_q(self, pts):
        lf = self.arm_features(pts, "left")
        rf = self.arm_features(pts, "right")
        if lf is None or rf is None or self.rest is None:
            return None

        q = self.BASE_Q.copy()

        def solve_side(side: str, f):
            """
            Absolute geometric shoulder IK.

            Body frame after mp_to_body:
              x = forward/back
              y = left
              z = up

            For left arm:
              outward lateral direction is +y
              shoulder_roll positive raises arm sideways/up.

            For right arm:
              outward lateral direction is -y
              shoulder_roll negative raises arm sideways/up.

            Down arm:
              upper.z ≈ -1 -> roll ≈ 0

            T-pose:
              upper lateral ≈ 1, upper.z ≈ 0 -> |roll| ≈ pi/2

            Arm up:
              upper.z ≈ +1 -> |roll| ≈ pi
            """
            upper = f["upper"]
            fore = f["fore"]

            u = unit(upper)
            if u is None:
                return None

            side_sign = 1.0 if side == "left" else -1.0

            forward = float(u[0])
            lateral = float(side_sign * u[1])
            vertical = float(u[2])

            # Убираем отрицательную lateral-компоненту, чтобы рука не уходила через корпус.
            lateral_pos = max(0.0, lateral)

            # Главный IK для подъёма:
            # down: atan2(0, 1) = 0
            # T:    atan2(1, 0) = pi/2
            # up:   atan2(0, -1) = pi
            roll_mag = math.atan2(lateral_pos, -vertical)
            roll_mag = clamp(roll_mag, 0.0, 3.02)

            shoulder_roll = side_sign * roll_mag

            # Pitch — только коррекция вперёд/назад, не основной подъём.
            pitch_raw = math.atan2(forward, max(0.15, math.sqrt(lateral_pos * lateral_pos + vertical * vertical)))
            shoulder_pitch = self.BASE_Q[0 if side == "left" else 4] + self.pitch_gain * pitch_raw
            shoulder_pitch = clamp(shoulder_pitch, -1.2, 1.2)

            # Elbow. При прямой руке bend около 0, при сгибании растёт.
            bend = angle_between(upper, fore)

            if side == "left":
                yaw = self.left_yaw_up if self.left_yaw_branch == 0 else self.left_yaw_down
                yaw = self.apply_forward_yaw_override(side, yaw, float(f.get("forward_ratio", 0.0)))
                elbow_q = self.BASE_Q[3] - self.elbow_gain * bend
            else:
                yaw = self.right_yaw_up if self.right_yaw_branch == 0 else self.right_yaw_down
                yaw = self.apply_forward_yaw_override(side, yaw, float(f.get("forward_ratio", 0.0)))
                elbow_q = self.BASE_Q[7] - self.elbow_gain * bend

            elbow_q = clamp(elbow_q, -1.10, 2.45)

            return shoulder_pitch, shoulder_roll, yaw, elbow_q

        self.update_yaw_branch("left", lf["residual_z"])
        self.update_yaw_branch("right", rf["residual_z"])

        lsol = solve_side("left", lf)
        rsol = solve_side("right", rf)
        if lsol is None or rsol is None:
            return None

        q[0], q[1], q[2], q[3] = lsol
        q[4], q[5], q[6], q[7] = rsol

        q = np.clip(q, self.LOWER, self.UPPER)

        # Ограничение скорости изменения команды.
        # Плечи двигаем плавно, локти быстрее, чтобы сгибание не запаздывало.
        delta = q - self.last_q
        step_limits = np.array([
            self.max_joint_step,   # left pitch
            self.max_joint_step,   # left roll
            self.yaw_max_step,     # left yaw: fast branch switching
            self.elbow_max_step,   # left elbow
            self.max_joint_step,   # right pitch
            self.max_joint_step,   # right roll
            self.yaw_max_step,     # right yaw: fast branch switching
            self.elbow_max_step,   # right elbow
        ], dtype=float)

        step = np.clip(delta, -step_limits, step_limits)
        q_limited = self.last_q + step

        q_filtered = self.q_filter.update(q_limited)
        q_filtered = np.clip(q_filtered, self.LOWER, self.UPPER)

        self.last_q = q_filtered.copy()

        return q_filtered

    def publish_q(self, stamp, q, valid=True, frame_id="h1_geometric_retarget"):
        msg = UpperBodyCommand()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.joint_names = list(self.JOINT_NAMES)
        msg.position = [float(v) for v in q]
        msg.confidence = [1.0 if valid else 0.0 for _ in q]
        msg.valid = bool(valid)
        self.pub.publish(msg)

    def smooth_towards_and_publish(self, target_q: np.ndarray, frame_id: str):
        step = np.clip(target_q - self.last_q, -self.max_joint_step, self.max_joint_step)
        q = self.last_q + step
        q = self.q_filter.update(q)
        q = np.clip(q, self.LOWER, self.UPPER)
        self.last_q = q.copy()
        self.publish_q(self.get_clock().now().to_msg(), q, True, frame_id)

    def on_timer(self):
        # До явной и успешной калибровки всегда держим T-позу.
        # Это убирает самопроизвольный переход к телеоперации.
        if self.rest is None:
            frame = "tpose_wait_manual_calibration"
            if self.calibrating:
                frame = "tpose_calibrating"
            self.smooth_towards_and_publish(self.TPOSE_Q, frame)
            return

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

        # До явной команды калибровки держим T-позу и не управляемся от человека.
        if self.rest is None and not self.calibration_requested:
            return

        if self.rest is None:
            if not self.calibrate_or_wait(pts, vis):
                return

        q = self.compute_q(pts)
        if q is None:
            return

        self.publish_q(msg.header.stamp, q, True)

        if self.msg_count % 30 == 0:
            self.get_logger().info(
                "q="
                + np.array2string(q, precision=3, suppress_small=True)
                + f" branches L/R={self.left_yaw_branch}/{self.right_yaw_branch}"
            )


def main():
    rclpy.init()
    node = H1OldRetargetWithTPoseYaw()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
