#!/usr/bin/env python3

import math
import threading
from typing import Dict, List, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from unitree_go.msg import LowCmd, LowState
from upper_body_msgs.msg import UpperBodyCommand


H1_NUM_MOTORS = 20

# Полный порядок H1 из официального xr_teleoperate.
H1_JOINT_INDEX = {
    "right_hip_roll": 0,
    "right_hip_pitch": 1,
    "right_knee": 2,
    "left_hip_roll": 3,
    "left_hip_pitch": 4,
    "left_knee": 5,
    "waist_yaw": 6,
    "left_hip_yaw": 7,
    "right_hip_yaw": 8,
    "not_used": 9,
    "left_ankle": 10,
    "right_ankle": 11,

    "right_shoulder_pitch_joint": 12,
    "right_shoulder_roll_joint": 13,
    "right_shoulder_yaw_joint": 14,
    "right_elbow_joint": 15,

    "left_shoulder_pitch_joint": 16,
    "left_shoulder_roll_joint": 17,
    "left_shoulder_yaw_joint": 18,
    "left_elbow_joint": 19,
}

# ВАЖНО: порядок как у Unitree для H1: left first, then right.
# Хотя в DDS motor_cmd сами индексы правой руки идут раньше.
H1_ARM_ORDER = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

WEAK_MOTOR_INDICES = {
    H1_JOINT_INDEX["left_ankle"],
    H1_JOINT_INDEX["right_ankle"],

    H1_JOINT_INDEX["left_shoulder_pitch_joint"],
    H1_JOINT_INDEX["left_shoulder_roll_joint"],
    H1_JOINT_INDEX["left_shoulder_yaw_joint"],
    H1_JOINT_INDEX["left_elbow_joint"],

    H1_JOINT_INDEX["right_shoulder_pitch_joint"],
    H1_JOINT_INDEX["right_shoulder_roll_joint"],
    H1_JOINT_INDEX["right_shoulder_yaw_joint"],
    H1_JOINT_INDEX["right_elbow_joint"],
}


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class H1UnitreeStyleArmController(Node):
    """
    Unitree-style arm controller for H1.

    Основная идея взята из xr_teleoperate:
      - читаем /lowstate;
      - запоминаем текущую позу всех моторов;
      - формируем LowCmd на всё тело;
      - тело удерживаем в текущей позе;
      - руки обновляем по q_target;
      - публикуем LowCmd постоянно, отдельным timer на 250 Гц;
      - ограничиваем скорость рук.
    """

    def __init__(self):
        super().__init__("h1_unitree_style_arm_controller")

        self.input_topic = str(self.declare_parameter("input_topic", "/upper_body/command").value)
        self.lowstate_topic = str(self.declare_parameter("lowstate_topic", "/lowstate").value)
        self.output_topic = str(self.declare_parameter("output_topic", "/arm_sdk").value)

        self.control_hz = float(self.declare_parameter("control_hz", 250.0).value)

        # Осторожнее, чем у Unitree. Для тестов не надо 20 rad/s.
        self.arm_velocity_limit = float(self.declare_parameter("arm_velocity_limit", 1.2).value)

        self.kp_high = float(self.declare_parameter("kp_high", 300.0).value)
        self.kd_high = float(self.declare_parameter("kd_high", 5.0).value)
        self.kp_low = float(self.declare_parameter("kp_low", 140.0).value)
        self.kd_low = float(self.declare_parameter("kd_low", 3.0).value)

        # test_mode:
        #   none
        #   left_pitch
        #   right_pitch
        #   left_roll
        #   right_roll
        #   left_yaw
        #   right_yaw
        #   left_elbow
        #   right_elbow
        self.test_mode = str(self.declare_parameter("test_mode", "none").value)
        self.test_amplitude = float(self.declare_parameter("test_amplitude", 0.25).value)
        self.test_frequency = float(self.declare_parameter("test_frequency", 0.20).value)

        self.print_every = int(self.declare_parameter("print_every", 250).value)

        qos_best_effort = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.lowstate_sub = self.create_subscription(
            LowState,
            self.lowstate_topic,
            self.on_lowstate,
            qos_best_effort,
        )

        self.cmd_sub = self.create_subscription(
            UpperBodyCommand,
            self.input_topic,
            self.on_upper_body_command,
            10,
        )

        self.pub = self.create_publisher(
            LowCmd,
            self.output_topic,
            qos_best_effort,
        )

        self.lock = threading.Lock()

        self.lowstate_received = False
        self.all_motor_q: Optional[np.ndarray] = None
        self.all_motor_dq: Optional[np.ndarray] = None

        self.base_all_q: Optional[np.ndarray] = None

        self.q_target = np.zeros(8, dtype=float)
        self.tau_target = np.zeros(8, dtype=float)

        self.last_cmd_time = None
        self.tick = 0
        self.start_time = self.get_clock().now().nanoseconds * 1e-9

        period = 1.0 / self.control_hz
        self.timer = self.create_timer(period, self.control_loop)

        self.get_logger().info("H1 Unitree-style arm controller started")
        self.get_logger().info(f"input_topic:    {self.input_topic}")
        self.get_logger().info(f"lowstate_topic: {self.lowstate_topic}")
        self.get_logger().info(f"output_topic:   {self.output_topic}")
        self.get_logger().info(f"test_mode:      {self.test_mode}")
        self.get_logger().info("Waiting for /lowstate...")

    def on_lowstate(self, msg: LowState):
        if len(msg.motor_state) < H1_NUM_MOTORS:
            self.get_logger().warn_once(
                f"LowState has only {len(msg.motor_state)} motors, expected {H1_NUM_MOTORS}"
            )
            return

        q = np.zeros(H1_NUM_MOTORS, dtype=float)
        dq = np.zeros(H1_NUM_MOTORS, dtype=float)

        for i in range(H1_NUM_MOTORS):
            q[i] = float(msg.motor_state[i].q)
            dq[i] = float(msg.motor_state[i].dq)

        with self.lock:
            self.all_motor_q = q
            self.all_motor_dq = dq

            if not self.lowstate_received:
                self.lowstate_received = True
                self.base_all_q = q.copy()

                # стартовая цель рук = текущая поза рук
                self.q_target = self.get_arm_q_from_all(q)

        if self.lowstate_received and self.base_all_q is not None and self.tick < 5:
            self.get_logger().info("Captured initial H1 state from /lowstate")
            self.get_logger().info(f"initial arm q left-first: {self.q_target}")

    def on_upper_body_command(self, msg: UpperBodyCommand):
        if not msg.valid:
            return

        name_to_pos: Dict[str, float] = {}
        for name, pos in zip(msg.joint_names, msg.position):
            name_to_pos[name] = float(pos)

        q = np.zeros(8, dtype=float)

        missing = []
        for i, joint in enumerate(H1_ARM_ORDER):
            if joint not in name_to_pos:
                missing.append(joint)
            else:
                q[i] = name_to_pos[joint]

        if missing:
            self.get_logger().warn_throttle(
                2.0,
                f"UpperBodyCommand missing joints: {missing}",
            )
            return

        with self.lock:
            self.q_target = q
            self.tau_target = np.zeros(8, dtype=float)
            self.last_cmd_time = self.get_clock().now().nanoseconds * 1e-9

    def get_arm_q_from_all(self, all_q: np.ndarray) -> np.ndarray:
        return np.array(
            [all_q[H1_JOINT_INDEX[joint]] for joint in H1_ARM_ORDER],
            dtype=float,
        )

    def clip_arm_q_target(self, target_q: np.ndarray, current_q: np.ndarray) -> np.ndarray:
        delta = target_q - current_q
        max_delta_allowed = self.arm_velocity_limit / self.control_hz

        max_abs_delta = float(np.max(np.abs(delta)))
        if max_abs_delta <= max_delta_allowed:
            return target_q

        scale = max_abs_delta / max_delta_allowed
        return current_q + delta / max(scale, 1.0)

    def apply_test_mode(self, q_target: np.ndarray) -> np.ndarray:
        if self.test_mode == "none":
            return q_target

        if self.base_all_q is None:
            return q_target

        q = self.get_arm_q_from_all(self.base_all_q)
        now = self.get_clock().now().nanoseconds * 1e-9
        t = now - self.start_time
        s = self.test_amplitude * math.sin(2.0 * math.pi * self.test_frequency * t)

        index_by_test = {
            "left_pitch": 0,
            "left_roll": 1,
            "left_yaw": 2,
            "left_elbow": 3,
            "right_pitch": 4,
            "right_roll": 5,
            "right_yaw": 6,
            "right_elbow": 7,
        }

        if self.test_mode in index_by_test:
            q[index_by_test[self.test_mode]] += s

        return q

    def make_lowcmd(self, clipped_arm_q: np.ndarray, tau_target: np.ndarray) -> LowCmd:
        msg = LowCmd()

        # Поля есть в unitree_go/msg/LowCmd, но на всякий случай проверяем hasattr.
        if hasattr(msg, "head") and len(msg.head) >= 2:
            msg.head[0] = 0xFE
            msg.head[1] = 0xEF

        if hasattr(msg, "level_flag"):
            msg.level_flag = 0xFF

        if hasattr(msg, "gpio"):
            msg.gpio = 0

        # Держим всё тело в той позе, которую поймали при старте контроллера.
        # Руки потом перезаписываются target-значениями.
        base = self.base_all_q
        if base is None:
            base = np.zeros(H1_NUM_MOTORS, dtype=float)

        for idx in range(min(H1_NUM_MOTORS, len(msg.motor_cmd))):
            mc = msg.motor_cmd[idx]

            if idx in WEAK_MOTOR_INDICES:
                mc.kp = self.kp_low
                mc.kd = self.kd_low
                mc.mode = 0x01
            else:
                mc.kp = self.kp_high
                mc.kd = self.kd_high
                mc.mode = 0x0A

            mc.q = float(base[idx])
            mc.dq = 0.0
            mc.tau = 0.0

        # Обновляем только руки.
        for i, joint in enumerate(H1_ARM_ORDER):
            motor_idx = H1_JOINT_INDEX[joint]
            mc = msg.motor_cmd[motor_idx]
            mc.q = float(clipped_arm_q[i])
            mc.dq = 0.0
            mc.tau = float(tau_target[i])
            mc.kp = self.kp_low
            mc.kd = self.kd_low
            mc.mode = 0x01

        return msg

    def control_loop(self):
        with self.lock:
            lowstate_ok = self.lowstate_received
            all_q = None if self.all_motor_q is None else self.all_motor_q.copy()
            q_target = self.q_target.copy()
            tau_target = self.tau_target.copy()

        if not lowstate_ok or all_q is None:
            return

        current_arm_q = self.get_arm_q_from_all(all_q)

        q_target = self.apply_test_mode(q_target)
        clipped = self.clip_arm_q_target(q_target, current_arm_q)

        msg = self.make_lowcmd(clipped, tau_target)
        self.pub.publish(msg)

        self.tick += 1

        if self.print_every > 0 and self.tick % self.print_every == 0:
            print()
            print("===== H1 UNITREE STYLE ARM CTRL =====")
            print(f"output: {self.output_topic}")
            print(f"test_mode: {self.test_mode}")
            print(f"current_arm_q: {np.round(current_arm_q, 4)}")
            print(f"target_arm_q:  {np.round(q_target, 4)}")
            print(f"clipped_q:     {np.round(clipped, 4)}")


def main():
    rclpy.init()
    node = H1UnitreeStyleArmController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
