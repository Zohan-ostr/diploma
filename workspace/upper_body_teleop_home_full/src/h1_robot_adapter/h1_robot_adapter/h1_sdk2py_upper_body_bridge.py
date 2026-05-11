#!/usr/bin/env python3
import os
import time
import threading
from enum import IntEnum
from typing import Dict, Optional

import numpy as np
import rclpy
from rclpy.node import Node

from upper_body_msgs.msg import UpperBodyCommand

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC


class H1JointIndex(IntEnum):
    kRightHipRoll = 0
    kRightHipPitch = 1
    kRightKnee = 2

    kLeftHipRoll = 3
    kLeftHipPitch = 4
    kLeftKnee = 5

    kWaistYaw = 6

    kLeftHipYaw = 7
    kRightHipYaw = 8

    kNotUsedJoint = 9

    kLeftAnkle = 10
    kRightAnkle = 11

    kRightShoulderPitch = 12
    kRightShoulderRoll = 13
    kRightShoulderYaw = 14
    kRightElbow = 15

    kLeftShoulderPitch = 16
    kLeftShoulderRoll = 17
    kLeftShoulderYaw = 18
    kLeftElbow = 19


ARM_JOINT_MAP = {
    "left_shoulder_pitch_joint": H1JointIndex.kLeftShoulderPitch,
    "left_shoulder_roll_joint": H1JointIndex.kLeftShoulderRoll,
    "left_shoulder_yaw_joint": H1JointIndex.kLeftShoulderYaw,
    "left_elbow_joint": H1JointIndex.kLeftElbow,

    "right_shoulder_pitch_joint": H1JointIndex.kRightShoulderPitch,
    "right_shoulder_roll_joint": H1JointIndex.kRightShoulderRoll,
    "right_shoulder_yaw_joint": H1JointIndex.kRightShoulderYaw,
    "right_elbow_joint": H1JointIndex.kRightElbow,
}

DEFAULT_ORDER = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]


def clamp_array_step(target: np.ndarray, current: np.ndarray, max_step: float) -> np.ndarray:
    return current + np.clip(target - current, -max_step, max_step)


class LowStateBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.msg = None

    def set(self, msg):
        with self.lock:
            self.msg = msg

    def get(self):
        with self.lock:
            return self.msg


class H1Sdk2PyUpperBodyBridge(Node):
    def __init__(self):
        super().__init__("h1_sdk2py_upper_body_bridge")

        self.declare_parameter("input_topic", os.environ.get("INPUT_TOPIC", "/upper_body/command_geom"))
        self.declare_parameter("command_timeout_sec", float(os.environ.get("COMMAND_TIMEOUT_SEC", "0.5")))

        self.declare_parameter("kp_arm", float(os.environ.get("KP_LOW", "400.0")))
        self.declare_parameter("kd_arm", float(os.environ.get("KD_LOW", "8.0")))

        self.declare_parameter("kp_body", float(os.environ.get("KP_HIGH", "300.0")))
        self.declare_parameter("kd_body", float(os.environ.get("KD_HIGH", "5.0")))

        self.declare_parameter("arm_velocity_limit", float(os.environ.get("ARM_VELOCITY_LIMIT", "100.0")))
        self.declare_parameter("control_hz", float(os.environ.get("CONTROL_HZ", "250.0")))

        self.declare_parameter("hold_current_on_timeout", True)
        self.declare_parameter("test_mode", os.environ.get("TEST_MODE", "none"))

        self.input_topic = self.get_parameter("input_topic").value
        self.command_timeout_sec = float(self.get_parameter("command_timeout_sec").value)

        self.kp_arm = float(self.get_parameter("kp_arm").value)
        self.kd_arm = float(self.get_parameter("kd_arm").value)

        self.kp_body = float(self.get_parameter("kp_body").value)
        self.kd_body = float(self.get_parameter("kd_body").value)

        self.arm_velocity_limit = float(self.get_parameter("arm_velocity_limit").value)
        self.control_hz = float(self.get_parameter("control_hz").value)
        self.control_dt = 1.0 / self.control_hz

        self.hold_current_on_timeout = bool(self.get_parameter("hold_current_on_timeout").value)
        self.test_mode = str(self.get_parameter("test_mode").value)

        self.lowstate_buffer = LowStateBuffer()
        self.crc = CRC()

        self.lowcmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_pub.Init()

        self.lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_sub.Init()

        self.lowstate_thread = threading.Thread(target=self.lowstate_loop, daemon=True)
        self.lowstate_thread.start()

        self.get_logger().info("Waiting for rt/lowstate...")
        while rclpy.ok() and self.lowstate_buffer.get() is None:
            time.sleep(0.05)

        self.get_logger().info("rt/lowstate OK")

        self.msg = unitree_go_msg_dds__LowCmd_()
        self.init_lowcmd_header()

        self.all_q = self.read_all_q()
        self.target_q_by_motor = self.all_q.copy()
        self.current_cmd_q = self.all_q.copy()

        self.configure_lock_all_joints()

        self.last_ros_cmd_time = 0.0
        self.last_valid_command_q = self.get_current_arm_q()

        self.cmd_lock = threading.Lock()

        self.sub = self.create_subscription(
            UpperBodyCommand,
            self.input_topic,
            self.on_upper_body_command,
            10,
        )

        self.timer = self.create_timer(self.control_dt, self.control_loop)

        self.get_logger().info("===== H1 SDK2PY UPPER BODY BRIDGE CLEAN VERSION =====")
        self.get_logger().info(f"input_topic:         {self.input_topic}")
        self.get_logger().info(f"kp_arm/kd_arm:       {self.kp_arm} / {self.kd_arm}")
        self.get_logger().info(f"arm_velocity_limit:  {self.arm_velocity_limit}")
        self.get_logger().info(f"command_timeout_sec: {self.command_timeout_sec}")
        self.get_logger().info(f"test_mode:           {self.test_mode}")

    def init_lowcmd_header(self):
        self.msg.head[0] = 0xFE
        self.msg.head[1] = 0xEF
        self.msg.level_flag = 0xFF
        self.msg.gpio = 0

    def lowstate_loop(self):
        while True:
            msg = self.lowstate_sub.Read()
            if msg is not None:
                self.lowstate_buffer.set(msg)
            time.sleep(0.002)

    def read_all_q(self) -> np.ndarray:
        st = self.lowstate_buffer.get()
        q = np.zeros(20, dtype=float)
        if st is None:
            return q
        for i in range(20):
            q[i] = float(st.motor_state[i].q)
        return q

    def get_current_arm_q(self) -> np.ndarray:
        q = self.read_all_q()
        return np.array([
            q[H1JointIndex.kLeftShoulderPitch],
            q[H1JointIndex.kLeftShoulderRoll],
            q[H1JointIndex.kLeftShoulderYaw],
            q[H1JointIndex.kLeftElbow],
            q[H1JointIndex.kRightShoulderPitch],
            q[H1JointIndex.kRightShoulderRoll],
            q[H1JointIndex.kRightShoulderYaw],
            q[H1JointIndex.kRightElbow],
        ], dtype=float)

    def is_arm_motor(self, idx: int) -> bool:
        return idx in {int(v) for v in ARM_JOINT_MAP.values()}

    def is_weak_motor(self, idx: int) -> bool:
        weak = {
            H1JointIndex.kLeftAnkle,
            H1JointIndex.kRightAnkle,

            H1JointIndex.kLeftShoulderPitch,
            H1JointIndex.kLeftShoulderRoll,
            H1JointIndex.kLeftShoulderYaw,
            H1JointIndex.kLeftElbow,

            H1JointIndex.kRightShoulderPitch,
            H1JointIndex.kRightShoulderRoll,
            H1JointIndex.kRightShoulderYaw,
            H1JointIndex.kRightElbow,
        }
        return H1JointIndex(idx) in weak

    def configure_lock_all_joints(self):
        q = self.read_all_q()
        self.target_q_by_motor = q.copy()
        self.current_cmd_q = q.copy()

        for idx in range(20):
            self.msg.motor_cmd[idx].q = float(q[idx])
            self.msg.motor_cmd[idx].dq = 0.0
            self.msg.motor_cmd[idx].tau = 0.0

            if self.is_arm_motor(idx):
                self.msg.motor_cmd[idx].mode = 0x01
                self.msg.motor_cmd[idx].kp = self.kp_arm
                self.msg.motor_cmd[idx].kd = self.kd_arm
            elif self.is_weak_motor(idx):
                self.msg.motor_cmd[idx].mode = 0x01
                self.msg.motor_cmd[idx].kp = self.kp_arm
                self.msg.motor_cmd[idx].kd = self.kd_arm
            else:
                self.msg.motor_cmd[idx].mode = 0x0A
                self.msg.motor_cmd[idx].kp = self.kp_body
                self.msg.motor_cmd[idx].kd = self.kd_body

        self.get_logger().info("Initial joints locked")

    def parse_command(self, msg: UpperBodyCommand) -> Optional[np.ndarray]:
        if not msg.valid:
            return None

        if len(msg.position) < 8:
            self.get_logger().warn("UpperBodyCommand has less than 8 positions")
            return None

        result = np.zeros(8, dtype=float)

        # Если joint_names совпадают — используем имена.
        if len(msg.joint_names) == len(msg.position) and len(msg.joint_names) > 0:
            by_name: Dict[str, float] = {}
            for name, pos in zip(msg.joint_names, msg.position):
                by_name[name] = float(pos)

            missing = [name for name in DEFAULT_ORDER if name not in by_name]
            if missing:
                self.get_logger().warn(f"Missing joints in command: {missing}")
                return None

            for i, name in enumerate(DEFAULT_ORDER):
                result[i] = by_name[name]
            return result

        # Иначе считаем, что порядок стандартный.
        return np.array(msg.position[:8], dtype=float)

    def on_upper_body_command(self, msg: UpperBodyCommand):
        q8 = self.parse_command(msg)
        if q8 is None:
            return

        with self.cmd_lock:
            self.last_valid_command_q = q8.copy()
            self.last_ros_cmd_time = time.time()

    def apply_q8_to_motor_targets(self, q8: np.ndarray):
        self.target_q_by_motor[H1JointIndex.kLeftShoulderPitch] = q8[0]
        self.target_q_by_motor[H1JointIndex.kLeftShoulderRoll] = q8[1]
        self.target_q_by_motor[H1JointIndex.kLeftShoulderYaw] = q8[2]
        self.target_q_by_motor[H1JointIndex.kLeftElbow] = q8[3]

        self.target_q_by_motor[H1JointIndex.kRightShoulderPitch] = q8[4]
        self.target_q_by_motor[H1JointIndex.kRightShoulderRoll] = q8[5]
        self.target_q_by_motor[H1JointIndex.kRightShoulderYaw] = q8[6]
        self.target_q_by_motor[H1JointIndex.kRightElbow] = q8[7]

    def make_test_command(self) -> Optional[np.ndarray]:
        if self.test_mode == "none":
            return None

        base = self.get_current_arm_q()
        t = time.time()
        amp = 0.45
        s = amp * math.sin(2.0 * math.pi * 0.25 * t)

        q = base.copy()

        if self.test_mode == "left_pitch":
            q[0] = -0.10 + s
        elif self.test_mode == "right_pitch":
            q[4] = -0.10 + s
        elif self.test_mode == "both_pitch":
            q[0] = -0.10 + s
            q[4] = -0.10 + s
        elif self.test_mode == "left_roll":
            q[1] = -0.015 + s
        elif self.test_mode == "right_roll":
            q[5] = -0.015 - s
        elif self.test_mode == "left_elbow":
            q[3] = 1.34 + s
        elif self.test_mode == "right_elbow":
            q[7] = 1.32 + s
        else:
            return None

        return q

    def control_loop(self):
        now = time.time()

        test_q = self.make_test_command()

        with self.cmd_lock:
            age = now - self.last_ros_cmd_time
            cmd_q = self.last_valid_command_q.copy()

        if test_q is not None:
            desired_q8 = test_q
        elif self.last_ros_cmd_time > 0 and age <= self.command_timeout_sec:
            desired_q8 = cmd_q
        else:
            if self.hold_current_on_timeout:
                desired_q8 = self.get_current_arm_q()
            else:
                desired_q8 = cmd_q

        self.apply_q8_to_motor_targets(desired_q8)

        max_step = self.arm_velocity_limit * self.control_dt
        self.current_cmd_q = clamp_array_step(self.target_q_by_motor, self.current_cmd_q, max_step)

        # Пишем все 20 моторов, но ноги/корпус остаются зафиксированы около стартовой позы.
        for idx in range(20):
            self.msg.motor_cmd[idx].q = float(self.current_cmd_q[idx])
            self.msg.motor_cmd[idx].dq = 0.0
            self.msg.motor_cmd[idx].tau = 0.0

            if self.is_arm_motor(idx):
                self.msg.motor_cmd[idx].mode = 0x01
                self.msg.motor_cmd[idx].kp = self.kp_arm
                self.msg.motor_cmd[idx].kd = self.kd_arm

        self.msg.crc = self.crc.Crc(self.msg)
        self.lowcmd_pub.Write(self.msg)


def main():
    sdk_domain = int(os.environ.get("UNITREE_SDK_DOMAIN", "1"))
    iface = os.environ.get("UNITREE_NET_IFACE", None)

    if iface:
        ChannelFactoryInitialize(sdk_domain, networkInterface=iface)
    else:
        ChannelFactoryInitialize(sdk_domain)

    rclpy.init()
    node = H1Sdk2PyUpperBodyBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
