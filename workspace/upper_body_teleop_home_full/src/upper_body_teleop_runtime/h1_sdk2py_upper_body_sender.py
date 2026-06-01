#!/usr/bin/env python3
# ============================================================
# H1 SDK2 PYTHON UPPER BODY SENDER
# ============================================================
#
# Назначение файла:
#   принять готовые команды верхней части тела из ROS 2
#   и отправить их на реального Unitree H1 через SDK2 Python / DDS.
#
# Вход:
#   /upper_body/command_geom
#
# Выход:
#   rt/lowcmd
#
# ВАЖНО ДЛЯ РЕАЛЬНОГО РОБОТА:
#   PD-регуляторы НЕ менять.
#   Для плеч и локтей используются официальные параметры из H1 SDK2
#   low-level example:
#
#       kp = 60.0
#       kd = 1.5
#
#   Эти значения нельзя "подбирать на глаз" на реальном роботе.
#   Любые эксперименты с PD — только в симуляции или на отдельном стенде.
# ============================================================

import os
import time
import threading
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from upper_body_msgs.msg import UpperBodyCommand

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

from upper_body_teleop_runtime.h1_robot_common import (
    ARM_MOTOR_IDS,
    TPOSE_8,
    clamp8,
    apply_self_collision_guard,
    default_step_limits,
    finite_array,
    format_q8,
    motor_arm_q_to_q8,
    parse_q8_from_upper_body_msg,
    q8_to_motor_targets,
    rate_limit8,
)

POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0
H1_NUM_MOTOR = 20


class LowStateBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.msg = None
        self.stamp = 0.0
        self.count = 0

    def set(self, msg):
        with self.lock:
            self.msg = msg
            self.stamp = time.monotonic()
            self.count += 1

    def get(self):
        with self.lock:
            return self.msg, self.stamp, self.count


class H1Sdk2PyUpperBodySender(Node):
    def __init__(self):
        super().__init__("h1_sdk2py_upper_body_sender")

        self.declare_parameter("input_topic", os.environ.get("INPUT_TOPIC", "/h1/upper_body_safe_cmd"))
        self.declare_parameter("sdk_cmd_topic", os.environ.get("SDK_CMD_TOPIC", "rt/lowcmd"))
        self.declare_parameter("sdk_state_topic", os.environ.get("SDK_STATE_TOPIC", "rt/lowstate"))

        self.declare_parameter("control_hz", float(os.environ.get("CONTROL_HZ", "100.0")))
        self.declare_parameter("command_timeout_sec", float(os.environ.get("COMMAND_TIMEOUT_SEC", "0.30")))
        self.declare_parameter("lowstate_timeout_sec", float(os.environ.get("LOWSTATE_TIMEOUT_SEC", "0.50")))

        self.declare_parameter("kp_arm", float(os.environ.get("KP_ARM", "60.0")))
        self.declare_parameter("kd_arm", float(os.environ.get("KD_ARM", "1.5")))


        self.input_topic = str(self.get_parameter("input_topic").value)
        self.sdk_cmd_topic = str(self.get_parameter("sdk_cmd_topic").value)
        self.sdk_state_topic = str(self.get_parameter("sdk_state_topic").value)

        self.control_hz = float(self.get_parameter("control_hz").value)
        self.dt = 1.0 / max(1.0, self.control_hz)

        self.command_timeout_sec = float(self.get_parameter("command_timeout_sec").value)
        self.lowstate_timeout_sec = float(self.get_parameter("lowstate_timeout_sec").value)

        self.kp_arm = float(self.get_parameter("kp_arm").value)
        self.kd_arm = float(self.get_parameter("kd_arm").value)

        self.crc = CRC()
        self.lowstate = LowStateBuffer()

        self.lowcmd_pub = ChannelPublisher(self.sdk_cmd_topic, LowCmd_)
        self.lowcmd_pub.Init()

        self.lowstate_sub = ChannelSubscriber(self.sdk_state_topic, LowState_)
        self.lowstate_sub.Init(self.lowstate_callback, 10)

        self.release_motion_mode()

        self.get_logger().info("Waiting for rt/lowstate callback...")
        t0 = time.monotonic()
        while rclpy.ok():
            msg, _, count = self.lowstate.get()
            if msg is not None:
                break
            if time.monotonic() - t0 > 5.0:
                raise RuntimeError("No lowstate received in 5 sec")
            time.sleep(0.02)

        self.lowcmd = unitree_go_msg_dds__LowCmd_()
        self.init_lowcmd()

        motor_q = self.read_motor_q()
        self.q8_sent = motor_arm_q_to_q8(motor_q)
        self.q8_target = self.q8_sent.copy()

        self.max_step = default_step_limits(self.control_hz)

        self.last_cmd_time = 0.0
        self.have_cmd = False
        self.cmd_lock = threading.Lock()

        self.sub = self.create_subscription(UpperBodyCommand, self.input_topic, self.on_cmd, 10)
        self.timer = self.create_timer(self.dt, self.on_timer)

        self.seq = 0

        self.get_logger().info("============================================================")
        self.get_logger().info("H1 SDK2PY UPPER BODY SENDER")
        self.get_logger().info("SDK IDL:             unitree_go")
        self.get_logger().info(f"input_topic:         {self.input_topic}")
        self.get_logger().info(f"sdk_cmd_topic:       {self.sdk_cmd_topic}")
        self.get_logger().info(f"sdk_state_topic:     {self.sdk_state_topic}")
        self.get_logger().info(f"control_hz:          {self.control_hz}")
        self.get_logger().info(f"kp/kd arm:           {self.kp_arm} / {self.kd_arm}")
        self.get_logger().info(f"initial q8:          {format_q8(self.q8_sent)}")
        self.get_logger().info("============================================================")

    def release_motion_mode(self):
        try:
            msc = MotionSwitcherClient()
            msc.SetTimeout(5.0)
            msc.Init()

            status, result = msc.CheckMode()
            self.get_logger().warn(f"MotionSwitcher CheckMode status={status}, result={result}")

            while isinstance(result, dict) and result.get("name"):
                self.get_logger().warn(f"Release motion mode: {result}")
                msc.ReleaseMode()
                time.sleep(1.0)
                status, result = msc.CheckMode()

            self.get_logger().info("Motion mode released or already empty.")
        except Exception as e:
            self.get_logger().warn(f"MotionSwitcher release skipped/failed: {e}")

    def lowstate_callback(self, msg: LowState_):
        self.lowstate.set(msg)

    def init_lowcmd(self):
        self.lowcmd.head[0] = 0xFE
        self.lowcmd.head[1] = 0xEF
        self.lowcmd.level_flag = 0xFF
        self.lowcmd.gpio = 0

        for i in range(H1_NUM_MOTOR):
            if i in ARM_MOTOR_IDS:
                self.lowcmd.motor_cmd[i].mode = 0x01
            else:
                self.lowcmd.motor_cmd[i].mode = 0x0A

            self.lowcmd.motor_cmd[i].q = POS_STOP_F
            self.lowcmd.motor_cmd[i].dq = VEL_STOP_F
            self.lowcmd.motor_cmd[i].tau = 0.0
            self.lowcmd.motor_cmd[i].kp = 0.0
            self.lowcmd.motor_cmd[i].kd = 0.0

    def read_motor_q(self) -> np.ndarray:
        msg, _, _ = self.lowstate.get()
        if msg is None:
            raise RuntimeError("lowstate is None")

        q = [float(m.q) for m in msg.motor_state[:H1_NUM_MOTOR]]
        return np.array(q, dtype=float)

    def lowstate_alive(self) -> bool:
        _, stamp, _ = self.lowstate.get()
        return stamp > 0.0 and (time.monotonic() - stamp) <= self.lowstate_timeout_sec

    def on_cmd(self, msg: UpperBodyCommand):
        if not msg.valid:
            return

        try:
            q8 = parse_q8_from_upper_body_msg(msg)
            q8 = apply_self_collision_guard(q8)
        except Exception as e:
            self.get_logger().warn(f"bad command: {e}", throttle_duration_sec=1.0)
            return

        if not finite_array(q8):
            self.get_logger().warn("bad command: non-finite values", throttle_duration_sec=1.0)
            return

        with self.cmd_lock:
            self.q8_target = q8.copy()
            self.last_cmd_time = time.monotonic()
            self.have_cmd = True

    def make_test_q8(self) -> Optional[np.ndarray]:
        if self.test_mode == "none":
            return None

        if self.test_mode == "tpose":
            return clamp8(TPOSE_8.copy())

        # Малые тесты вокруг текущей позы, чтобы не угадывать абсолютные углы.
        # Индексы q8:
        # 0 left_pitch, 1 left_roll, 2 left_yaw, 3 left_elbow,
        # 4 right_pitch, 5 right_roll, 6 right_yaw, 7 right_elbow.
        test_map = {
            "left_pitch": 0,
            "left_roll": 1,
            "left_yaw": 2,
            "left_elbow": 3,
            "right_pitch": 4,
            "right_roll": 5,
            "right_yaw": 6,
            "right_elbow": 7,
        }

        if self.test_mode not in test_map:
            return None

        base = self.q8_sent.copy()
        idx = test_map[self.test_mode]

        amp = float(os.environ.get("TEST_AMP", "0.20"))
        freq = float(os.environ.get("TEST_FREQ", "0.12"))
        delta = amp * np.sin(2.0 * np.pi * freq * time.monotonic())

        base[idx] = self.q8_sent[idx] + delta
        return clamp8(base)

    def fill_lowcmd(self, motor_q: np.ndarray):
        for i in range(H1_NUM_MOTOR):
            cmd = self.lowcmd.motor_cmd[i]

            if i in ARM_MOTOR_IDS:
                cmd.mode = 0x01
                cmd.q = float(motor_q[i])
                cmd.dq = 0.0
                cmd.tau = 0.0
                cmd.kp = self.kp_arm
                cmd.kd = self.kd_arm
            else:
                cmd.q = POS_STOP_F
                cmd.dq = VEL_STOP_F
                cmd.tau = 0.0
                cmd.kp = 0.0
                cmd.kd = 0.0

        self.lowcmd.crc = self.crc.Crc(self.lowcmd)

    def on_timer(self):
        self.seq += 1

        if not self.lowstate_alive():
            self.get_logger().warn("lowstate timeout: command not sent", throttle_duration_sec=1.0)
            return

        with self.cmd_lock:
            age = time.monotonic() - self.last_cmd_time
            have = self.have_cmd
            target = self.q8_target.copy()

        if have and age <= self.command_timeout_sec:
            desired_q8 = target
        else:
            desired_q8 = self.q8_sent.copy()

        desired_q8 = apply_self_collision_guard(desired_q8)
        self.q8_sent = rate_limit8(desired_q8, self.q8_sent, self.max_step)
        self.q8_sent = apply_self_collision_guard(self.q8_sent)

        current_motor_q = self.read_motor_q()
        motor_target_q = q8_to_motor_targets(self.q8_sent, current_motor_q)

        self.fill_lowcmd(motor_target_q)
        self.lowcmd_pub.Write(self.lowcmd)

        if self.seq % int(max(1, self.control_hz)) == 0:
            _, _, count = self.lowstate.get()
            self.get_logger().info(
                f"sent q8={format_q8(self.q8_sent)} "
                f"have={int(have)} age={age:.3f} lowstate_count={count}"
            )


def main():
    domain = int(os.environ.get("UNITREE_DOMAIN_ID", os.environ.get("ROS_DOMAIN_ID", "0")))
    iface = os.environ.get("UNITREE_NET_IFACE", "")

    print()
    print("============================================================")
    print("H1 SDK2 Python sender")
    print("IDL: unitree_go")
    print(f"domain={domain}")
    print(f"iface={iface!r}")
    print("WARNING: low-level sender writes rt/lowcmd.")
    print("Make sure no high-level motion service conflicts with it.")
    print("============================================================")
    print()

    if iface:
        ChannelFactoryInitialize(domain, iface)
    else:
        ChannelFactoryInitialize(domain)

    rclpy.init()
    node = H1Sdk2PyUpperBodySender()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
