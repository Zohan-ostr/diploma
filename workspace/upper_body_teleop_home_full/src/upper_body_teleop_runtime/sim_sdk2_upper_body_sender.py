#!/usr/bin/env python3
import inspect
import math
import os
import threading
import time
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node

from upper_body_msgs.msg import UpperBodyCommand

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.utils.crc import CRC

# Важно:
# В official unitree_mujoco для ROBOT="h1" используется unitree_go IDL.
# unitree_hg в этом симуляторе включается только для ROBOT="g1".
from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
    LowCmd_,
    LowState_,
    MotorCmd_,
    BmsCmd_,
)


POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0

# Временная карта как для real H1 через unitree_go.
# После просмотра actuator order в MuJoCo при необходимости поправим.
ARM_IDS = [12, 13, 14, 15, 16, 17, 18, 19]

H1_ARM_ID: Dict[str, int] = {
    "right_shoulder_pitch": 12,
    "right_shoulder_roll": 13,
    "right_shoulder_yaw": 14,
    "right_elbow": 15,

    "left_shoulder_pitch": 16,
    "left_shoulder_roll": 17,
    "left_shoulder_yaw": 18,
    "left_elbow": 19,

    "right_shoulder_pitch_joint": 12,
    "right_shoulder_roll_joint": 13,
    "right_shoulder_yaw_joint": 14,
    "right_elbow_joint": 15,

    "left_shoulder_pitch_joint": 16,
    "left_shoulder_roll_joint": 17,
    "left_shoulder_yaw_joint": 18,
    "left_elbow_joint": 19,
}


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def fmt_arm(q: List[float]) -> str:
    return "[" + ", ".join(f"{q[i]: .3f}" for i in ARM_IDS if i < len(q)) + "]"


def construct_dataclass_like(cls, defaults: Dict):
    try:
        return cls()
    except TypeError:
        pass

    sig = inspect.signature(cls)
    kwargs = {}

    for name in sig.parameters:
        if name in defaults:
            kwargs[name] = defaults[name]
        else:
            raise TypeError(f"No default value for {cls.__name__}.{name}")

    return cls(**kwargs)


def make_motor_cmd() -> MotorCmd_:
    return construct_dataclass_like(
        MotorCmd_,
        {
            "mode": 0,
            "q": 0.0,
            "dq": 0.0,
            "tau": 0.0,
            "kp": 0.0,
            "kd": 0.0,
            "reserve": [0, 0, 0],
        },
    )


def make_bms_cmd() -> BmsCmd_:
    return construct_dataclass_like(
        BmsCmd_,
        {
            "off": 0,
            "reserve": [0, 0, 0],
        },
    )


def make_lowcmd_template() -> LowCmd_:
    return construct_dataclass_like(
        LowCmd_,
        {
            "head": [0xFE, 0xEF],
            "level_flag": 0xFF,
            "frame_reserve": 0,
            "sn": [0, 0],
            "version": [0, 0],
            "bandwidth": 0,
            "motor_cmd": [make_motor_cmd() for _ in range(20)],
            "bms_cmd": make_bms_cmd(),
            "wireless_remote": [0] * 40,
            "led": [0] * 12,
            "fan": [0] * 2,
            "gpio": 0,
            "reserve": 0,
            "crc": 0,
        },
    )


class SimSdk2UpperBodySender(Node):
    def __init__(self):
        super().__init__("sim_sdk2_upper_body_sender")

        self.declare_parameter("input_topic", "/upper_body/command_geom")
        self.declare_parameter("control_hz", 250.0)
        self.declare_parameter("kp_arm", 700.0)
        self.declare_parameter("kd_arm", 12.0)
        self.declare_parameter("max_step_rad", 0.012)
        self.declare_parameter("command_timeout_sec", 0.5)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.control_hz = float(self.get_parameter("control_hz").value)
        self.kp_arm = float(self.get_parameter("kp_arm").value)
        self.kd_arm = float(self.get_parameter("kd_arm").value)
        self.max_step_rad = float(self.get_parameter("max_step_rad").value)
        self.command_timeout_sec = float(self.get_parameter("command_timeout_sec").value)

        self.sdk_cmd_topic = os.environ.get("SDK_CMD_TOPIC", "rt/lowcmd")
        self.sdk_state_topic = os.environ.get("SDK_STATE_TOPIC", "rt/lowstate")

        self.lock = threading.Lock()

        self.lowstate_count = 0
        self.last_lowstate_time = 0.0

        self.current_q: Optional[List[float]] = None
        self.target_q: Optional[List[float]] = None
        self.sent_q: Optional[List[float]] = None

        self.have_cmd = False
        self.last_cmd_time = time.monotonic()

        self.pub = ChannelPublisher(self.sdk_cmd_topic, LowCmd_)
        self.pub.Init()

        self.sub = ChannelSubscriber(self.sdk_state_topic, LowState_)
        self.sub.Init(self.lowstate_cb, 10)

        self.cmd_sub = self.create_subscription(
            UpperBodyCommand,
            self.input_topic,
            self.command_cb,
            10,
        )

        self.timer = self.create_timer(
            1.0 / max(1.0, self.control_hz),
            self.timer_cb,
        )

        self.seq = 0
        self.crc = CRC()

        self.get_logger().info("============================================================")
        self.get_logger().info("SIM SDK2 UPPER BODY SENDER")
        self.get_logger().info("SDK IDL:              unitree_go")
        self.get_logger().info(f"input_topic:          {self.input_topic}")
        self.get_logger().info(f"sdk_cmd_topic:        {self.sdk_cmd_topic}")
        self.get_logger().info(f"sdk_state_topic:      {self.sdk_state_topic}")
        self.get_logger().info(f"control_hz:           {self.control_hz}")
        self.get_logger().info(f"kp_arm / kd_arm:      {self.kp_arm} / {self.kd_arm}")
        self.get_logger().info(f"max_step_rad:         {self.max_step_rad}")
        self.get_logger().info(f"command_timeout_sec:  {self.command_timeout_sec}")
        self.get_logger().info(f"arm ids:              {ARM_IDS}")
        self.get_logger().info("============================================================")

    def lowstate_cb(self, msg: LowState_):
        q = [float(m.q) for m in msg.motor_state]

        if len(q) < 20:
            return

        now = time.monotonic()

        with self.lock:
            self.lowstate_count += 1
            self.last_lowstate_time = now
            self.current_q = q

            if self.target_q is None:
                self.target_q = list(q)
                self.sent_q = list(q)
                self.get_logger().info("lowstate OK, initial arm q: " + fmt_arm(q))

    def command_cb(self, msg: UpperBodyCommand):
        if not msg.valid:
            return

        if len(msg.joint_names) != len(msg.position):
            self.get_logger().warn("Bad UpperBodyCommand sizes")
            return

        with self.lock:
            if self.target_q is None:
                return

            mapped = 0

            for name, pos in zip(msg.joint_names, msg.position):
                mid = H1_ARM_ID.get(str(name))
                if mid is None or mid >= len(self.target_q):
                    continue

                q = float(pos)
                if not math.isfinite(q):
                    continue

                self.target_q[mid] = q
                mapped += 1

            if mapped > 0:
                self.have_cmd = True
                self.last_cmd_time = time.monotonic()

    def make_lowcmd(self) -> LowCmd_:
        msg = make_lowcmd_template()

        assert self.sent_q is not None

        for i, motor in enumerate(msg.motor_cmd):
            if i in ARM_IDS and i < len(self.sent_q):
                motor.mode = 0x01
                motor.q = float(self.sent_q[i])
                motor.dq = 0.0
                motor.kp = float(self.kp_arm)
                motor.kd = float(self.kd_arm)
                motor.tau = 0.0
            else:
                motor.mode = 0x01
                motor.q = float(POS_STOP_F)
                motor.dq = float(VEL_STOP_F)
                motor.kp = 0.0
                motor.kd = 0.0
                motor.tau = 0.0

        msg.crc = self.crc.Crc(msg)
        return msg

    def timer_cb(self):
        with self.lock:
            if self.target_q is None or self.sent_q is None:
                return

            age = time.monotonic() - self.last_cmd_time
            timeout = (not self.have_cmd) or (age > self.command_timeout_sec)

            if timeout:
                for i in ARM_IDS:
                    if i < len(self.target_q):
                        self.target_q[i] = self.sent_q[i]

            for i in ARM_IDS:
                if i >= len(self.sent_q) or i >= len(self.target_q):
                    continue

                delta = self.target_q[i] - self.sent_q[i]
                delta = clamp(delta, -self.max_step_rad, self.max_step_rad)
                self.sent_q[i] += delta

            lowcmd = self.make_lowcmd()

        self.pub.Write(lowcmd)
        self.seq += 1

        if self.seq % int(max(1.0, self.control_hz)) == 0:
            with self.lock:
                target = self.target_q if self.target_q else [0.0] * 20
                sent = self.sent_q if self.sent_q else [0.0] * 20
                current = self.current_q if self.current_q else [0.0] * 20

            self.get_logger().info(
                f"timeout={int(timeout)} "
                f"lowstate_count={self.lowstate_count} "
                f"target={fmt_arm(target)} "
                f"sent={fmt_arm(sent)} "
                f"current={fmt_arm(current)}"
            )


def main():
    domain_id = int(os.environ.get("UNITREE_DOMAIN_ID", "42"))
    iface = os.environ.get("UNITREE_NET_IFACE", "lo")

    print("============================================================")
    print("SIM SDK2 SENDER INIT")
    print("SDK IDL:   unitree_go")
    print(f"domain_id: {domain_id}")
    print(f"iface:     {iface}")
    print("============================================================")

    ChannelFactoryInitialize(domain_id, iface)

    rclpy.init()
    node = SimSdk2UpperBodySender()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
