#!/usr/bin/env python3
import math
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node

from upper_body_msgs.msg import UpperBodyCommand
from unitree_go.msg import LowCmd, LowState


POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0

H1_ARM_ID = {
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

ARM_IDS = [12, 13, 14, 15, 16, 17, 18, 19]


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class SimUpperBodyToLowCmdBridge(Node):
    def __init__(self):
        super().__init__("sim_upper_body_to_lowcmd_bridge")

        self.declare_parameter("input_topic", "/upper_body/command_geom")
        self.declare_parameter("lowcmd_topic", "/lowcmd")
        self.declare_parameter("lowstate_topic", "/lowstate")

        self.declare_parameter("rate_hz", 250.0)
        self.declare_parameter("kp_arm", 700.0)
        self.declare_parameter("kd_arm", 12.0)
        self.declare_parameter("max_step_rad", 0.012)
        self.declare_parameter("command_timeout_sec", 0.5)

        self.input_topic = self.get_parameter("input_topic").value
        self.lowcmd_topic = self.get_parameter("lowcmd_topic").value
        self.lowstate_topic = self.get_parameter("lowstate_topic").value

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.kp_arm = float(self.get_parameter("kp_arm").value)
        self.kd_arm = float(self.get_parameter("kd_arm").value)
        self.max_step = float(self.get_parameter("max_step_rad").value)
        self.timeout_sec = float(self.get_parameter("command_timeout_sec").value)

        self.current_q: Optional[List[float]] = None
        self.target_q: Optional[List[float]] = None
        self.sent_q: Optional[List[float]] = None

        self.last_cmd_time = self.get_clock().now()
        self.got_cmd = False
        self.seq = 0

        self.pub = self.create_publisher(LowCmd, self.lowcmd_topic, 10)

        self.state_sub = self.create_subscription(
            LowState,
            self.lowstate_topic,
            self.lowstate_cb,
            10,
        )

        self.cmd_sub = self.create_subscription(
            UpperBodyCommand,
            self.input_topic,
            self.cmd_cb,
            10,
        )

        self.timer = self.create_timer(
            1.0 / max(1.0, self.rate_hz),
            self.timer_cb,
        )

        self.get_logger().info("============================================================")
        self.get_logger().info("SIM UPPER BODY COMMAND -> LOWCMD BRIDGE")
        self.get_logger().info(f"input_topic:         {self.input_topic}")
        self.get_logger().info(f"lowcmd_topic:        {self.lowcmd_topic}")
        self.get_logger().info(f"lowstate_topic:      {self.lowstate_topic}")
        self.get_logger().info(f"kp_arm:              {self.kp_arm}")
        self.get_logger().info(f"kd_arm:              {self.kd_arm}")
        self.get_logger().info(f"max_step_rad:        {self.max_step}")
        self.get_logger().info(f"command_timeout_sec: {self.timeout_sec}")
        self.get_logger().info("============================================================")

    def lowstate_cb(self, msg: LowState):
        q = [float(m.q) for m in msg.motor_state]

        if len(q) < 20:
            return

        self.current_q = q

        if self.target_q is None:
            self.target_q = list(q)
            self.sent_q = list(q)
            self.get_logger().info(
                "lowstate OK, initial arm q: "
                + str([round(self.sent_q[i], 4) for i in ARM_IDS])
            )

    def cmd_cb(self, msg: UpperBodyCommand):
        if not msg.valid:
            return

        if self.target_q is None:
            return

        if len(msg.joint_names) != len(msg.position):
            self.get_logger().warn("Bad UpperBodyCommand sizes")
            return

        mapped = 0
        for name, pos in zip(msg.joint_names, msg.position):
            mid = H1_ARM_ID.get(name)
            if mid is None:
                continue

            q = float(pos)
            if not math.isfinite(q):
                continue

            self.target_q[mid] = q
            mapped += 1

        if mapped > 0:
            self.got_cmd = True
            self.last_cmd_time = self.get_clock().now()

        if self.seq % 125 == 0:
            self.get_logger().info(
                f"cmd mapped={mapped} target_arm="
                f"{[round(self.target_q[i], 3) for i in ARM_IDS]}"
            )

    def make_lowcmd(self) -> LowCmd:
        msg = LowCmd()

        # H1/unitree_go style header
        msg.head[0] = 0xFE
        msg.head[1] = 0xEF
        msg.level_flag = 0xFF
        msg.gpio = 0

        # По умолчанию не трогаем не-arm моторы.
        # Для arm моторов задаём позиционный PD.
        for i, motor in enumerate(msg.motor_cmd):
            if i in ARM_IDS:
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

        return msg

    def timer_cb(self):
        if self.target_q is None or self.sent_q is None:
            return

        age = (self.get_clock().now() - self.last_cmd_time).nanoseconds * 1e-9
        timeout = (not self.got_cmd) or (age > self.timeout_sec)

        # Если команды пропали, удерживаем последнее отправленное положение.
        if timeout:
            for i in ARM_IDS:
                self.target_q[i] = self.sent_q[i]

        for i in ARM_IDS:
            d = self.target_q[i] - self.sent_q[i]
            d = clamp(d, -self.max_step, self.max_step)
            self.sent_q[i] += d

        lowcmd = self.make_lowcmd()
        self.pub.publish(lowcmd)
        self.seq += 1

        if self.seq % 125 == 0:
            cur = self.current_q if self.current_q is not None else [0.0] * 20
            self.get_logger().info(
                f"timeout={int(timeout)} "
                f"target={[round(self.target_q[i], 3) for i in ARM_IDS]} "
                f"sent={[round(self.sent_q[i], 3) for i in ARM_IDS]} "
                f"current={[round(cur[i], 3) for i in ARM_IDS]}"
            )


def main():
    rclpy.init()
    node = SimUpperBodyToLowCmdBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
