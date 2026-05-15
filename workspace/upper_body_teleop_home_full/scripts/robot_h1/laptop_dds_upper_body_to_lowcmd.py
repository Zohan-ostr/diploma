#!/usr/bin/env python3
import math
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from upper_body_msgs.msg import UpperBodyCommand
from unitree_go.msg import LowCmd, LowState


POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0

ARM_IDS = [12, 13, 14, 15, 16, 17, 18, 19]

JOINT_TO_ID = {
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


class LaptopDDSUpperBodyToLowCmd(Node):
    def __init__(self):
        super().__init__("laptop_dds_upper_body_to_lowcmd")

        self.declare_parameter("input_topic", "/upper_body/command_geom")
        self.declare_parameter("lowcmd_topic", "/lowcmd")
        self.declare_parameter("lowstate_topic", "/lowstate")

        self.declare_parameter("rate_hz", 250.0)

        # ВАЖНО: безопасные коэффициенты для реального робота.
        self.declare_parameter("kp_arm", 25.0)
        self.declare_parameter("kd_arm", 1.5)

        self.declare_parameter("max_step_rad", 0.012)
        self.declare_parameter("yaw_max_step_rad", 0.020)
        self.declare_parameter("elbow_max_step_rad", 0.025)
        self.declare_parameter("command_timeout_sec", 0.35)

        self.input_topic = self.get_parameter("input_topic").value
        self.lowcmd_topic = self.get_parameter("lowcmd_topic").value
        self.lowstate_topic = self.get_parameter("lowstate_topic").value

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.kp_arm = float(self.get_parameter("kp_arm").value)
        self.kd_arm = float(self.get_parameter("kd_arm").value)

        self.max_step = float(self.get_parameter("max_step_rad").value)
        self.yaw_max_step = float(self.get_parameter("yaw_max_step_rad").value)
        self.elbow_max_step = float(self.get_parameter("elbow_max_step_rad").value)
        self.timeout_sec = float(self.get_parameter("command_timeout_sec").value)

        self.current_q: Optional[List[float]] = None
        self.target_q: Optional[List[float]] = None
        self.sent_q: Optional[List[float]] = None

        self.last_cmd_time = self.get_clock().now()
        self.got_cmd = False
        self.seq = 0

        qos_best_effort = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        qos_reliable = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.pub = self.create_publisher(LowCmd, self.lowcmd_topic, qos_best_effort)

        self.state_sub = self.create_subscription(
            LowState,
            self.lowstate_topic,
            self.lowstate_cb,
            qos_best_effort,
        )

        self.cmd_sub = self.create_subscription(
            UpperBodyCommand,
            self.input_topic,
            self.cmd_cb,
            qos_reliable,
        )

        self.timer = self.create_timer(1.0 / max(1.0, self.rate_hz), self.timer_cb)

        self.get_logger().info("============================================================")
        self.get_logger().info("LAPTOP DDS UPPER BODY COMMAND -> REAL H1 /lowcmd")
        self.get_logger().info(f"input_topic:         {self.input_topic}")
        self.get_logger().info(f"lowcmd_topic:        {self.lowcmd_topic}")
        self.get_logger().info(f"lowstate_topic:      {self.lowstate_topic}")
        self.get_logger().info(f"rate_hz:             {self.rate_hz}")
        self.get_logger().info(f"kp_arm:              {self.kp_arm}")
        self.get_logger().info(f"kd_arm:              {self.kd_arm}")
        self.get_logger().info(f"max_step_rad:        {self.max_step}")
        self.get_logger().info(f"yaw_max_step_rad:    {self.yaw_max_step}")
        self.get_logger().info(f"elbow_max_step_rad:  {self.elbow_max_step}")
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
            motor_id = JOINT_TO_ID.get(name)
            if motor_id is None:
                continue

            q = float(pos)
            if not math.isfinite(q):
                continue

            self.target_q[motor_id] = q
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

        msg.head[0] = 0xFE
        msg.head[1] = 0xEF
        msg.level_flag = 0xFF
        msg.gpio = 0

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

        if timeout:
            for i in ARM_IDS:
                self.target_q[i] = self.sent_q[i]

        for i in ARM_IDS:
            if i in (14, 18):
                limit = self.yaw_max_step
            elif i in (15, 19):
                limit = self.elbow_max_step
            else:
                limit = self.max_step

            delta = self.target_q[i] - self.sent_q[i]
            delta = clamp(delta, -limit, limit)
            self.sent_q[i] += delta

        self.pub.publish(self.make_lowcmd())
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
    node = LaptopDDSUpperBodyToLowCmd()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
