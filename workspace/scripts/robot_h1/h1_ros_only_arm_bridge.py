#!/usr/bin/env python3
import math
import time
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node

from upper_body_msgs.msg import UpperBodyCommand
from unitree_go.msg import LowCmd, LowState


H1_ARM_IDS = {
    "right_shoulder_pitch": 12,
    "right_shoulder_roll": 13,
    "right_shoulder_yaw": 14,
    "right_elbow": 15,

    "left_shoulder_pitch": 16,
    "left_shoulder_roll": 17,
    "left_shoulder_yaw": 18,
    "left_elbow": 19,
}

# fallback: если в UpperBodyCommand нет имён или они другие
FALLBACK_ORDER = [
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
]


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def make_lowcmd() -> LowCmd:
    msg = LowCmd()
    try:
        msg.head[0] = 0xFE
        msg.head[1] = 0xEF
    except Exception:
        pass

    if hasattr(msg, "level_flag"):
        msg.level_flag = 0xFF
    if hasattr(msg, "gpio"):
        msg.gpio = 0

    return msg


class H1RosOnlyArmBridge(Node):
    def __init__(self):
        super().__init__("h1_ros_only_arm_bridge")

        self.declare_parameter("input_topic", "/upper_body/command_geom")
        self.declare_parameter("output_topic", "/arm_sdk")
        self.declare_parameter("lowstate_topic", "/lowstate")

        self.declare_parameter("rate_hz", 100.0)
        self.declare_parameter("command_timeout_sec", 0.35)

        # Для первого реального запуска держим мягко
        self.declare_parameter("kp_arm", 45.0)
        self.declare_parameter("kd_arm", 2.0)

        # Ограничение скорости догоняния цели, rad/sec
        self.declare_parameter("velocity_limit", 0.8)

        # Сглаживание цели: 1.0 = без сглаживания, 0.2 = мягко
        self.declare_parameter("target_alpha", 0.35)

        # Если confidence ниже — сустав не обновляем
        self.declare_parameter("min_confidence", 0.25)

        # Публиковать только руки, остальные motor_cmd оставлять нулевыми
        self.declare_parameter("arms_only", True)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.lowstate_topic = self.get_parameter("lowstate_topic").value

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.command_timeout_sec = float(self.get_parameter("command_timeout_sec").value)
        self.kp_arm = float(self.get_parameter("kp_arm").value)
        self.kd_arm = float(self.get_parameter("kd_arm").value)
        self.velocity_limit = float(self.get_parameter("velocity_limit").value)
        self.target_alpha = float(self.get_parameter("target_alpha").value)
        self.min_confidence = float(self.get_parameter("min_confidence").value)
        self.arms_only = bool(self.get_parameter("arms_only").value)

        self.current_q: Dict[str, float] = {}
        self.filtered_target_q: Dict[str, float] = {}
        self.last_cmd_time: Optional[float] = None
        self.last_publish_time = time.time()

        self.pub = self.create_publisher(LowCmd, self.output_topic, 10)

        self.create_subscription(
            LowState,
            self.lowstate_topic,
            self.on_lowstate,
            10,
        )

        self.create_subscription(
            UpperBodyCommand,
            self.input_topic,
            self.on_upper_body_command,
            10,
        )

        period = 1.0 / max(self.rate_hz, 1.0)
        self.timer = self.create_timer(period, self.on_timer)

        self.get_logger().info("============================================================")
        self.get_logger().info("H1 ROS-ONLY ARM BRIDGE")
        self.get_logger().info(f"input_topic:         {self.input_topic}")
        self.get_logger().info(f"output_topic:        {self.output_topic}")
        self.get_logger().info(f"lowstate_topic:      {self.lowstate_topic}")
        self.get_logger().info(f"rate_hz:             {self.rate_hz}")
        self.get_logger().info(f"kp_arm:              {self.kp_arm}")
        self.get_logger().info(f"kd_arm:              {self.kd_arm}")
        self.get_logger().info(f"velocity_limit:      {self.velocity_limit}")
        self.get_logger().info(f"target_alpha:        {self.target_alpha}")
        self.get_logger().info(f"min_confidence:      {self.min_confidence}")
        self.get_logger().info(f"arms_only:           {self.arms_only}")
        self.get_logger().info("Publishes real robot commands to /arm_sdk")
        self.get_logger().info("============================================================")

    def on_lowstate(self, msg: LowState):
        try:
            self.current_q["right_shoulder_pitch"] = float(msg.motor_state[12].q)
            self.current_q["right_shoulder_roll"] = float(msg.motor_state[13].q)
            self.current_q["right_shoulder_yaw"] = float(msg.motor_state[14].q)
            self.current_q["right_elbow"] = float(msg.motor_state[15].q)

            self.current_q["left_shoulder_pitch"] = float(msg.motor_state[16].q)
            self.current_q["left_shoulder_roll"] = float(msg.motor_state[17].q)
            self.current_q["left_shoulder_yaw"] = float(msg.motor_state[18].q)
            self.current_q["left_elbow"] = float(msg.motor_state[19].q)

            if not self.filtered_target_q:
                self.filtered_target_q = dict(self.current_q)

        except Exception as e:
            self.get_logger().warn(f"Failed to parse lowstate: {e}")

    def normalize_name(self, name: str) -> str:
        name = name.strip()
        name = name.replace("_joint", "")
        name = name.replace("kLeft", "left_")
        name = name.replace("kRight", "right_")
        return name

    def command_to_dict(self, msg: UpperBodyCommand) -> Dict[str, float]:
        out: Dict[str, float] = {}

        if not msg.valid:
            return out

        positions = list(msg.position)
        names = [self.normalize_name(n) for n in list(msg.joint_names)]
        conf = list(msg.confidence)

        if len(names) == len(positions) and len(names) > 0:
            for i, name in enumerate(names):
                if name not in H1_ARM_IDS:
                    continue

                c = conf[i] if i < len(conf) else 1.0
                if c < self.min_confidence:
                    continue

                out[name] = float(positions[i])
            return out

        # fallback по порядку
        for i, name in enumerate(FALLBACK_ORDER):
            if i >= len(positions):
                break

            c = conf[i] if i < len(conf) else 1.0
            if c < self.min_confidence:
                continue

            out[name] = float(positions[i])

        return out

    def on_upper_body_command(self, msg: UpperBodyCommand):
        cmd = self.command_to_dict(msg)
        if not cmd:
            return

        now = time.time()
        self.last_cmd_time = now

        if not self.filtered_target_q:
            if self.current_q:
                self.filtered_target_q = dict(self.current_q)
            else:
                self.filtered_target_q = {k: 0.0 for k in H1_ARM_IDS}

        for name, q in cmd.items():
            prev = self.filtered_target_q.get(name, self.current_q.get(name, q))
            self.filtered_target_q[name] = (
                (1.0 - self.target_alpha) * prev + self.target_alpha * q
            )

    def make_safe_target(self, dt: float) -> Dict[str, float]:
        # Если команд давно нет — удерживаем текущие положения, а не продолжаем старую команду
        now = time.time()
        if self.last_cmd_time is None or now - self.last_cmd_time > self.command_timeout_sec:
            if self.current_q:
                return dict(self.current_q)
            return {k: 0.0 for k in H1_ARM_IDS}

        target = {}
        max_step = self.velocity_limit * dt

        for name in H1_ARM_IDS:
            cur = self.current_q.get(name, self.filtered_target_q.get(name, 0.0))
            desired = self.filtered_target_q.get(name, cur)

            delta = clamp(desired - cur, -max_step, max_step)
            target[name] = cur + delta

        return target

    def on_timer(self):
        if not self.current_q:
            return

        now = time.time()
        dt = max(1.0 / self.rate_hz, now - self.last_publish_time)
        self.last_publish_time = now

        target = self.make_safe_target(dt)

        msg = make_lowcmd()

        for name, motor_id in H1_ARM_IDS.items():
            cmd = msg.motor_cmd[motor_id]
            cmd.mode = 0x01
            cmd.q = float(target[name])
            cmd.dq = 0.0
            cmd.tau = 0.0
            cmd.kp = self.kp_arm
            cmd.kd = self.kd_arm

        self.pub.publish(msg)


def main():
    rclpy.init()
    node = H1RosOnlyArmBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
