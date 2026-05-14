#!/usr/bin/env python3
import math
import socket
import struct

import rclpy
from rclpy.node import Node
from upper_body_msgs.msg import UpperBodyCommand

MAGIC = 0x48314152  # H1AR
PACK_FMT = "<IId8f"

ALIASES = {
    "right_shoulder_pitch": 0,
    "right_shoulder_pitch_joint": 0,
    "right_shoulder_roll": 1,
    "right_shoulder_roll_joint": 1,
    "right_shoulder_yaw": 2,
    "right_shoulder_yaw_joint": 2,
    "right_elbow": 3,
    "right_elbow_joint": 3,

    "left_shoulder_pitch": 4,
    "left_shoulder_pitch_joint": 4,
    "left_shoulder_roll": 5,
    "left_shoulder_roll_joint": 5,
    "left_shoulder_yaw": 6,
    "left_shoulder_yaw_joint": 6,
    "left_elbow": 7,
    "left_elbow_joint": 7,
}

class Forwarder(Node):
    def __init__(self):
        super().__init__("laptop_upper_body_udp_forwarder")

        self.declare_parameter("input_topic", "/upper_body/command_geom")
        self.declare_parameter("robot_ip", "192.168.123.162")
        self.declare_parameter("robot_port", 50051)

        self.input_topic = self.get_parameter("input_topic").value
        self.robot_ip = self.get_parameter("robot_ip").value
        self.robot_port = int(self.get_parameter("robot_port").value)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0

        self.sub = self.create_subscription(
            UpperBodyCommand,
            self.input_topic,
            self.cb,
            10,
        )

        self.get_logger().info("Laptop UDP forwarder started")
        self.get_logger().info(f"input_topic: {self.input_topic}")
        self.get_logger().info(f"udp target:  {self.robot_ip}:{self.robot_port}")

    def cb(self, msg: UpperBodyCommand):
        if not msg.valid:
            return

        q = [math.nan] * 8
        mapped = 0

        for name, pos in zip(msg.joint_names, msg.position):
            idx = ALIASES.get(name)
            if idx is None:
                continue
            if math.isfinite(float(pos)):
                q[idx] = float(pos)
                mapped += 1

        if mapped == 0:
            self.get_logger().warn(
                "No known joint names mapped. Incoming names: " + ", ".join(msg.joint_names)
            )
            return

        stamp = self.get_clock().now().nanoseconds * 1e-9
        packet = struct.pack(PACK_FMT, MAGIC, self.seq, stamp, *q)
        self.sock.sendto(packet, (self.robot_ip, self.robot_port))

        if self.seq % 15 == 0:
            self.get_logger().info(
                f"seq={self.seq} mapped={mapped} "
                f"q_r=[{q[0]:.3f} {q[1]:.3f} {q[2]:.3f} {q[3]:.3f}] "
                f"q_l=[{q[4]:.3f} {q[5]:.3f} {q[6]:.3f} {q[7]:.3f}]"
            )

        self.seq += 1

def main():
    rclpy.init()
    node = Forwarder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
