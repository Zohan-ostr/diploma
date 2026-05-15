#!/usr/bin/env python3
import json
import socket

import rclpy
from rclpy.node import Node

from upper_body_msgs.msg import UpperBodyCommand


class UpperBodyUdpForwarder(Node):
    def __init__(self):
        super().__init__("laptop_ros_to_udp_forwarder")

        self.declare_parameter("input_topic", "/upper_body/command_geom")
        self.declare_parameter("udp_host", "127.0.0.1")
        self.declare_parameter("udp_port", 50051)

        self.input_topic = self.get_parameter("input_topic").value
        self.udp_host = self.get_parameter("udp_host").value
        self.udp_port = int(self.get_parameter("udp_port").value)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0

        self.sub = self.create_subscription(
            UpperBodyCommand,
            self.input_topic,
            self.cb,
            10,
        )

        self.get_logger().info("============================================================")
        self.get_logger().info("ROS UPPER BODY COMMAND -> UDP")
        self.get_logger().info(f"input_topic: {self.input_topic}")
        self.get_logger().info(f"udp:         {self.udp_host}:{self.udp_port}")
        self.get_logger().info("============================================================")

    def cb(self, msg: UpperBodyCommand):
        if not msg.valid:
            return

        payload = {
            "seq": self.seq,
            "stamp_sec": int(msg.header.stamp.sec),
            "stamp_nanosec": int(msg.header.stamp.nanosec),
            "joint_names": list(msg.joint_names),
            "position": [float(x) for x in msg.position],
            "confidence": [float(x) for x in msg.confidence],
            "valid": bool(msg.valid),
        }

        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.sock.sendto(data, (self.udp_host, self.udp_port))

        self.seq += 1

        if self.seq % 60 == 0:
            self.get_logger().info(f"sent udp seq={self.seq} bytes={len(data)}")


def main():
    rclpy.init()
    node = UpperBodyUdpForwarder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
