#!/usr/bin/env python3
import json
import socket
import time

import rclpy
from rclpy.node import Node

from upper_body_msgs.msg import PoseLandmarks3D


class UdpPoseLandmarksToRos(Node):
    def __init__(self):
        super().__init__("udp_pose_landmarks_to_ros")

        self.declare_parameter("udp_host", "0.0.0.0")
        self.declare_parameter("udp_port", 50060)
        self.declare_parameter("output_topic", "/pose/landmarks")
        self.declare_parameter("frame_id", "realsense_color_optical_frame")

        self.udp_host = self.get_parameter("udp_host").value
        self.udp_port = int(self.get_parameter("udp_port").value)
        self.output_topic = self.get_parameter("output_topic").value
        self.frame_id = self.get_parameter("frame_id").value

        self.pub = self.create_publisher(PoseLandmarks3D, self.output_topic, 10)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.udp_host, self.udp_port))
        self.sock.setblocking(False)

        self.seq = 0
        self.last_rx = time.time()

        self.timer = self.create_timer(1.0 / 120.0, self.on_timer)

        self.get_logger().info("============================================================")
        self.get_logger().info("UDP POSE LANDMARKS -> ROS")
        self.get_logger().info(f"udp:          {self.udp_host}:{self.udp_port}")
        self.get_logger().info(f"output_topic: {self.output_topic}")
        self.get_logger().info("============================================================")

    def on_timer(self):
        got = 0

        while True:
            try:
                data, _addr = self.sock.recvfrom(65535)
            except BlockingIOError:
                break

            got += 1

            try:
                payload = json.loads(data.decode("utf-8"))
            except Exception as e:
                self.get_logger().warn(f"bad udp json: {e}")
                continue

            names = payload.get("names", [])
            xs = payload.get("x", [])
            ys = payload.get("y", [])
            zs = payload.get("z", [])
            visibility = payload.get("visibility", [])
            valid = bool(payload.get("valid", False))

            if not (len(names) == len(xs) == len(ys) == len(zs)):
                self.get_logger().warn("bad landmark array sizes")
                continue

            msg = PoseLandmarks3D()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id

            msg.names = [str(v) for v in names]
            msg.x = [float(v) for v in xs]
            msg.y = [float(v) for v in ys]
            msg.z = [float(v) for v in zs]

            if len(visibility) == len(names):
                msg.visibility = [float(v) for v in visibility]
            else:
                msg.visibility = [1.0 for _ in names]

            msg.valid = valid

            self.pub.publish(msg)

            self.seq += 1
            self.last_rx = time.time()

        if got and self.seq % 60 == 0:
            self.get_logger().info(f"published pose landmarks seq={self.seq}")


def main():
    rclpy.init()
    node = UdpPoseLandmarksToRos()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
