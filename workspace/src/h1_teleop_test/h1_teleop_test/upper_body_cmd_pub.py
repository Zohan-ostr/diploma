import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class UpperBodyCmdPub(Node):
    def __init__(self):
        super().__init__('upper_body_cmd_pub')
        self.pub = self.create_publisher(JointState, '/joint_states', 10)
        self.t = 0.0
        self.timer = self.create_timer(0.05, self.tick)

        self.names = [
            'torso_joint',
            'left_shoulder_pitch_joint',
            'left_shoulder_roll_joint',
            'left_shoulder_yaw_joint',
            'left_elbow_joint',
            'right_shoulder_pitch_joint',
            'right_shoulder_roll_joint',
            'right_shoulder_yaw_joint',
            'right_elbow_joint',
        ]

    def tick(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.names

        self.t += 0.05
        msg.position = [
            0.20 * math.sin(self.t),
            0.40 * math.sin(self.t),
            0.30 * math.sin(self.t * 0.7),
            0.20 * math.cos(self.t),
            0.50 + 0.30 * math.sin(self.t),
            0.40 * math.sin(self.t),
            -0.30 * math.sin(self.t * 0.7),
            -0.20 * math.cos(self.t),
            0.50 + 0.30 * math.sin(self.t),
        ]
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = UpperBodyCmdPub()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
