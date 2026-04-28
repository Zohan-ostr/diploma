import math
import rclpy
from rclpy.node import Node
from upper_body_msgs.msg import PoseLandmarks3D

LANDMARKS = [
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip'
]

class MockPoseSource(Node):
    def __init__(self):
        super().__init__('mock_pose_source')
        self.pub = self.create_publisher(PoseLandmarks3D, '/pose/landmarks', 10)
        self.t = 0.0
        self.timer = self.create_timer(1.0 / 30.0, self.tick)
        self.get_logger().info('Mock pose source started. This mode does not use webcam.')

    def tick(self):
        self.t += 1.0 / 30.0
        s = math.sin(self.t)
        c = math.cos(self.t * 0.8)

        pts = {
            'left_shoulder':  (-0.20,  0.25, 1.55),
            'right_shoulder': (-0.20, -0.25, 1.55),
            'left_hip':       (0.00,  0.12, 1.05),
            'right_hip':      (0.00, -0.12, 1.05),
            'left_elbow':     (-0.10 + 0.10*s,  0.48, 1.35 + 0.18*c),
            'right_elbow':    (-0.10 + 0.10*s, -0.48, 1.35 + 0.18*c),
            'left_wrist':     (0.05 + 0.20*s,   0.65, 1.25 + 0.30*c),
            'right_wrist':    (0.05 + 0.20*s,  -0.65, 1.25 + 0.30*c),
        }

        msg = PoseLandmarks3D()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        msg.names = LANDMARKS
        msg.x = [pts[n][0] for n in LANDMARKS]
        msg.y = [pts[n][1] for n in LANDMARKS]
        msg.z = [pts[n][2] for n in LANDMARKS]
        msg.visibility = [1.0] * len(LANDMARKS)
        msg.valid = True
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = MockPoseSource()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
