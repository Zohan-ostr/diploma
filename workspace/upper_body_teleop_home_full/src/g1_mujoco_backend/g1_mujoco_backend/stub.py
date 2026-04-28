import rclpy
from rclpy.node import Node

class Stub(Node):
    def __init__(self):
        super().__init__('g1_mujoco_backend_stub')
        self.get_logger().warn('G1 MuJoCo backend is a placeholder. Home mode does not use simulation.')

def main():
    rclpy.init()
    node = Stub()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
