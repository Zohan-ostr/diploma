#!/usr/bin/env python3

import copy
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from unitree_go.msg import LowCmd, LowState


# MuJoCo H1 actuator indices from simulator printout:
# 0  right_hip_roll_joint
# 1  right_hip_pitch_joint
# 2  right_knee_joint
# 3  left_hip_roll_joint
# 4  left_hip_pitch_joint
# 5  left_knee_joint
# 6  torso_joint
# 7  left_hip_yaw_joint
# 8  right_hip_yaw_joint
# 9  not_use_joint
# 10 left_ankle_joint
# 11 right_ankle_joint
#
# 12 right_shoulder_pitch_joint
# 13 right_shoulder_roll_joint
# 14 right_shoulder_yaw_joint
# 15 right_elbow_joint
# 16 left_shoulder_pitch_joint
# 17 left_shoulder_roll_joint
# 18 left_shoulder_yaw_joint
# 19 left_elbow_joint

LOWER_AND_TORSO_INDICES = [
    0, 1, 2,
    3, 4, 5,
    6,
    7, 8,
    10, 11,
]

UPPER_INDICES = [
    12, 13, 14, 15,
    16, 17, 18, 19,
]


class ArmSdkToLowCmdBridge(Node):
    """
    Simulation-only bridge.

    Main project interface:
        /arm_sdk

    MuJoCo backend interface:
        /lowcmd

    Why this node is needed:
        Real H1 has its own stabilizing controller.
        MuJoCo /lowcmd is low-level, so if we send only arms,
        legs and torso are not actively held and the robot starts rotating/falling.

    This bridge:
        1. listens to /lowstate;
        2. captures initial lower-body + torso q as hold pose;
        3. forwards upper-body commands from /arm_sdk;
        4. injects lower-body + torso hold commands into /lowcmd.
    """

    def __init__(self):
        super().__init__('arm_sdk_to_lowcmd_bridge')

        self.lower_kp = float(self.declare_parameter('lower_kp', 80.0).value)
        self.lower_kd = float(self.declare_parameter('lower_kd', 4.0).value)
        self.torso_kp = float(self.declare_parameter('torso_kp', 60.0).value)
        self.torso_kd = float(self.declare_parameter('torso_kd', 3.0).value)

        self.capture_hold_pose_once = bool(
            self.declare_parameter('capture_hold_pose_once', True).value
        )

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.hold_q: Optional[Dict[int, float]] = None
        self.lowstate_counter = 0
        self.bridge_counter = 0

        self.create_subscription(
            LowState,
            '/lowstate',
            self.on_lowstate,
            qos,
        )

        self.create_subscription(
            LowCmd,
            '/arm_sdk',
            self.on_arm_sdk,
            qos,
        )

        self.pub = self.create_publisher(
            LowCmd,
            '/lowcmd',
            qos,
        )

        self.get_logger().info('Stabilizing bridge started: /arm_sdk -> /lowcmd')
        self.get_logger().info('This bridge is for MuJoCo only.')
        self.get_logger().info(
            f'lower_kp={self.lower_kp}, lower_kd={self.lower_kd}, '
            f'torso_kp={self.torso_kp}, torso_kd={self.torso_kd}'
        )

    def on_lowstate(self, msg: LowState):
        self.lowstate_counter += 1

        if self.hold_q is not None and self.capture_hold_pose_once:
            return

        if not hasattr(msg, 'motor_state'):
            self.get_logger().warn('LowState has no motor_state field', throttle_duration_sec=2.0)
            return

        if len(msg.motor_state) <= max(LOWER_AND_TORSO_INDICES):
            self.get_logger().warn(
                f'LowState motor_state is too short: {len(msg.motor_state)}',
                throttle_duration_sec=2.0,
            )
            return

        hold = {}
        for idx in LOWER_AND_TORSO_INDICES:
            hold[idx] = float(msg.motor_state[idx].q)

        self.hold_q = hold

        self.get_logger().info('Captured lower-body/torso hold pose from /lowstate:')
        for idx in LOWER_AND_TORSO_INDICES:
            self.get_logger().info(f'  idx={idx:02d} q={self.hold_q[idx]:+.4f}')

    def inject_hold_commands(self, msg: LowCmd):
        if self.hold_q is None:
            return False

        if not hasattr(msg, 'motor_cmd'):
            self.get_logger().error('LowCmd has no motor_cmd field')
            return False

        if len(msg.motor_cmd) <= max(LOWER_AND_TORSO_INDICES):
            self.get_logger().error(f'LowCmd motor_cmd too short: {len(msg.motor_cmd)}')
            return False

        for idx, q in self.hold_q.items():
            mc = msg.motor_cmd[idx]

            mc.mode = 1
            mc.q = float(q)
            mc.dq = 0.0
            mc.tau = 0.0

            if idx == 6:
                mc.kp = float(self.torso_kp)
                mc.kd = float(self.torso_kd)
            else:
                mc.kp = float(self.lower_kp)
                mc.kd = float(self.lower_kd)

        return True

    def on_arm_sdk(self, msg: LowCmd):
        if self.hold_q is None:
            self.get_logger().warn(
                'No /lowstate hold pose yet; not forwarding /arm_sdk',
                throttle_duration_sec=2.0,
            )
            return

        out = copy.deepcopy(msg)

        ok = self.inject_hold_commands(out)
        if not ok:
            return

        self.pub.publish(out)

        self.bridge_counter += 1
        if self.bridge_counter % 100 == 0:
            self.get_logger().info('Published stabilized /lowcmd')


def main():
    rclpy.init()
    node = ArmSdkToLowCmdBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
