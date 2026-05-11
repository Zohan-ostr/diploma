#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from unitree_go.msg import LowCmd, LowState


H1_ARM_INDEX = {
    'right_shoulder_pitch_joint': 12,
    'right_shoulder_roll_joint': 13,
    'right_shoulder_yaw_joint': 14,
    'right_elbow_joint': 15,

    'left_shoulder_pitch_joint': 16,
    'left_shoulder_roll_joint': 17,
    'left_shoulder_yaw_joint': 18,
    'left_elbow_joint': 19,
}

ARM_JOINTS = list(H1_ARM_INDEX.keys())


class H1ArmJointTest(Node):
    def __init__(self):
        super().__init__('h1_arm_joint_test')

        self.joint = str(self.declare_parameter('joint', 'right_shoulder_pitch_joint').value)
        self.delta = float(self.declare_parameter('delta', 0.03).value)
        self.duration = float(self.declare_parameter('duration', 3.0).value)
        self.kp = float(self.declare_parameter('kp', 3.0).value)
        self.kd = float(self.declare_parameter('kd', 0.15).value)
        self.dry_run = bool(self.declare_parameter('dry_run', True).value)

        if self.joint not in H1_ARM_INDEX:
            raise RuntimeError(f'Unknown joint: {self.joint}. Allowed: {ARM_JOINTS}')

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.last_lowstate = None
        self.baseline = None
        self.start_time = None
        self.finished = False

        self.create_subscription(LowState, '/lowstate', self.on_lowstate, qos)
        self.create_subscription(LowState, '/lf/lowstate', self.on_lowstate, qos)

        self.pub = self.create_publisher(LowCmd, '/arm_sdk', qos)
        self.timer = self.create_timer(0.02, self.tick)

        self.get_logger().info('H1 single-joint arm test started.')
        self.get_logger().info(f'joint={self.joint}, delta={self.delta}, duration={self.duration}')
        self.get_logger().info(f'kp={self.kp}, kd={self.kd}, dry_run={self.dry_run}')
        self.get_logger().info('Waiting for /lowstate or /lf/lowstate...')

    def on_lowstate(self, msg):
        self.last_lowstate = msg

        if self.baseline is None:
            self.baseline = {}
            for name, idx in H1_ARM_INDEX.items():
                self.baseline[name] = float(msg.motor_state[idx].q)

            self.start_time = self.get_clock().now().nanoseconds * 1e-9

            self.get_logger().info('Captured baseline arm pose:')
            for name in ARM_JOINTS:
                self.get_logger().info(f'  {name:35s} idx={H1_ARM_INDEX[name]:02d} q={self.baseline[name]:+.4f}')

    def make_cmd(self, t):
        cmd = LowCmd()

        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        cmd.gpio = 0

        for i in range(len(cmd.motor_cmd)):
            cmd.motor_cmd[i].mode = 0x00
            cmd.motor_cmd[i].q = 0.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].tau = 0.0
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].kd = 0.0

        # Плавный профиль: 0 -> delta -> 0
        half = self.duration / 2.0
        if t <= half:
            phase = t / half
        else:
            phase = max(0.0, 1.0 - (t - half) / half)

        # smoothstep
        phase = phase * phase * (3.0 - 2.0 * phase)

        for name, idx in H1_ARM_INDEX.items():
            q = self.baseline[name]

            if name == self.joint:
                q = q + self.delta * phase

            mc = cmd.motor_cmd[idx]
            mc.mode = 0x01
            mc.q = float(q)
            mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = float(self.kp)
            mc.kd = float(self.kd)

        cmd.crc = 0
        return cmd, phase

    def tick(self):
        if self.finished:
            return

        if self.baseline is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        t = now - self.start_time

        if t > self.duration:
            self.finished = True
            self.get_logger().info('Test finished. Command returned to baseline.')
            return

        cmd, phase = self.make_cmd(t)

        target = self.baseline[self.joint] + self.delta * phase

        if int(t * 10) % 5 == 0:
            self.get_logger().info(
                f't={t:.2f}s phase={phase:.2f} {self.joint}: '
                f'base={self.baseline[self.joint]:+.4f} target={target:+.4f}'
            )

        if not self.dry_run:
            self.pub.publish(cmd)


def main():
    rclpy.init()
    node = H1ArmJointTest()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
