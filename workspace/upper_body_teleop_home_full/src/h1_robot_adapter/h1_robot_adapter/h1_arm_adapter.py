#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from upper_body_msgs.msg import UpperBodyCommand
from unitree_go.msg import LowCmd


# Индексы подтверждены выводом unitree_mujoco:
# Actuator_index 12..19.
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


# Таблица знаков. Если в MuJoCo или на реальном роботе движение пойдёт
# в обратную сторону, меняем знак здесь, не трогая основной IK.
ROBOT_SIGN = {
    'left_shoulder_pitch_joint': 1.0,
    'left_shoulder_roll_joint': 1.0,
    'left_shoulder_yaw_joint': 1.0,
    'left_elbow_joint': 1.0,

    'right_shoulder_pitch_joint': 1.0,
    'right_shoulder_roll_joint': 1.0,
    'right_shoulder_yaw_joint': 1.0,
    'right_elbow_joint': 1.0,
}


ROBOT_OFFSET = {
    'left_shoulder_pitch_joint': 0.0,
    'left_shoulder_roll_joint': 0.0,
    'left_shoulder_yaw_joint': 0.0,
    'left_elbow_joint': 0.0,

    'right_shoulder_pitch_joint': 0.0,
    'right_shoulder_roll_joint': 0.0,
    'right_shoulder_yaw_joint': 0.0,
    'right_elbow_joint': 0.0,
}


# Для первого теста лимиты умеренные.
# Потом можно расширить после проверки направлений.
ROBOT_LIMITS = {
    'left_shoulder_pitch_joint': (-1.2, 1.2),
    'left_shoulder_roll_joint': (-1.2, 1.2),
    'left_shoulder_yaw_joint': (-3.14, 3.14),
    'left_elbow_joint': (0.0, 2.2),

    'right_shoulder_pitch_joint': (-1.2, 1.2),
    'right_shoulder_roll_joint': (-1.2, 1.2),
    'right_shoulder_yaw_joint': (-3.14, 3.14),
    'right_elbow_joint': (0.0, 2.2),
}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class H1ArmAdapter(Node):
    def __init__(self):
        super().__init__('h1_arm_adapter')

        self.dry_run = bool(self.declare_parameter('dry_run', True).value)
        self.publish_enabled = bool(self.declare_parameter('publish_enabled', False).value)
        self.output_topic = str(self.declare_parameter('output_topic', '/arm_sdk').value)

        self.kp = float(self.declare_parameter('kp', 15.0).value)
        self.kd = float(self.declare_parameter('kd', 1.0).value)

        qos_best_effort = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(
            UpperBodyCommand,
            '/upper_body/command',
            self.on_upper_body_command,
            10,
        )

        self.pub = self.create_publisher(
            LowCmd,
            self.output_topic,
            qos_best_effort,
        )

        self.counter = 0

        self.get_logger().info('H1 arm adapter started')
        self.get_logger().info('Input : /upper_body/command')
        self.get_logger().info(f'Output: {self.output_topic}')
        self.get_logger().info(f'dry_run={self.dry_run}, publish_enabled={self.publish_enabled}')
        self.get_logger().info(f'kp={self.kp}, kd={self.kd}')

    def convert_upper_body(self, msg: UpperBodyCommand):
        result = {}

        for name, value in zip(msg.joint_names, msg.position):
            if name not in H1_ARM_INDEX:
                continue

            q = ROBOT_SIGN[name] * float(value) + ROBOT_OFFSET[name]
            lo, hi = ROBOT_LIMITS[name]
            q = clamp(q, lo, hi)

            result[name] = q

        return result

    def make_lowcmd(self, arm_cmd):
        low = LowCmd()

        # Важно: не трогаем ноги. Для всех неиспользуемых моторов mode остаётся 0.
        # Активируем только 8 актуаторов верхней части тела.
        for joint_name, q in arm_cmd.items():
            idx = H1_ARM_INDEX[joint_name]

            if idx >= len(low.motor_cmd):
                self.get_logger().error(
                    f'Index {idx} is out of motor_cmd range {len(low.motor_cmd)}'
                )
                continue

            mc = low.motor_cmd[idx]

            mc.mode = 1
            mc.q = float(q)
            mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = float(self.kp)
            mc.kd = float(self.kd)

        return low

    def on_upper_body_command(self, msg: UpperBodyCommand):
        if not msg.valid:
            self.get_logger().warn('Invalid /upper_body/command, ignoring', throttle_duration_sec=2.0)
            return

        arm_cmd = self.convert_upper_body(msg)
        low = self.make_lowcmd(arm_cmd)

        self.counter += 1

        if self.counter % 20 == 0:
            print()
            print('===== H1 ARM ADAPTER =====')
            for name in [
                'right_shoulder_pitch_joint',
                'right_shoulder_roll_joint',
                'right_shoulder_yaw_joint',
                'right_elbow_joint',
                'left_shoulder_pitch_joint',
                'left_shoulder_roll_joint',
                'left_shoulder_yaw_joint',
                'left_elbow_joint',
            ]:
                if name in arm_cmd:
                    print(f'{name:35s} idx={H1_ARM_INDEX[name]:02d} q={arm_cmd[name]:+.4f}')

        if self.dry_run or not self.publish_enabled:
            return

        self.pub.publish(low)


def main():
    rclpy.init()
    node = H1ArmAdapter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
