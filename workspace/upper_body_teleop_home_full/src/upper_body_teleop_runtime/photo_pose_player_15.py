#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PHOTO POSE PLAYER FOR UNITREE H1

Режим для демонстрационных фото:
  - не использует камеру;
  - не использует MediaPipe;
  - не использует retarget;
  - публикует одну из 15 заранее заданных поз рук;
  - переключение клавишами [ и ].

Управляются только 8 суставов рук.
Torso не публикуется.
"""

import math
import select
import sys
import termios
import tty
from dataclasses import dataclass
from typing import List

import rclpy
from rclpy.node import Node

from upper_body_msgs.msg import UpperBodyCommand


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


@dataclass
class Pose:
    name: str
    description: str
    q: List[float]


class PhotoPosePlayer15(Node):
    JOINT_NAMES = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
    ]

    # Ограничения оставлены широкими, но команды в таблице ниже подобраны умеренно.
    LOWER = [
        -2.87, -0.34, -1.30, -1.25,
        -2.87, -3.11, -4.45, -1.25,
    ]

    UPPER = [
         2.87,  3.11,  4.45,  2.61,
         2.87,  0.34,  1.30,  2.61,
    ]

    def __init__(self):
        super().__init__("photo_pose_player_15")

        self.declare_parameter("output_topic", "/upper_body/command_geom")
        self.declare_parameter("publish_hz", 50.0)
        self.declare_parameter("max_step_rad", 0.018)

        self.output_topic = str(self.get_parameter("output_topic").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.max_step_rad = float(self.get_parameter("max_step_rad").value)

        self.poses = self.make_poses()
        self.index = 0

        # Тест локтей:
        # каждое нажатие "e" ставит руки в T-позу и меняет только elbow.
        self.elbow_test_values = [1.57, 1.00, 0.50, 0.00]
        self.elbow_test_index = 0
        self.custom_frame_id = ""

        self.current_q = list(self.poses[0].q)
        self.target_q = list(self.poses[0].q)

        self.pub = self.create_publisher(UpperBodyCommand, self.output_topic, 10)
        self.timer = self.create_timer(1.0 / self.publish_hz, self.on_timer)

        self.old_terminal_settings = None
        self.setup_keyboard()

        self.print_help()
        self.print_pose()

        self.get_logger().info("============================================================")
        self.get_logger().info("PHOTO POSE PLAYER 15")
        self.get_logger().info(f"output_topic: {self.output_topic}")
        self.get_logger().info("keys: [ previous, ] next, r T-pose, q quit")
        self.get_logger().info("============================================================")

    def setup_keyboard(self):
        if sys.stdin.isatty():
            self.old_terminal_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

    def restore_keyboard(self):
        if self.old_terminal_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_terminal_settings)
            self.old_terminal_settings = None

    def make_poses(self) -> List[Pose]:
        E = 1.57

        # Формат:
        # [L_pitch, L_roll, L_yaw, L_elbow,
        #  R_pitch, R_roll, R_yaw, R_elbow]
        #
        # Фактическая настройка после проверки на реальном H1:
        #   roll  ≈ 0       -> руки вниз;
        #   roll  ≈ ±1.57   -> руки в стороны;
        #   roll  ≈ ±2.00   -> руки примерно 45° вверх;
        #   roll  ≈ ±3.05   -> руки максимально вверх;
        #   pitch < 0       -> руки вперёд;
        #   elbow < 0       -> более явный сгиб локтя.

        return [
            Pose(
                "01 T-поза",
                "Обе руки в стороны, локти прямые",
                [-0.10,  1.57,  1.74, E,
                 -0.10, -1.57, -1.74, E],
            ),

            Pose(
                "02 Руки вниз",
                "Обе руки опущены вдоль корпуса",
                [-0.10,  0.08, -1.30, E,
                 -0.10, -0.08,  1.30, E],
            ),

            Pose(
                "03 Руки вперёд",
                "Обе руки вытянуты перед корпусом",
                [-1.35,  0.12,  0.00, E,
                 -1.35, -0.12,  0.00, E],
            ),

            Pose(
                "04 Руки вверх",
                "Обе руки максимально подняты вверх через shoulder_roll",
                [-0.10,  3.05,  0.00, E,
                 -0.10, -3.05,  0.00, E],
            ),

            Pose(
                "05 Руки 45° вверх-в стороны",
                "Обе руки диагонально вверх-в стороны, ниже полной верхней позы",
                [-0.10,  2.00,  0.65, E,
                 -0.10, -2.00, -0.65, E],
            ),

            Pose(
                "06 Руки 45° вниз-в стороны",
                "Обе руки диагонально вниз-в стороны",
                [-0.10,  0.75, -1.25, E,
                 -0.10, -0.75,  0.70, E],
            ),

            Pose(
                "07 Левая вверх, правая вниз",
                "Асимметричная поза для сравнения сторон",
                [-0.10,  3.05,  0.00, E,
                 -0.10, -0.08,  1.30, E],
            ),

            Pose(
                "08 Правая вверх, левая вниз",
                "Асимметричная поза для сравнения сторон",
                [-0.10,  0.08, -1.30, E,
                 -0.10, -3.05,  0.00, E],
            ),

            Pose(
                "09 Левая в сторону, правая вперёд",
                "Проверка различия side/forward",
                [-0.10,  1.57,  1.74, E,
                 -1.35, -0.12,  0.00, E],
            ),

            Pose(
                "10 Правая в сторону, левая вперёд",
                "Проверка различия side/forward",
                [-1.35,  0.12,  0.00, E,
                 -0.10, -1.57, -1.74, E],
            ),

            Pose(
                "11 Левая согнута, правая прямая",
                "Показательный сильный сгиб левого локтя",
                [-1.10,  0.45,  0.30, -0.55,
                 -0.10, -1.57, -1.74, E],
            ),

            Pose(
                "12 Правая согнута, левая прямая",
                "Показательный сильный сгиб правого локтя",
                [-0.10,  1.57,  1.74, E,
                 -1.10, -1.25, -0.30, -0.55],
            ),

            Pose(
                "13 Обе руки согнуты перед собой",
                "Обе руки перед корпусом, локти сильно согнуты",
                [-1.10,  0.35,  0.30, -1.25,
                 -1.10, -0.35, -0.30, -0.70],
            ),

            Pose(
                "14 Одна рука вверх, другая в сторону",
                "Показательная асимметрия для фото",
                [-0.10,  3.00,  0.20, E,
                 -0.10, -1.57, -1.74, E],
            ),

            Pose(
                "15 Финальная симметричная",
                "Красивая симметричная поза: руки 45° вверх, локти заметно согнуты",
                [-0.35,  2.00,  0.55, -1.25,
                 -0.35, -2.00, -1.25, -0.45],
            ),
        ]


    def print_help(self):
        print()
        print("============================================================")
        print(" PHOTO POSE PLAYER 15")
        print("============================================================")
        print("]  следующая поза")
        print("[  предыдущая поза")
        print("r  T-поза")
        print("e  тест локтя: T-поза + следующий elbow")
        print("q  выход")
        print("============================================================")
        print()

    def print_pose(self):
        pose = self.poses[self.index]
        print()
        print("------------------------------------------------------------")
        print(f"POSE {self.index + 1:02d}/{len(self.poses)}: {pose.name}")
        print(pose.description)
        print("q:")
        print("  left :  pitch={:+.3f}, roll={:+.3f}, yaw={:+.3f}, elbow={:+.3f}".format(*pose.q[0:4]))
        print("  right:  pitch={:+.3f}, roll={:+.3f}, yaw={:+.3f}, elbow={:+.3f}".format(*pose.q[4:8]))
        print("------------------------------------------------------------")
        print()

    def set_pose(self, idx: int):
        self.custom_frame_id = ""
        self.index = idx % len(self.poses)
        pose = self.poses[self.index]
        self.target_q = [
            clamp(float(v), self.LOWER[i], self.UPPER[i])
            for i, v in enumerate(pose.q)
        ]
        self.print_pose()

    def set_elbow_test_pose(self):
        """
        T-поза + последовательная проверка сгиба локтей.

        Значения:
          1.57 — прямая рука;
          1.00 — слабый сгиб;
          0.50 — средний сгиб;
          0.00 — примерно 90 градусов, если маппинг локтя правильный.
        """
        elbow_q = float(self.elbow_test_values[self.elbow_test_index])

        q = list(self.poses[0].q)  # T-поза
        q[3] = elbow_q             # left_elbow_joint
        q[7] = elbow_q             # right_elbow_joint

        self.target_q = [
            clamp(float(v), self.LOWER[i], self.UPPER[i])
            for i, v in enumerate(q)
        ]

        self.custom_frame_id = f"elbow_test_{self.elbow_test_index}_{elbow_q:.2f}".replace(".", "_")

        print()
        print("------------------------------------------------------------")
        print("ELBOW TEST")
        print("T-поза, меняются только локти")
        print(f"left_elbow_joint  = {elbow_q:+.3f}")
        print(f"right_elbow_joint = {elbow_q:+.3f}")
        print("Последовательность: 1.57 -> 1.00 -> 0.50 -> 0.00")
        print("------------------------------------------------------------")
        print()

        self.elbow_test_index = (self.elbow_test_index + 1) % len(self.elbow_test_values)


    def read_key(self):
        if not sys.stdin.isatty():
            return None

        r, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not r:
            return None

        return sys.stdin.read(1)

    def handle_key(self, key):
        if key == "]":
            self.set_pose(self.index + 1)
        elif key == "[":
            self.set_pose(self.index - 1)
        elif key in ("r", "R"):
            self.set_pose(0)
        elif key in ("e", "E"):
            self.set_elbow_test_pose()
        elif key in ("q", "Q"):
            print("Exiting photo pose player...")
            rclpy.shutdown()

    def step_to_target(self):
        for i in range(len(self.current_q)):
            delta = self.target_q[i] - self.current_q[i]
            delta = clamp(delta, -self.max_step_rad, self.max_step_rad)
            self.current_q[i] = clamp(
                self.current_q[i] + delta,
                self.LOWER[i],
                self.UPPER[i],
            )

    def publish_current(self):
        msg = UpperBodyCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.custom_frame_id if self.custom_frame_id else f"photo_pose_{self.index + 1:02d}"
        msg.joint_names = list(self.JOINT_NAMES)
        msg.position = [float(v) for v in self.current_q]
        msg.confidence = [1.0 for _ in self.current_q]
        msg.valid = True
        self.pub.publish(msg)

    def on_timer(self):
        key = self.read_key()
        if key is not None:
            self.handle_key(key)

        self.step_to_target()
        self.publish_current()

    def destroy_node(self):
        self.restore_keyboard()
        super().destroy_node()


def main():
    rclpy.init()
    node = PhotoPosePlayer15()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
