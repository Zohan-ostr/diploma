#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DIRECT SDK2 PHOTO POSE PLAYER FOR UNITREE H1

Без ROS sender.
Без /upper_body/command_geom.
Команды отправляются напрямую через unitree_sdk2py в rt/lowcmd.

Клавиши:
  ]  следующая поза
  [  предыдущая поза
  r  T-поза
  e  тест локтя: перебор значений elbow
  q  выход
"""

import os
import sys
import time
import math
import select
import termios
import tty
from dataclasses import dataclass
from typing import List, Optional

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_

try:
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
except Exception:
    unitree_go_msg_dds__LowCmd_ = None

try:
    from unitree_sdk2py.utils.crc import CRC
except Exception:
    CRC = None


def clamp(x, lo, hi):
    return max(lo, min(hi, float(x)))


@dataclass
class Pose:
    name: str
    description: str
    q: List[float]


class SDK2PhotoPosePlayer:
    # Порядок q:
    # left_pitch, left_roll, left_yaw, left_elbow,
    # right_pitch, right_roll, right_yaw, right_elbow
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

    # motor id H1:
    # right arm: 12,13,14,15
    # left arm:  16,17,18,19
    ARM_MOTOR_IDS = [16, 17, 18, 19, 12, 13, 14, 15]

    LEFT_ELBOW_ID = 19
    RIGHT_ELBOW_ID = 15

    # Для фото-поз оставляем умеренные ограничения, но локоть даём тестировать шире.
    LOWER = [-2.87, -0.34, -1.30, -1.25, -2.87, -3.11, -4.45, -1.25]
    UPPER = [ 2.87,  3.11,  4.45,  2.61,  2.87,  0.34,  1.30,  2.61]

    def __init__(self):
        self.iface = os.environ.get("UNITREE_NET_IFACE", "enx00e04c36022c")
        self.domain = int(os.environ.get("UNITREE_DOMAIN_ID", "0"))

        self.kp_arm = 60.0
        self.kd_arm = 1.5
        self.hz = 250.0
        self.max_step = 0.018

        self.lowstate = None
        self.lowstate_count = 0

        self.pub = None
        self.crc = CRC() if CRC is not None else None

        self.poses = self.make_poses()
        self.index = 0

        self.current_q = list(self.poses[0].q)
        self.target_q = list(self.poses[0].q)

        # Тест локтя. Если 0.0 всё ещё слабовато, дальше проверяем отрицательные значения.
        self.elbow_test_values = [1.57, 1.20, 1.00, 0.80, 0.50, 0.20, 0.00, -0.30, -0.60, -0.90, -1.15, 1.57]
        self.elbow_test_index = 0

        self.old_terminal_settings = None
        self.running = True

    def make_poses(self):
        E = 1.57

        return [
            Pose("01 T-поза", "Обе руки в стороны, локти прямые",
                 [-0.10,  1.57,  1.74, E, -0.10, -1.57, -1.74, E]),

            Pose("02 Руки вниз", "Обе руки опущены вдоль корпуса",
                 [-0.10,  0.08, -1.30, E, -0.10, -0.08,  1.30, E]),

            Pose("03 Руки вперёд", "Обе руки вытянуты перед корпусом",
                 [-1.35,  0.12,  0.00, E, -1.35, -0.12,  0.00, E]),

            Pose("04 Руки вверх", "Обе руки максимально подняты вверх через shoulder_roll",
                 [-0.10,  3.05,  0.00, E, -0.10, -3.05,  0.00, E]),

            Pose("05 Руки 45° вверх-в стороны", "Диагонально вверх-в стороны",
                 [-0.10,  2.00,  0.65, E, -0.10, -2.00, -0.65, E]),

            Pose("06 Руки 45° вниз-в стороны", "Диагонально вниз-в стороны",
                 [-0.10,  0.75, -0.70, E, -0.10, -0.75,  0.70, E]),

            Pose("07 Левая вверх, правая вниз", "Асимметричная поза",
                 [-0.10,  3.05,  0.00, E, -0.10, -0.08,  1.30, E]),

            Pose("08 Правая вверх, левая вниз", "Асимметричная поза",
                 [-0.10,  0.08, -1.30, E, -0.10, -3.05,  0.00, E]),

            Pose("09 Левая в сторону, правая вперёд", "Side/forward",
                 [-0.10,  1.57,  1.74, E, -1.35, -0.12,  0.00, E]),

            Pose("10 Правая в сторону, левая вперёд", "Side/forward",
                 [-1.35,  0.12,  0.00, E, -0.10, -1.57, -1.74, E]),

            Pose("11 Левая согнута, правая прямая", "Сгиб левого локтя",
                 [-1.10,  0.45,  0.30, 0.00, -0.10, -1.57, -1.74, E]),

            Pose("12 Правая согнута, левая прямая", "Сгиб правого локтя",
                 [-0.10,  1.57,  1.74, E, -1.10, -0.45, -0.30, 0.00]),

            Pose("13 Обе руки согнуты перед собой", "Обе руки перед корпусом, локти около 90°",
                 [-1.10,  0.35,  0.30, 0.00, -1.10, -0.35, -0.30, 0.00]),

            Pose("14 Одна рука вверх, другая в сторону", "Асимметрия для фото",
                 [-0.10,  3.00,  0.20, E, -0.10, -1.57, -1.74, E]),

            Pose("15 Финальная симметричная", "Руки 45° вверх, локти согнуты",
                 [-0.35,  2.00,  0.55, 0.00, -0.35, -2.00, -0.55, 0.00]),
        ]

    def motor_states(self):
        if self.lowstate is None:
            return None
        for name in ("motor_state", "motor_state_"):
            if hasattr(self.lowstate, name):
                return getattr(self.lowstate, name)
        return None

    def motor_cmds(self, cmd):
        for name in ("motor_cmd", "motor_cmd_"):
            if hasattr(cmd, name):
                return getattr(cmd, name)
        return None

    def lowstate_cb(self, msg):
        self.lowstate = msg
        self.lowstate_count += 1

    def setup_keyboard(self):
        if sys.stdin.isatty():
            self.old_terminal_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

    def restore_keyboard(self):
        if self.old_terminal_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_terminal_settings)
            self.old_terminal_settings = None

    def read_key(self):
        if not sys.stdin.isatty():
            return None
        r, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not r:
            return None
        return sys.stdin.read(1)

    def print_help(self):
        print()
        print("============================================================")
        print(" DIRECT SDK2 PHOTO POSE PLAYER 15")
        print("============================================================")
        print("]  следующая поза")
        print("[  предыдущая поза")
        print("r  T-поза")
        print("e  тест локтя: T-поза + следующий elbow")
        print("q  выход")
        print("------------------------------------------------------------")
        print(f"SDK2 domain: {self.domain}")
        print(f"iface:       {self.iface}")
        print(f"kp/kd:       {self.kp_arm} / {self.kd_arm}")
        print("motor ids:   L=[16,17,18,19], R=[12,13,14,15]")
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

    def set_pose(self, idx):
        self.index = idx % len(self.poses)
        pose = self.poses[self.index]
        self.target_q = [
            clamp(v, self.LOWER[i], self.UPPER[i])
            for i, v in enumerate(pose.q)
        ]
        self.print_pose()

    def set_elbow_test_pose(self):
        elbow_q = float(self.elbow_test_values[self.elbow_test_index])

        q = list(self.poses[0].q)  # T-поза
        q[3] = elbow_q
        q[7] = elbow_q

        self.target_q = [
            clamp(v, self.LOWER[i], self.UPPER[i])
            for i, v in enumerate(q)
        ]

        print()
        print("------------------------------------------------------------")
        print("ELBOW SDK2 TEST")
        print("T-поза, меняются только локти напрямую через rt/lowcmd")
        print(f"left_elbow_joint  id19 = {elbow_q:+.3f}")
        print(f"right_elbow_joint id15 = {elbow_q:+.3f}")
        print("sequence:", " -> ".join(f"{v:+.2f}" for v in self.elbow_test_values))
        print("------------------------------------------------------------")
        print()

        self.elbow_test_index = (self.elbow_test_index + 1) % len(self.elbow_test_values)

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
            self.running = False

    def step_to_target(self):
        for i in range(len(self.current_q)):
            d = self.target_q[i] - self.current_q[i]
            self.current_q[i] += clamp(d, -self.max_step, self.max_step)
            self.current_q[i] = clamp(self.current_q[i], self.LOWER[i], self.UPPER[i])

    def make_lowcmd(self):
        if unitree_go_msg_dds__LowCmd_ is None:
            raise RuntimeError(
                "unitree_go_msg_dds__LowCmd_ not found. "
                "Cannot create default LowCmd for unitree_go IDL."
            )
        return unitree_go_msg_dds__LowCmd_()

    def init_cmd_base(self, cmd):
        # Поля у разных IDL могут немного отличаться, поэтому всё через hasattr.
        if hasattr(cmd, "head") and len(cmd.head) >= 2:
            cmd.head[0] = 0xFE
            cmd.head[1] = 0xEF

        if hasattr(cmd, "level_flag"):
            cmd.level_flag = 0xFF

        if hasattr(cmd, "gpio"):
            cmd.gpio = 0

    def fill_cmd_from_lowstate(self, cmd):
        states = self.motor_states()
        motor_cmd = self.motor_cmds(cmd)

        if states is None or motor_cmd is None:
            raise RuntimeError("Cannot access motor_state or motor_cmd fields")

        n = min(len(states), len(motor_cmd))

        # Для неуправляемых моторов ничего активно не задаём: kp=0, kd=0, tau=0.
        # q берём текущий, чтобы структура была заполнена.
        for i in range(n):
            ms = states[i]
            mc = motor_cmd[i]

            if hasattr(mc, "mode"):
                mc.mode = 0x01

            if hasattr(mc, "q"):
                mc.q = float(getattr(ms, "q", 0.0))
            if hasattr(mc, "dq"):
                mc.dq = 0.0
            if hasattr(mc, "kp"):
                mc.kp = 0.0
            if hasattr(mc, "kd"):
                mc.kd = 0.0
            if hasattr(mc, "tau"):
                mc.tau = 0.0

        # Руки управляем напрямую.
        for q_idx, motor_id in enumerate(self.ARM_MOTOR_IDS):
            if motor_id >= n:
                continue

            mc = motor_cmd[motor_id]
            q = float(self.current_q[q_idx])

            if hasattr(mc, "mode"):
                mc.mode = 0x01
            if hasattr(mc, "q"):
                mc.q = q
            if hasattr(mc, "dq"):
                mc.dq = 0.0
            if hasattr(mc, "kp"):
                mc.kp = self.kp_arm
            if hasattr(mc, "kd"):
                mc.kd = self.kd_arm
            if hasattr(mc, "tau"):
                mc.tau = 0.0

        if self.crc is not None and hasattr(cmd, "crc"):
            cmd.crc = self.crc.Crc(cmd)

    def publish_lowcmd(self):
        cmd = self.make_lowcmd()
        self.init_cmd_base(cmd)
        self.fill_cmd_from_lowstate(cmd)
        self.pub.Write(cmd)

    def connect(self):
        print(f"Initializing SDK2: domain={self.domain}, iface={self.iface}")
        ChannelFactoryInitialize(self.domain, self.iface)

        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(self.lowstate_cb, 10)

        print("Waiting rt/lowstate...")
        t0 = time.time()
        while self.lowstate is None and time.time() - t0 < 5.0:
            print(".", end="", flush=True)
            time.sleep(0.1)
        print()

        if self.lowstate is None:
            raise RuntimeError("No rt/lowstate received")

        states = self.motor_states()
        if states is None:
            raise RuntimeError("lowstate has no motor_state field")

        print("rt/lowstate OK")
        print("initial elbows:",
              f"right id15={float(states[self.RIGHT_ELBOW_ID].q):+.3f}",
              f"left id19={float(states[self.LEFT_ELBOW_ID].q):+.3f}")

        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()

    def run(self):
        self.connect()
        self.setup_keyboard()
        self.print_help()
        self.print_pose()

        period = 1.0 / self.hz
        next_log = time.time() + 1.0

        try:
            while self.running:
                key = self.read_key()
                if key is not None:
                    self.handle_key(key)

                self.step_to_target()
                self.publish_lowcmd()

                now = time.time()
                if now >= next_log:
                    print(
                        f"sent elbows: right id15={self.current_q[7]:+.3f}, "
                        f"left id19={self.current_q[3]:+.3f}"
                    )
                    next_log = now + 1.0

                time.sleep(period)
        finally:
            print("Returning to T-pose target before exit...")
            self.target_q = list(self.poses[0].q)
            for _ in range(int(self.hz * 1.5)):
                self.step_to_target()
                self.publish_lowcmd()
                time.sleep(period)
            self.restore_keyboard()


def main():
    player = SDK2PhotoPosePlayer()
    player.run()


if __name__ == "__main__":
    main()
