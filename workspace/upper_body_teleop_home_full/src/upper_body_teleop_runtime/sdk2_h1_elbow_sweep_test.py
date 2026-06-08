#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SDK2 DIRECT H1 ELBOW SWEEP TEST

Тестирует только локти H1 напрямую через Unitree SDK2 Python:
  right_elbow_joint -> motor id 15
  left_elbow_joint  -> motor id 19

Без ROS.
Без /upper_body/command_geom.
Без h1_sdk2py_upper_body_sender.

Цель:
  понять, реально ли low-level SDK даёт согнуть локоть ниже 1.0.
"""

import os
import time
import math

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_

try:
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
except Exception:
    unitree_go_msg_dds__LowCmd_ = None

try:
    from unitree_sdk2py.utils.crc import CRC
except Exception:
    CRC = None


RIGHT_ELBOW_ID = 15
LEFT_ELBOW_ID = 19

# Руки H1:
# right: 12,13,14,15
# left:  16,17,18,19
ARM_IDS = [12, 13, 14, 15, 16, 17, 18, 19]

# По фактической проверке H1 работает через unitree_go idl.
DOMAIN = int(os.environ.get("UNITREE_DOMAIN_ID", "0"))
IFACE = os.environ.get("UNITREE_NET_IFACE", "enx00e04c36022c")

KP_ARM = float(os.environ.get("KP_ARM", "60.0"))
KD_ARM = float(os.environ.get("KD_ARM", "1.5"))

HZ = 250.0
DT = 1.0 / HZ

# Медленный шаг, чтобы не дёргать локти резко.
MAX_STEP_RAD = float(os.environ.get("MAX_STEP_RAD", "0.004"))

# Проверяем не только до 0.0, но и ниже,
# чтобы понять, есть ли реальная прошивочная/режимная граница около 1.0.
ELBOW_TARGETS = [
    1.57,
    1.30,
    1.15,
    1.00,
    0.85,
    0.70,
    0.50,
    0.30,
    0.15,
    0.00,
    -0.20,
    -0.50,
    -0.80,
    -1.10,
    1.57,
]


class ElbowSweepTest:
    def __init__(self):
        self.lowstate = None
        self.lowstate_count = 0
        self.pub = None
        self.crc = CRC() if CRC is not None else None

        self.current_left_elbow_cmd = 1.57
        self.current_right_elbow_cmd = 1.57

    def lowstate_cb(self, msg):
        self.lowstate = msg
        self.lowstate_count += 1

    def motor_states(self):
        if self.lowstate is None:
            return None
        for field in ("motor_state", "motor_state_"):
            if hasattr(self.lowstate, field):
                return getattr(self.lowstate, field)
        return None

    def motor_cmds(self, cmd):
        for field in ("motor_cmd", "motor_cmd_"):
            if hasattr(cmd, field):
                return getattr(cmd, field)
        return None

    def get_q(self, motor_id):
        states = self.motor_states()
        if states is None or motor_id >= len(states):
            return float("nan")
        return float(getattr(states[motor_id], "q", float("nan")))

    def connect(self):
        print("============================================================")
        print(" SDK2 DIRECT H1 ELBOW SWEEP TEST")
        print("============================================================")
        print(f"DOMAIN:   {DOMAIN}")
        print(f"IFACE:    {IFACE}")
        print(f"KP/KD:    {KP_ARM} / {KD_ARM}")
        print(f"RIGHT ELBOW ID: {RIGHT_ELBOW_ID}")
        print(f"LEFT  ELBOW ID: {LEFT_ELBOW_ID}")
        print("============================================================")

        ChannelFactoryInitialize(DOMAIN, IFACE)

        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(self.lowstate_cb, 10)

        print("Waiting for rt/lowstate...")
        t0 = time.time()
        while self.lowstate is None and time.time() - t0 < 5.0:
            print(".", end="", flush=True)
            time.sleep(0.1)
        print()

        if self.lowstate is None:
            raise RuntimeError("No rt/lowstate received")

        states = self.motor_states()
        if states is None:
            raise RuntimeError("Cannot access motor_state field")

        self.current_right_elbow_cmd = self.get_q(RIGHT_ELBOW_ID)
        self.current_left_elbow_cmd = self.get_q(LEFT_ELBOW_ID)

        print("rt/lowstate OK")
        print(
            "initial actual elbows:",
            f"right id15={self.current_right_elbow_cmd:+.4f}",
            f"left id19={self.current_left_elbow_cmd:+.4f}",
        )

        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()

    def make_lowcmd(self):
        if unitree_go_msg_dds__LowCmd_ is None:
            raise RuntimeError(
                "unitree_go_msg_dds__LowCmd_ not found. "
                "Cannot create default LowCmd for unitree_go IDL."
            )
        return unitree_go_msg_dds__LowCmd_()

    def init_cmd_base(self, cmd):
        if hasattr(cmd, "head") and len(cmd.head) >= 2:
            cmd.head[0] = 0xFE
            cmd.head[1] = 0xEF

        if hasattr(cmd, "level_flag"):
            cmd.level_flag = 0xFF

        if hasattr(cmd, "gpio"):
            cmd.gpio = 0

    def fill_cmd(self, left_elbow_target, right_elbow_target):
        cmd = self.make_lowcmd()
        self.init_cmd_base(cmd)

        states = self.motor_states()
        motor_cmd = self.motor_cmds(cmd)

        if states is None:
            raise RuntimeError("No motor states")
        if motor_cmd is None:
            raise RuntimeError("No motor_cmd field")

        n = min(len(states), len(motor_cmd))

        # По умолчанию ничего активно не двигаем.
        # q заполняем текущими значениями, kp/kd = 0.
        for i in range(n):
            ms = states[i]
            mc = motor_cmd[i]

            if hasattr(mc, "mode"):
                mc.mode = 0x00
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

        # Плечи удерживаем в текущем положении, чтобы двигались только локти.
        for motor_id in ARM_IDS:
            if motor_id >= n:
                continue

            ms = states[motor_id]
            mc = motor_cmd[motor_id]

            if hasattr(mc, "mode"):
                mc.mode = 0x01
            if hasattr(mc, "q"):
                mc.q = float(getattr(ms, "q", 0.0))
            if hasattr(mc, "dq"):
                mc.dq = 0.0
            if hasattr(mc, "kp"):
                mc.kp = KP_ARM
            if hasattr(mc, "kd"):
                mc.kd = KD_ARM
            if hasattr(mc, "tau"):
                mc.tau = 0.0

        # Локти — целевые значения теста.
        for motor_id, target in [
            (LEFT_ELBOW_ID, left_elbow_target),
            (RIGHT_ELBOW_ID, right_elbow_target),
        ]:
            if motor_id >= n:
                continue

            mc = motor_cmd[motor_id]

            if hasattr(mc, "mode"):
                mc.mode = 0x01
            if hasattr(mc, "q"):
                mc.q = float(target)
            if hasattr(mc, "dq"):
                mc.dq = 0.0
            if hasattr(mc, "kp"):
                mc.kp = KP_ARM
            if hasattr(mc, "kd"):
                mc.kd = KD_ARM
            if hasattr(mc, "tau"):
                mc.tau = 0.0

        if self.crc is not None and hasattr(cmd, "crc"):
            cmd.crc = self.crc.Crc(cmd)

        return cmd

    def publish(self, left_elbow_target, right_elbow_target):
        cmd = self.fill_cmd(left_elbow_target, right_elbow_target)
        self.pub.Write(cmd)

    def move_elbows_to(self, target, hold_sec=2.0):
        print()
        print("------------------------------------------------------------")
        print(f"TARGET ELBOW q = {target:+.3f}")
        print("------------------------------------------------------------")

        t0 = time.time()
        next_log = t0

        while time.time() - t0 < hold_sec:
            # Плавно ведём команду к target.
            dl = target - self.current_left_elbow_cmd
            dr = target - self.current_right_elbow_cmd

            self.current_left_elbow_cmd += max(-MAX_STEP_RAD, min(MAX_STEP_RAD, dl))
            self.current_right_elbow_cmd += max(-MAX_STEP_RAD, min(MAX_STEP_RAD, dr))

            self.publish(self.current_left_elbow_cmd, self.current_right_elbow_cmd)

            now = time.time()
            if now >= next_log:
                print(
                    f"cmd: L={self.current_left_elbow_cmd:+.3f}, R={self.current_right_elbow_cmd:+.3f} | "
                    f"actual lowstate: L_id19={self.get_q(LEFT_ELBOW_ID):+.3f}, "
                    f"R_id15={self.get_q(RIGHT_ELBOW_ID):+.3f}"
                )
                next_log = now + 0.25

            time.sleep(DT)

        # Небольшая фиксация на target.
        t1 = time.time()
        while time.time() - t1 < 0.7:
            self.publish(target, target)
            time.sleep(DT)

        print(
            f"FINAL actual: L_id19={self.get_q(LEFT_ELBOW_ID):+.4f}, "
            f"R_id15={self.get_q(RIGHT_ELBOW_ID):+.4f}"
        )

    def run(self):
        self.connect()

        print()
        print("ВНИМАНИЕ: тест напрямую отправляет rt/lowcmd только на руки.")
        print("Старый ROS sender должен быть остановлен.")
        print("Робот должен стоять устойчиво, руки свободны.")
        ans = input("Type START to begin elbow sweep: ").strip()
        if ans != "START":
            print("Cancelled.")
            return

        for target in ELBOW_TARGETS:
            self.move_elbows_to(target, hold_sec=2.0)

        print()
        print("Done. Elbows returned to 1.57 target.")


def main():
    test = ElbowSweepTest()
    test.run()


if __name__ == "__main__":
    main()
