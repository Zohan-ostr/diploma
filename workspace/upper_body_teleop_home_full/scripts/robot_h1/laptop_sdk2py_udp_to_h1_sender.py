#!/usr/bin/env python3
import inspect
import json
import math
import socket
import time
from typing import Dict, List, Optional

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
import unitree_sdk2py.idl.unitree_go.msg.dds_ as dds
from unitree_sdk2py.utils.crc import CRC


POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0

ARM_IDS = [12, 13, 14, 15, 16, 17, 18, 19]

JOINT_TO_ID = {
    "right_shoulder_pitch": 12,
    "right_shoulder_roll": 13,
    "right_shoulder_yaw": 14,
    "right_elbow": 15,

    "left_shoulder_pitch": 16,
    "left_shoulder_roll": 17,
    "left_shoulder_yaw": 18,
    "left_elbow": 19,

    "right_shoulder_pitch_joint": 12,
    "right_shoulder_roll_joint": 13,
    "right_shoulder_yaw_joint": 14,
    "right_elbow_joint": 15,

    "left_shoulder_pitch_joint": 16,
    "left_shoulder_roll_joint": 17,
    "left_shoulder_yaw_joint": 18,
    "left_elbow_joint": 19,
}


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--net_iface", default="enx00e04c36022c")
    parser.add_argument("--domain", type=int, default=0)

    # Для Unitree SDK2 обычно используются rt/... DDS topic names.
    # Если rt/arm_sdk не сработает, попробуем rt/lowcmd.
    parser.add_argument("--sdk_cmd_topic", default="rt/arm_sdk")
    parser.add_argument("--sdk_state_topic", default="rt/lowstate")

    parser.add_argument("--udp_bind_host", default="0.0.0.0")
    parser.add_argument("--udp_port", type=int, default=50051)

    parser.add_argument("--rate_hz", type=float, default=250.0)

    # Безопасный старт.
    parser.add_argument("--kp_arm", type=float, default=25.0)
    parser.add_argument("--kd_arm", type=float, default=1.5)

    parser.add_argument("--max_step_rad", type=float, default=0.012)
    parser.add_argument("--yaw_max_step_rad", type=float, default=0.020)
    parser.add_argument("--elbow_max_step_rad", type=float, default=0.025)
    parser.add_argument("--command_timeout_sec", type=float, default=0.35)

    # On sender start, move arms to T-pose even before UDP commands arrive.
    # If negative, keep startup T-pose until the first valid UDP command.
    parser.add_argument("--startup_tpose_sec", type=float, default=-1.0)

    return parser.parse_args()


class Sdk2PyUdpToH1Sender:
    def __init__(self, args):
        self.args = args

        print("=" * 60)
        print("SDK2PY UDP -> H1 LOWCMD SENDER")
        print("=" * 60)
        print(f"net_iface:           {args.net_iface}")
        print(f"domain:              {args.domain}")
        print(f"sdk_cmd_topic:       {args.sdk_cmd_topic}")
        print(f"sdk_state_topic:     {args.sdk_state_topic}")
        print(f"udp:                 {args.udp_bind_host}:{args.udp_port}")
        print(f"rate_hz:             {args.rate_hz}")
        print(f"kp_arm:              {args.kp_arm}")
        print(f"kd_arm:              {args.kd_arm}")
        print(f"max_step_rad:        {args.max_step_rad}")
        print(f"yaw_max_step_rad:    {args.yaw_max_step_rad}")
        print(f"elbow_max_step_rad:  {args.elbow_max_step_rad}")
        print(f"command_timeout_sec: {args.command_timeout_sec}")
        print(f"startup_tpose_sec:   {args.startup_tpose_sec}")
        print("=" * 60)

        ChannelFactoryInitialize(args.domain, args.net_iface)

        self.lowcmd_pub = ChannelPublisher(args.sdk_cmd_topic, LowCmd_)
        self.lowcmd_pub.Init()

        self.lowstate_sub = ChannelSubscriber(args.sdk_state_topic, LowState_)
        self.lowstate_sub.Init()

        self.crc = CRC()

        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.bind((args.udp_bind_host, args.udp_port))
        self.udp.setblocking(False)

        self.current_q: Optional[List[float]] = None
        self.target_q: Optional[List[float]] = None
        self.sent_q: Optional[List[float]] = None

        self.last_udp_time = 0.0
        self.got_udp = False
        self.seq = 0

        if float(args.startup_tpose_sec) < 0.0:
            self.startup_tpose_until = float("inf")
            self.startup_tpose_active = True
        else:
            self.startup_tpose_until = time.monotonic() + float(args.startup_tpose_sec)
            self.startup_tpose_active = float(args.startup_tpose_sec) > 0.0

    def read_lowstate(self):
        msg = self.lowstate_sub.Read()
        if msg is None:
            return

        q = [float(m.q) for m in msg.motor_state]
        if len(q) < 20:
            return

        self.current_q = q

        if self.target_q is None:
            self.target_q = list(q)
            self.sent_q = list(q)

            # Initial T-pose target on every sender start.
            # Order by motor ids:
            # 12 right_shoulder_pitch
            # 13 right_shoulder_roll
            # 14 right_shoulder_yaw
            # 15 right_elbow
            # 16 left_shoulder_pitch
            # 17 left_shoulder_roll
            # 18 left_shoulder_yaw
            # 19 left_elbow
            if self.startup_tpose_active:
                self.target_q[12] = -0.10
                self.target_q[13] = -1.57
                self.target_q[14] = -1.89
                self.target_q[15] = 1.30

                self.target_q[16] = -0.10
                self.target_q[17] = 1.57
                self.target_q[18] = 1.89
                self.target_q[19] = 1.30

            print("lowstate OK, initial arm q:", [round(self.sent_q[i], 4) for i in ARM_IDS])
            if self.startup_tpose_active:
                print("startup T-pose target:", [round(self.target_q[i], 4) for i in ARM_IDS])

    def read_udp(self):
        while True:
            try:
                data, _addr = self.udp.recvfrom(65535)
            except BlockingIOError:
                break

            try:
                payload = json.loads(data.decode("utf-8"))
            except Exception as e:
                print("bad udp json:", e)
                continue

            if not payload.get("valid", False):
                continue

            if self.target_q is None:
                continue

            names = payload.get("joint_names", [])
            pos = payload.get("position", [])

            mapped = 0
            for name, q in zip(names, pos):
                motor_id = JOINT_TO_ID.get(str(name))
                if motor_id is None:
                    continue

                q = float(q)
                if not math.isfinite(q):
                    continue

                self.target_q[motor_id] = q
                mapped += 1

            if mapped > 0:
                self.got_udp = True
                self.last_udp_time = time.monotonic()

                if self.startup_tpose_active:
                    self.startup_tpose_active = False
                    print("first UDP command received: startup T-pose released")

            if self.seq % 125 == 0:
                print("udp mapped:", mapped, "target_arm:", [round(self.target_q[i], 3) for i in ARM_IDS])

    def _make_default_idl_obj(self, cls):
        """
        unitree_sdk2py generated IDL classes in this robot image require
        all constructor arguments. This helper builds a zero/default object
        using the constructor signature.
        """
        name = cls.__name__
        sig = inspect.signature(cls)
        kwargs = {}

        for field in sig.parameters.keys():
            if name == "LowCmd_":
                if field == "head":
                    kwargs[field] = [0, 0]
                elif field == "level_flag":
                    kwargs[field] = 0
                elif field == "frame_reserve":
                    kwargs[field] = 0
                elif field == "sn":
                    kwargs[field] = [0, 0]
                elif field == "version":
                    kwargs[field] = [0, 0]
                elif field == "bandwidth":
                    kwargs[field] = 0
                elif field == "motor_cmd":
                    MotorCmd_ = getattr(dds, "MotorCmd_")
                    kwargs[field] = [self._make_default_idl_obj(MotorCmd_) for _ in range(20)]
                elif field == "bms_cmd":
                    BmsCmd_ = getattr(dds, "BmsCmd_")
                    kwargs[field] = self._make_default_idl_obj(BmsCmd_)
                elif field == "wireless_remote":
                    kwargs[field] = [0] * 40
                elif field == "led":
                    kwargs[field] = [0] * 12
                elif field == "fan":
                    kwargs[field] = [0, 0]
                elif field == "gpio":
                    kwargs[field] = 0
                elif field == "reserve":
                    kwargs[field] = 0
                elif field == "crc":
                    kwargs[field] = 0
                else:
                    kwargs[field] = 0

            elif name == "MotorCmd_":
                if field == "mode":
                    kwargs[field] = 0
                elif field in ("q", "dq", "tau", "kp", "kd"):
                    kwargs[field] = 0.0
                elif field == "reserve":
                    kwargs[field] = [0, 0, 0]
                else:
                    kwargs[field] = 0

            elif name == "BmsCmd_":
                if field == "off":
                    kwargs[field] = 0
                elif field == "reserve":
                    kwargs[field] = [0, 0, 0]
                else:
                    kwargs[field] = 0

            elif name == "LED_":
                kwargs[field] = 0

            else:
                kwargs[field] = 0

        return cls(**kwargs)

    def make_lowcmd(self):
        try:
            msg = self._make_default_idl_obj(LowCmd_)
        except Exception as e:
            print("FAILED to construct LowCmd_:", repr(e))
            print("LowCmd_ signature:", inspect.signature(LowCmd_))
            for cls_name in ("MotorCmd_", "BmsCmd_"):
                cls = getattr(dds, cls_name, None)
                if cls is not None:
                    print(f"{cls_name} signature:", inspect.signature(cls))
            raise

        msg.head[0] = 0xFE
        msg.head[1] = 0xEF
        msg.level_flag = 0xFF
        msg.gpio = 0

        for i, motor in enumerate(msg.motor_cmd):
            if i in ARM_IDS:
                motor.mode = 0x01
                motor.q = float(self.sent_q[i])
                motor.dq = 0.0
                motor.kp = float(self.args.kp_arm)
                motor.kd = float(self.args.kd_arm)
                motor.tau = 0.0
            else:
                motor.mode = 0x01
                motor.q = float(POS_STOP_F)
                motor.dq = float(VEL_STOP_F)
                motor.kp = 0.0
                motor.kd = 0.0
                motor.tau = 0.0

        msg.crc = self.crc.Crc(msg)
        return msg

    def step(self):
        self.read_lowstate()
        self.read_udp()

        if self.target_q is None or self.sent_q is None:
            return

        now = time.monotonic()

        if self.startup_tpose_active and now > self.startup_tpose_until:
            self.startup_tpose_active = False

        age = now - self.last_udp_time
        timeout = (not self.got_udp) or (age > self.args.command_timeout_sec)

        # Normal safety behavior: if UDP commands disappear, hold current sent position.
        # Exception: during startup T-pose window we keep moving toward T-pose.
        if timeout and not self.startup_tpose_active:
            for i in ARM_IDS:
                self.target_q[i] = self.sent_q[i]

        for i in ARM_IDS:
            if i in (14, 18):
                limit = self.args.yaw_max_step_rad
            elif i in (15, 19):
                limit = self.args.elbow_max_step_rad
            else:
                limit = self.args.max_step_rad

            delta = self.target_q[i] - self.sent_q[i]
            delta = clamp(delta, -limit, limit)
            self.sent_q[i] += delta

        self.lowcmd_pub.Write(self.make_lowcmd())

        self.seq += 1

        if self.seq % 125 == 0:
            cur = self.current_q if self.current_q is not None else [0.0] * 20
            print(
                "timeout=", int(timeout),
                " target=", [round(self.target_q[i], 3) for i in ARM_IDS],
                " sent=", [round(self.sent_q[i], 3) for i in ARM_IDS],
                " current=", [round(cur[i], 3) for i in ARM_IDS],
            )

    def run(self):
        period = 1.0 / max(1.0, self.args.rate_hz)
        next_t = time.monotonic()

        while True:
            self.step()

            next_t += period
            sleep_t = next_t - time.monotonic()

            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                next_t = time.monotonic()


def main():
    args = parse_args()

    print()
    print("Это отправляет команды в Unitree SDK2 DDS topic.")
    print("Перед запуском убедись, что robot high-level motion service не конфликтует с low-level руками.")
    print("Starting without interactive confirmation.")
    print()

    sender = Sdk2PyUdpToH1Sender(args)
    sender.run()


if __name__ == "__main__":
    main()
