#!/usr/bin/env python3

import time
import math
import numpy as np
import os
from enum import IntEnum

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmd, LowState_ as LowState
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC


kTopicLowCommand = "rt/lowcmd"
kTopicLowState = "rt/lowstate"


class H1JointIndex(IntEnum):
    kRightHipRoll = 0
    kRightHipPitch = 1
    kRightKnee = 2
    kLeftHipRoll = 3
    kLeftHipPitch = 4
    kLeftKnee = 5
    kWaistYaw = 6
    kLeftHipYaw = 7
    kRightHipYaw = 8
    kNotUsedJoint = 9
    kLeftAnkle = 10
    kRightAnkle = 11

    kRightShoulderPitch = 12
    kRightShoulderRoll = 13
    kRightShoulderYaw = 14
    kRightElbow = 15

    kLeftShoulderPitch = 16
    kLeftShoulderRoll = 17
    kLeftShoulderYaw = 18
    kLeftElbow = 19


WEAK_MOTORS = {
    H1JointIndex.kLeftAnkle.value,
    H1JointIndex.kRightAnkle.value,

    H1JointIndex.kLeftShoulderPitch.value,
    H1JointIndex.kLeftShoulderRoll.value,
    H1JointIndex.kLeftShoulderYaw.value,
    H1JointIndex.kLeftElbow.value,

    H1JointIndex.kRightShoulderPitch.value,
    H1JointIndex.kRightShoulderRoll.value,
    H1JointIndex.kRightShoulderYaw.value,
    H1JointIndex.kRightElbow.value,
}


def read_lowstate(sub):
    while True:
        msg = sub.Read()
        if msg is not None:
            return msg
        time.sleep(0.01)


def main():
    domain_id = int(os.environ.get("UNITREE_DOMAIN_ID", os.environ.get("ROS_DOMAIN_ID", "42")))
    net_iface = os.environ.get("UNITREE_NET_IFACE", os.environ.get("MUJOCO_NET_IFACE", ""))

    print(f"ChannelFactoryInitialize(domain_id={domain_id}, networkInterface={net_iface or None})")
    if net_iface:
        ChannelFactoryInitialize(domain_id, networkInterface=net_iface)
    else:
        ChannelFactoryInitialize(domain_id)

    pub = ChannelPublisher(kTopicLowCommand, LowCmd)
    pub.Init()

    sub = ChannelSubscriber(kTopicLowState, LowState)
    sub.Init()

    crc = CRC()

    print("Waiting for rt/lowstate...")
    lowstate = read_lowstate(sub)

    base_q = np.zeros(20)
    for i in range(20):
        base_q[i] = lowstate.motor_state[i].q

    print("Got lowstate.")
    print("Initial q:", np.round(base_q, 4))

    msg = unitree_go_msg_dds__LowCmd_()
    msg.head[0] = 0xFE
    msg.head[1] = 0xEF
    msg.level_flag = 0xFF
    msg.gpio = 0

    for i in range(20):
        mc = msg.motor_cmd[i]

        if i in WEAK_MOTORS:
            mc.mode = 0x01
            mc.kp = 140.0
            mc.kd = 3.0
        else:
            mc.mode = 0x0A
            mc.kp = 300.0
            mc.kd = 5.0

        mc.q = float(base_q[i])
        mc.dq = 0.0
        mc.tau = 0.0

    print("Publishing sine to LEFT shoulder pitch, motor 16.")
    print("Press Ctrl+C to stop.")

    t0 = time.time()
    dt = 1.0 / 250.0

    try:
        while True:
            t = time.time() - t0

            q = base_q.copy()
            q[H1JointIndex.kLeftShoulderPitch.value] += 0.35 * math.sin(2.0 * math.pi * 0.15 * t)

            for i in range(20):
                msg.motor_cmd[i].q = float(q[i])
                msg.motor_cmd[i].dq = 0.0
                msg.motor_cmd[i].tau = 0.0

            msg.crc = crc.Crc(msg)
            pub.Write(msg)

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\nStopping. Sending base pose for 1 second...")
        for _ in range(250):
            for i in range(20):
                msg.motor_cmd[i].q = float(base_q[i])
                msg.motor_cmd[i].dq = 0.0
                msg.motor_cmd[i].tau = 0.0
            msg.crc = crc.Crc(msg)
            pub.Write(msg)
            time.sleep(dt)


if __name__ == "__main__":
    main()
