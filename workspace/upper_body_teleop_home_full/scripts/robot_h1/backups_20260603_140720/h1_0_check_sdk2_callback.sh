#!/usr/bin/env bash
set -e

source ~/diploma/workspace/upper_body_teleop_home_full/scripts/robot_h1/h1_robot_env.sh

python3 - <<'PY'
import os
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

domain = int(os.environ.get("UNITREE_DOMAIN_ID", "0"))
iface = os.environ.get("UNITREE_NET_IFACE", "")

count = 0
last = None

def handler(msg):
    global count, last
    count += 1
    last = msg

print("SDK family: unitree_go")
print("domain:", domain)
print("iface:", iface)

if iface:
    ChannelFactoryInitialize(domain, iface)
else:
    ChannelFactoryInitialize(domain)

sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init(handler, 10)

print("Waiting rt/lowstate callback for 5 sec...")
for _ in range(10):
    print(".", end="", flush=True)
    time.sleep(0.5)

print()
print("lowstate messages:", count)

if last is not None:
    qs = [round(float(m.q), 4) for m in last.motor_state[:20]]
    print("first 20 q:", qs)
    print("OK: rt/lowstate callback received")
else:
    print("NO LOWSTATE CALLBACK")
    print("Network ping may work, but DDS lowstate was not received.")
PY
