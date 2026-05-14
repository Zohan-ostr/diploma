#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/build"
export LD_LIBRARY_PATH=${HOME}/WS_OZ/unitree_sdk2/thirdparty/lib/x86_64:/usr/local/lib:${LD_LIBRARY_PATH:-}
IFACE=${IFACE:-eth0}
MOTOR_ID=${MOTOR_ID:-12}
AMPLITUDE=${AMPLITUDE:-0.10}
KP=${KP:-20.0}
KD=${KD:-1.0}
DURATION=${DURATION:-6.0}
./h1_sdk2_right_arm_test "$IFACE" "$MOTOR_ID" "$AMPLITUDE" "$KP" "$KD" "$DURATION"
