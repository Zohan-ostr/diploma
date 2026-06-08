#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."
source scripts/robot_h1/h1_env.sh

INPUT_TOPIC="${INPUT_TOPIC:-/upper_body/command_geom}"
CONTROL_HZ="${CONTROL_HZ:-250.0}"

# ============================================================
# ВАЖНО:
# Для реального Unitree H1 коэффициенты PD-регулятора
# фиксированы и не должны переопределяться извне.
# ============================================================
KP_ARM="60.0"
KD_ARM="1.5"

MAX_STEP_RAD="${MAX_STEP_RAD:-0.012}"
ARM_VELOCITY_LIMIT="${ARM_VELOCITY_LIMIT:-2.6}"
COMMAND_TIMEOUT_SEC="${COMMAND_TIMEOUT_SEC:-0.5}"

print_h1_env

echo "============================================================"
echo " H1 4: START SDK2 SENDER"
echo "============================================================"
echo "INPUT_TOPIC:         $INPUT_TOPIC"
echo "UNITREE_NET_IFACE:   $UNITREE_NET_IFACE"
echo "UNITREE_DOMAIN_ID:   $UNITREE_DOMAIN_ID"
echo "ROS_DOMAIN_ID:       $ROS_DOMAIN_ID"
echo "CONTROL_HZ:          $CONTROL_HZ"
echo "KP_ARM/KD_ARM:       $KP_ARM / $KD_ARM"
echo "MAX_STEP_RAD:        $MAX_STEP_RAD"
echo "ARM_VELOCITY_LIMIT:  $ARM_VELOCITY_LIMIT"
echo "COMMAND_TIMEOUT_SEC: $COMMAND_TIMEOUT_SEC"
echo "============================================================"

if [ "$KP_ARM" != "60.0" ] || [ "$KD_ARM" != "1.5" ]; then
  echo "ERROR: For real H1 expected KP_ARM=60.0 and KD_ARM=1.5"
  echo "Current: KP_ARM=$KP_ARM KD_ARM=$KD_ARM"
  exit 1
fi

echo "Stopping old H1 SDK2 sender..."
pkill -f "upper_body_teleop_runtime.h1_sdk2py_upper_body_sender" 2>/dev/null || true
sleep 1

python3 -m upper_body_teleop_runtime.h1_sdk2py_upper_body_sender \
  --ros-args \
  -p input_topic:="$INPUT_TOPIC" \
  -p control_hz:="$CONTROL_HZ" \
  -p kp_arm:="$KP_ARM" \
  -p kd_arm:="$KD_ARM" \
  -p max_step_rad:="$MAX_STEP_RAD" \
  -p arm_velocity_limit:="$ARM_VELOCITY_LIMIT" \
  -p command_timeout_sec:="$COMMAND_TIMEOUT_SEC"
