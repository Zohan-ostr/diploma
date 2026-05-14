#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")/../.."

ROS_DOMAIN_ID_VALUE="${ROS_DOMAIN_ID_VALUE:-0}"
OUTPUT_TOPIC="${OUTPUT_TOPIC:-/arm_sdk}"
LOWSTATE_TOPIC="${LOWSTATE_TOPIC:-/lowstate}"

AMPLITUDE_RAD="${AMPLITUDE_RAD:-0.10}"
KP_ARM="${KP_ARM:-60.0}"
KD_ARM="${KD_ARM:-3.0}"
RATE_HZ="${RATE_HZ:-250.0}"
DURATION_SEC="${DURATION_SEC:-12.0}"
TEST_MOTOR_ID="${TEST_MOTOR_ID:-12}"

source scripts/robot_h1/h1_unitree_ros2_env.sh
export ROS_DOMAIN_ID="$ROS_DOMAIN_ID_VALUE"

echo "============================================================"
echo " H1 UNITREE OFFICIAL ROS2 ARM TEST"
echo "============================================================"
echo "ROS_DOMAIN_ID:    $ROS_DOMAIN_ID"
echo "RMW:              $RMW_IMPLEMENTATION"
echo "OUTPUT_TOPIC:     $OUTPUT_TOPIC"
echo "LOWSTATE_TOPIC:   $LOWSTATE_TOPIC"
echo "AMPLITUDE_RAD:    $AMPLITUDE_RAD"
echo "KP_ARM:           $KP_ARM"
echo "KD_ARM:           $KD_ARM"
echo "RATE_HZ:          $RATE_HZ"
echo "TEST_MOTOR_ID:    $TEST_MOTOR_ID"
echo "============================================================"
echo
echo "Это отправляет команды в $OUTPUT_TOPIC."
echo "Для первого теста держи амплитуду маленькой."
read -r -p "Type YES to start: " CONFIRM

if [ "$CONFIRM" != "YES" ]; then
  echo "Cancelled."
  exit 1
fi

/usr/bin/python3 scripts/robot_h1/h1_arm_sdk_full_lowcmd_test.py \
  --ros-args \
  -p output_topic:="$OUTPUT_TOPIC" \
  -p lowstate_topic:="$LOWSTATE_TOPIC" \
  -p amplitude_rad:="$AMPLITUDE_RAD" \
  -p kp_arm:="$KP_ARM" \
  -p kd_arm:="$KD_ARM" \
  -p rate_hz:="$RATE_HZ" \
  -p duration_sec:="$DURATION_SEC" \
  -p test_motor_id:="$TEST_MOTOR_ID"
