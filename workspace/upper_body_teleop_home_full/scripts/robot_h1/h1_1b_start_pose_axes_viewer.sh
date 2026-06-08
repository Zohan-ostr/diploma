#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."
source scripts/robot_h1/h1_env.sh

INPUT_TOPIC="${INPUT_TOPIC:-/pose/landmarks}"
DEPTH_GAIN="${DEPTH_GAIN:-1.5}"

print_h1_env

echo "============================================================"
echo " H1 1B: START POSE AXES VIEWER"
echo "============================================================"
echo "INPUT_TOPIC: $INPUT_TOPIC"
echo "DEPTH_GAIN:  $DEPTH_GAIN"
echo "============================================================"

python3 -m upper_body_teleop_runtime.pose_axes_viewer \
  --ros-args \
  -p input_topic:="$INPUT_TOPIC" \
  -p depth_gain:="$DEPTH_GAIN" \
  -p standard_z_sign:=1.0
