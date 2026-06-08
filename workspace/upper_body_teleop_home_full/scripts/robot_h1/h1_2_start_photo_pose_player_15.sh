#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."
source scripts/robot_h1/h1_env.sh

OUTPUT_TOPIC="${OUTPUT_TOPIC:-/upper_body/command_geom}"

print_h1_env

echo "============================================================"
echo " H1 2: START PHOTO POSE PLAYER 15"
echo "============================================================"
echo "OUTPUT_TOPIC: $OUTPUT_TOPIC"
echo "CONTROL:      [ previous, ] next, r T-pose, q quit"
echo "ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo "============================================================"

pkill -f "upper_body_teleop_runtime.vector_fabrik_retarget" 2>/dev/null || true
pkill -f "upper_body_teleop_runtime.template_pose_retarget_2" 2>/dev/null || true
pkill -f "upper_body_teleop_runtime.photo_pose_player_15" 2>/dev/null || true
sleep 1

python3 -m upper_body_teleop_runtime.photo_pose_player_15 \
  --ros-args \
  -p output_topic:="$OUTPUT_TOPIC" \
  -p publish_hz:=50.0 \
  -p max_step_rad:=0.018
