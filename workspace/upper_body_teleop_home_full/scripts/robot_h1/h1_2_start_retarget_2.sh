#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."
source scripts/robot_h1/h1_env.sh

INPUT_TOPIC="${INPUT_TOPIC:-/pose/landmarks}"
OUTPUT_TOPIC="${OUTPUT_TOPIC:-/upper_body/command_geom}"
CALIBRATION_TOPIC="${CALIBRATION_TOPIC:-/upper_body/start_calibration}"

print_h1_env

echo "============================================================"
echo " H1 2: START TEMPLATE POSE RETARGET 2"
echo "============================================================"
echo "INPUT:             $INPUT_TOPIC"
echo "OUTPUT:            $OUTPUT_TOPIC"
echo "CALIBRATION_TOPIC: $CALIBRATION_TOPIC"
echo "ROS_DOMAIN_ID:     $ROS_DOMAIN_ID"
echo "MODE:              6 pitch/roll templates + elbow angle + yaw search"
echo "============================================================"

pkill -f "upper_body_teleop_runtime.vector_fabrik_retarget" 2>/dev/null || true
pkill -f "upper_body_teleop_runtime.template_pose_retarget_2" 2>/dev/null || true
pkill -f "template_pose_retarget_2.py" 2>/dev/null || true
sleep 1

python3 -m upper_body_teleop_runtime.template_pose_retarget_2 \
  --ros-args \
  -p input_topic:="$INPUT_TOPIC" \
  -p output_topic:="$OUTPUT_TOPIC" \
  -p calibration_topic:="$CALIBRATION_TOPIC" \
  -p calibration_frames:=30 \
  -p min_visibility:=0.01 \
  -p landmark_alpha:=0.55 \
  -p joint_alpha:=0.68 \
  -p max_joint_step:=0.220 \
  -p yaw_max_step:=0.320 \
  -p elbow_max_step:=0.260 \
  -p left_yaw_down:=-1.30 \
  -p left_yaw_up:=1.89 \
  -p right_yaw_down:=1.30 \
  -p right_yaw_up:=-1.89 \
  -p yaw_grid:=121 \
  -p yaw_direction_weight:=1.0 \
  -p yaw_continuity_weight:=0.025 \
  -p no_back_yaw_penalty:=12.0 \
  -p no_back_z_margin:=0.02 \
  -p left_elbow_straight:=1.57 \
  -p right_elbow_straight:=1.57 \
  -p elbow_gain:=2.20 \
  -p elbow_bend_deadzone:=0.0 \
  -p elbow_bend_response_gain:=3.20 \
  -p use_calibrated_elbow_bias:=true \
  -p template_switch_margin:=0.08 \
  -p depth_gain:=1.5 \
  -p elbow_bias_scale:=0.0 \
  -p standard_z_sign:=1.0
