#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."
source scripts/robot_h1/h1_env.sh

INPUT_TOPIC="${INPUT_TOPIC:-/pose/landmarks}"
CALIBRATION_TOPIC="${CALIBRATION_TOPIC:-/upper_body/start_calibration}"

print_h1_env

echo "============================================================"
echo " H1 2: START VECTOR FABRIK RETARGET DIRECT SDK2"
echo "============================================================"
echo "INPUT_TOPIC:       $INPUT_TOPIC"
echo "CALIBRATION_TOPIC: $CALIBRATION_TOPIC"
echo "UNITREE_NET_IFACE: $UNITREE_NET_IFACE"
echo "UNITREE_DOMAIN_ID: $UNITREE_DOMAIN_ID"
echo "ROS_DOMAIN_ID:     $ROS_DOMAIN_ID"
echo "OUTPUT:            direct SDK2 rt/lowcmd"
echo "KP/KD:             60.0 / 1.5"
echo "============================================================"

echo "Stopping old ROS sender / pose players / retarget..."
pkill -f "upper_body_teleop_runtime.h1_sdk2py_upper_body_sender" 2>/dev/null || true
pkill -f "upper_body_teleop_runtime.photo_pose_player_15" 2>/dev/null || true
pkill -f "upper_body_teleop_runtime.sdk2_photo_pose_player_15_direct" 2>/dev/null || true
pkill -f "upper_body_teleop_runtime.vector_fabrik_retarget" 2>/dev/null || true
sleep 1

python3 -m upper_body_teleop_runtime.vector_fabrik_retarget \
  --ros-args \
  -p input_topic:="$INPUT_TOPIC" \
  -p calibration_topic:="$CALIBRATION_TOPIC" \
  -p calibration_frames:=90 \
  -p calibration_duration_sec:=3.0 \
  -p min_visibility:=0.01 \
  -p landmark_alpha:=0.35 \
  -p joint_alpha:=0.50 \
  -p robot_upper_len:=0.31 \
  -p robot_fore_len:=0.31 \
  -p pitch_grid:=15 \
  -p roll_grid:=21 \
  -p yaw_grid:=61 \
  -p pitch_window:=1.30 \
  -p roll_window:=1.60 \
  -p upper_direction_weight:=1.0 \
  -p upper_continuity_weight:=0.015 \
  -p yaw_direction_weight:=1.0 \
  -p yaw_wrist_position_weight:=4.0 \
  -p yaw_refine_window:=0.18 \
  -p yaw_refine_grid:=9 \
  -p yaw_continuity_weight:=0.05 \
  -p pitch_geom_gain:=1.0 \
  -p left_yaw_down:=-1.30 \
  -p left_yaw_up:=1.74 \
  -p right_yaw_down:=1.30 \
  -p right_yaw_up:=-1.74 \
  -p left_elbow_straight:=1.57 \
  -p right_elbow_straight:=1.57 \
  -p elbow_gain:=0.85 \
  -p elbow_bend_deadzone:=0.025 \
  -p elbow_bend_response_gain:=1.15 \
  -p use_calibrated_elbow_bias:=true \
  -p max_joint_step:=0.110 \
  -p yaw_max_step:=0.260 \
  -p elbow_max_step:=0.140 \
  -p standard_z_sign:=1.0 \
  -p use_input_frame_direct:=true \
  -p input_z_sign:=-1.0 \
  -p unitree_net_iface:="$UNITREE_NET_IFACE" \
  -p unitree_domain_id:="$UNITREE_DOMAIN_ID" \
  -p sdk_control_hz:=250.0 \
  -p kp_arm:=60.0 \
  -p kd_arm:=1.5 \
  -p sdk_max_step_rad:=0.018
