#!/usr/bin/env bash
set -e

source ~/diploma/workspace/upper_body_teleop_home_full/scripts/robot_h1/h1_robot_env.sh

cd ~/diploma/workspace/upper_body_teleop_home_full

echo "============================================================"
echo " H1 2: START VECTOR FABRIK RETARGET"
echo "============================================================"
echo "INPUT:  /pose/landmarks"
echo "OUTPUT: /upper_body/command_geom"
echo "============================================================"

python3 -m upper_body_teleop_runtime.vector_fabrik_retarget \
  --ros-args \
  -p input_topic:=/pose/landmarks \
  -p output_topic:=/upper_body/command_geom \
  -p calibration_topic:=/upper_body/start_calibration \
  -p calibration_frames:=45 \
  -p landmark_alpha:=0.35 \
  -p joint_alpha:=0.75 \
  -p robot_upper_len:=0.31 \
  -p robot_fore_len:=0.31 \
  -p pitch_grid:=21 \
  -p roll_grid:=31 \
  -p yaw_grid:=81 \
  -p pitch_window:=1.30 \
  -p roll_window:=1.60 \
  -p upper_direction_weight:=1.0 \
  -p upper_continuity_weight:=0.015 \
  -p yaw_direction_weight:=1.0 \
  -p yaw_wrist_position_weight:=4.0 \
  -p yaw_refine_window:=0.18 \
  -p yaw_refine_grid:=17 \
  -p yaw_continuity_weight:=0.05 \
  -p pitch_geom_gain:=1.0 \
  -p left_yaw_down:=-1.30 \
  -p left_yaw_up:=1.74 \
  -p right_yaw_down:=1.30 \
  -p right_yaw_up:=-1.74 \
  -p left_elbow_straight:=1.57 \
  -p right_elbow_straight:=1.57 \
  -p elbow_gain:=1.55 \
  -p elbow_bend_deadzone:=0.015 \
  -p elbow_bend_response_gain:=1.75 \
  -p use_calibrated_elbow_bias:=true \
  -p max_joint_step:=0.180 \
  -p yaw_max_step:=0.450 \
  -p elbow_max_step:=0.320 \
  -p standard_z_sign:=1.0
