#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-h1_camera_pipeline}"

echo "============================================================"
echo " SIM 3B: START GEOMETRIC RETARGET"
echo "============================================================"
echo "CONTAINER_NAME: $CONTAINER_NAME"
echo "INPUT:          /pose/landmarks"
echo "OUTPUT:         /upper_body/command_geom"
echo "============================================================"

docker exec -it "$CONTAINER_NAME" bash -lc '
set -e

cd /workspace

source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash

export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0

echo "Killing old geometric retarget if exists..."
pkill -f "[h]1_geometric_retarget_node" || true

echo
echo "Available executables:"
ros2 pkg executables h1_robot_adapter | grep -E "geometric|retarget|bridge|h1" || true

echo
echo "Starting h1_geometric_retarget_node..."
ros2 run h1_robot_adapter h1_geometric_retarget_node \
  --ros-args \
  -p input_topic:=/pose/landmarks \
  -p output_topic:=/upper_body/command_geom \
  -p tpose_hold_sec:=4.0 \
  -p calibration_sec:=1.5 \
  -p max_joint_step_rad:=0.025 \
  -p yaw_switch_threshold_m:=0.045
'
