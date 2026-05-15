#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-h1_camera_pipeline}"

echo "============================================================"
echo " LAPTOP 2: START RETARGET FOR REAL H1"
echo "============================================================"
echo "CONTAINER_NAME: $CONTAINER_NAME"
echo "INPUT:          /pose/landmarks"
echo "OUTPUT:         /upper_body/command_geom"
echo "ROS_DOMAIN_ID:  42"
echo "============================================================"

for i in $(seq 1 60); do
  if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    break
  fi

  if [ "$i" -eq 60 ]; then
    echo "ERROR: camera container is not running: $CONTAINER_NAME"
    echo "Start camera first:"
    echo "  bash scripts/robot_h1/laptop_1_start_camera.sh 0"
    exit 1
  fi

  sleep 1
done

docker exec -it "$CONTAINER_NAME" bash -lc '
set -e

cd /workspace

source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash

export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0

pkill -f "sim_geometric_retarget_old_with_tpose_yaw.py" || true
pkill -f "sim_geometric_retarget_ik2.py" || true

python3 /workspace/scripts/simulation/sim_geometric_retarget_old_with_tpose_yaw.py \
  --ros-args \
  -p input_topic:=/pose/landmarks \
  -p output_topic:=/upper_body/command_geom \
  -p calibration_frames:=45 \
  -p max_joint_step:=0.090 \
  -p elbow_max_step:=0.180 \
  -p yaw_max_step:=0.300 \
  -p joint_alpha:=0.45 \
  -p map_x_from_z:=1.0 \
  -p pitch_gain:=0.45 \
  -p roll_gain:=1.00 \
  -p elbow_gain:=1.35 \
  -p left_yaw_up:=1.89 \
  -p right_yaw_up:=-1.89 \
  -p left_yaw_down:=-1.30 \
  -p right_yaw_down:=1.30 \
  -p yaw_hysteresis:=0.045 \
  -p forward_yaw_threshold:=1.50 \
  -p forward_yaw_blend_width:=0.20
'
