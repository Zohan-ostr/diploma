#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-h1_camera_pipeline}"

echo "============================================================"
echo " SIM 4: START RETARGET + ABSOLUTE IK + FAST YAW"
echo "============================================================"
echo "CONTAINER_NAME: $CONTAINER_NAME"
echo "SCRIPT:         /workspace/scripts/simulation/sim_geometric_retarget_old_with_tpose_yaw.py"
echo "INPUT:          /pose/landmarks"
echo "OUTPUT:         /upper_body/command_geom"
echo "ROS_DOMAIN_ID:  42"
echo "============================================================"

echo "Waiting for container: $CONTAINER_NAME"
for i in $(seq 1 60); do
  if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "Container is running."
    break
  fi

  if [ "$i" -eq 60 ]; then
    echo "ERROR: container is not running: $CONTAINER_NAME"
    echo "Start camera first:"
    echo "  bash scripts/simulation/sim_2_start_camera.sh 0"
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

python3 /workspace/scripts/simulation/sim_geometric_retarget_old_with_tpose_yaw.py \
  --ros-args \
  -p input_topic:=/pose/landmarks \
  -p output_topic:=/upper_body/command_geom \
  -p tpose_hold_sec:=4.0 \
  -p calibration_frames:=45 \
  -p max_joint_step:=0.045 \
  -p elbow_max_step:=0.090 \
  -p yaw_max_step:=0.180 \
  -p map_x_from_z:=1.0 \
  -p pitch_gain:=0.45 \
  -p roll_gain:=1.00 \
  -p elbow_gain:=1.35 \
  -p left_yaw_down:=-1.25 \
  -p right_yaw_down:=1.25 \
  -p yaw_hysteresis:=0.045
'
