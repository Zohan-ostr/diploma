#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-h1_camera_pipeline}"

echo "============================================================"
echo " SIM 5: CHECK CAMERA / RETARGET TOPICS"
echo "============================================================"
echo "CONTAINER_NAME: $CONTAINER_NAME"
echo "============================================================"

docker exec -it "$CONTAINER_NAME" bash -lc '
cd /workspace

source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash

export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0

echo
echo "===== nodes ====="
ros2 node list | grep -E "media|camera|retarget|geom|bridge|h1" || true

echo
echo "===== topics ====="
ros2 topic list | grep -E "pose|upper_body|lowcmd|lowstate" || true

echo
echo "===== /pose/landmarks info ====="
ros2 topic info /pose/landmarks -v || true

echo
echo "===== /upper_body/command_geom info ====="
ros2 topic info /upper_body/command_geom -v || true

echo
echo "===== /upper_body/command_geom one message ====="
timeout 5 ros2 topic echo /upper_body/command_geom --once || true
'
