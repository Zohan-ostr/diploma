#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-h1_camera_pipeline}"
UDP_HOST="${UDP_HOST:-127.0.0.1}"
UDP_PORT="${UDP_PORT:-50051}"

echo "============================================================"
echo " LAPTOP 3: START ROS -> UDP FORWARDER"
echo "============================================================"
echo "CONTAINER_NAME: $CONTAINER_NAME"
echo "INPUT:          /upper_body/command_geom"
echo "UDP:            $UDP_HOST:$UDP_PORT"
echo "============================================================"

docker exec -it "$CONTAINER_NAME" bash -lc "
set -e

cd /workspace

source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash

export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

python3 /workspace/scripts/robot_h1/laptop_ros_to_udp_forwarder.py \
  --ros-args \
  -p input_topic:=/upper_body/command_geom \
  -p udp_host:=${UDP_HOST} \
  -p udp_port:=${UDP_PORT}
"
