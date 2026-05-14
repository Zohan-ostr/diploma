#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-h1_camera_pipeline}"
ROBOT_IP="${ROBOT_IP:-192.168.123.162}"
ROBOT_PORT="${ROBOT_PORT:-50051}"
INPUT_TOPIC="${INPUT_TOPIC:-/upper_body/command_geom}"

echo "============================================================"
echo " RUN LAPTOP UDP FORWARDER TO H1 ROBOT"
echo "============================================================"
echo "CONTAINER_NAME: $CONTAINER_NAME"
echo "INPUT_TOPIC:    $INPUT_TOPIC"
echo "ROBOT_IP:       $ROBOT_IP"
echo "ROBOT_PORT:     $ROBOT_PORT"
echo "============================================================"

docker exec -it "$CONTAINER_NAME" bash -lc "
set -e
cd /workspace
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

echo 'Checking command topic...'
ros2 topic info $INPUT_TOPIC -v || true

python3 scripts/simulation/laptop_upper_body_udp_forwarder.py \
  --ros-args \
  -p input_topic:=$INPUT_TOPIC \
  -p robot_ip:=$ROBOT_IP \
  -p robot_port:=$ROBOT_PORT
"
