#!/usr/bin/env bash
set -euo pipefail

CAMERA_ID="${1:-0}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

CONTAINER_NAME="${CONTAINER_NAME:-h1_camera_pipeline}"
ROS_DOMAIN_ID_VALUE="${ROS_DOMAIN_ID_VALUE:-42}"
ROS_LOCALHOST_ONLY_VALUE="${ROS_LOCALHOST_ONLY_VALUE:-0}"
UDP_POSE_PORT="${UDP_POSE_PORT:-50060}"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker compose -f compose/compose.home.yaml run -d \
  --name "$CONTAINER_NAME" \
  -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID_VALUE" \
  -e ROS_LOCALHOST_ONLY="$ROS_LOCALHOST_ONLY_VALUE" \
  home-dev bash -lc "sleep infinity" >/dev/null

for i in $(seq 1 60); do
  docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME" && break
  sleep 1
done

docker exec -d "$CONTAINER_NAME" bash -lc "
set -e
cd /workspace
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=$ROS_DOMAIN_ID_VALUE
export ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY_VALUE

python3 /workspace/scripts/robot_h1/udp_pose_landmarks_to_ros.py \
  --ros-args \
  -p udp_port:=$UDP_POSE_PORT \
  -p output_topic:=/pose/landmarks
"

python3 scripts/robot_h1/webcam_mediapipe_to_udp.py \
  --camera "$CAMERA_ID" \
  --width 640 \
  --height 480 \
  --fps 30 \
  --udp_host 127.0.0.1 \
  --udp_port "$UDP_POSE_PORT" \
  --mirror
