#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

CONTAINER_NAME="${CONTAINER_NAME:-h1_camera_pipeline}"
ROS_DOMAIN_ID_VALUE="${ROS_DOMAIN_ID_VALUE:-0}"
ROS_LOCALHOST_ONLY_VALUE="${ROS_LOCALHOST_ONLY_VALUE:-0}"
UDP_POSE_PORT="${UDP_POSE_PORT:-50060}"

echo "============================================================"
echo " LAPTOP 1: START REALSENSE POSE FOR REAL H1"
echo "============================================================"
echo "CONTAINER_NAME:          $CONTAINER_NAME"
echo "ROS_DOMAIN_ID_VALUE:     $ROS_DOMAIN_ID_VALUE"
echo "ROS_LOCALHOST_ONLY:      $ROS_LOCALHOST_ONLY_VALUE"
echo "UDP_POSE_PORT:           $UDP_POSE_PORT"
echo "RealSense profile:       RGB8 + Z16 640x480@30"
echo "============================================================"

xhost +local:docker >/dev/null 2>&1 || true

echo "Stopping old container if exists..."
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "Starting ROS container without old webcam node..."
docker compose -f compose/compose.home.yaml run -d \
  --name "$CONTAINER_NAME" \
  -e DISPLAY="$DISPLAY" \
  -e QT_X11_NO_MITSHM=1 \
  -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID_VALUE" \
  -e ROS_LOCALHOST_ONLY="$ROS_LOCALHOST_ONLY_VALUE" \
  home-dev bash -lc "sleep infinity" >/dev/null

echo "Waiting for container..."
for i in $(seq 1 60); do
  if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    break
  fi

  if [ "$i" -eq 60 ]; then
    echo "ERROR: container did not start"
    exit 1
  fi

  sleep 1
done

echo "Starting UDP -> ROS pose publisher inside container..."
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

echo
echo "Starting host RealSense + MediaPipe window..."
echo "Press q in the video window to stop."
echo

python3 scripts/robot_h1/realsense_mediapipe_to_udp.py \
  --width 640 \
  --height 480 \
  --fps 30 \
  --udp_host 127.0.0.1 \
  --udp_port "$UDP_POSE_PORT"
