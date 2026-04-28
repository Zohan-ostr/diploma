#!/usr/bin/env bash
set -e

CAMERA_ID="${1:-0}"
VIDEO_DEVICE="/dev/video${CAMERA_ID}"

if [ ! -e "$VIDEO_DEVICE" ]; then
  echo "Camera device not found: $VIDEO_DEVICE"
  echo "Available video devices:"
  ls -l /dev/video* 2>/dev/null || true
  exit 1
fi

xhost +local:docker

export VIDEO_DEVICE="$VIDEO_DEVICE"

docker compose -f compose/compose.home.yaml run --rm home-dev bash -lc "
  source /opt/ros/humble/setup.bash &&
  cd /workspace &&
  colcon build &&
  source install/setup.bash &&
  ros2 launch home_pipeline home_camera.launch.py camera_id:=${CAMERA_ID}
"
