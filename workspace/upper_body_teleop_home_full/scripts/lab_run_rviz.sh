#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
CAMERA_ID="${1:-0}"
VIDEO_DEVICE="/dev/video${CAMERA_ID}"
if [ ! -e "$VIDEO_DEVICE" ]; then
  echo "Camera device not found: $VIDEO_DEVICE"
  ls -l /dev/video* 2>/dev/null || true
  exit 1
fi
xhost +local:docker
export VIDEO_DEVICE="$VIDEO_DEVICE"
docker compose -f compose/compose.lab.yaml run --rm lab-dev bash -lc "
  source /opt/ros/jazzy/setup.bash &&
  cd /workspace &&
  colcon build &&
  source install/setup.bash &&
  ros2 launch home_pipeline rviz_camera.launch.py camera_index:=${CAMERA_ID}
"
