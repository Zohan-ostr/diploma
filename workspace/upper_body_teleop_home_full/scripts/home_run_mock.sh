#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
xhost +local:docker >/dev/null || true
docker compose -f compose.home.yaml run --rm home-dev bash -lc '
  set -e
  source /opt/ros/humble/setup.bash
  cd /workspace
  colcon build
  source install/setup.bash
  ros2 launch home_pipeline home_mock.launch.py
'
