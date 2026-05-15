#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-h1_camera_pipeline}"

echo "============================================================"
echo " LAPTOP 3: START CALIBRATION"
echo "============================================================"
echo "CONTAINER_NAME: $CONTAINER_NAME"
echo "TOPIC:          /upper_body/start_calibration"
echo "COUNTDOWN:      3 sec"
echo "============================================================"
echo
echo "Встань перед камерой в нейтральную позу."
echo "Калибровка начнётся через 3 секунды."
echo

for i in 3 2 1; do
  echo "$i..."
  sleep 1
done

echo "START CALIBRATION"

docker exec -it "$CONTAINER_NAME" bash -lc '
set -e

cd /workspace

source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash

export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0

ros2 topic pub --once /upper_body/start_calibration std_msgs/msg/Bool "{data: true}"
'
