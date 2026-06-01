#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SIM 6. ПРОВЕРКА СОСТОЯНИЯ СИМУЛЯЦИОННОГО КОНТУРА
# ============================================================
#
# Назначение:
#   быстро проверить, что основные ROS 2 topics существуют
#   и что retarget реально публикует команды.
#
# Проверяем:
#   /pose/landmarks
#   /upper_body/start_calibration
#   /upper_body/command_geom
#
# Если /pose/landmarks нет:
#   не запущен SIM 2 или сломан camera/UDP bridge.
#
# Если /upper_body/command_geom нет:
#   не запущен SIM 4 или retarget упал.
#
# Если калибровка не проходит:
#   проверить, есть ли subscriber у /upper_body/start_calibration.
# ============================================================

CONTAINER_NAME="${CONTAINER_NAME:-h1_camera_pipeline}"
ROS_DOMAIN_ID_VALUE="${ROS_DOMAIN_ID_VALUE:-42}"
ROS_LOCALHOST_ONLY_VALUE="${ROS_LOCALHOST_ONLY_VALUE:-0}"

echo "============================================================"
echo " SIM 6: CHECK SIMULATION TOPICS"
echo "============================================================"
echo "CONTAINER_NAME: $CONTAINER_NAME"
echo "ROS_DOMAIN_ID:  $ROS_DOMAIN_ID_VALUE"
echo "============================================================"

docker exec -it "$CONTAINER_NAME" bash -lc "
set -e

cd /workspace

export PYTHONPATH=/workspace/src:/workspace/src/home_pipeline:$PYTHONPATH

export PYTHONPATH=/workspace/src/upper_body_teleop_runtime:$PYTHONPATH

source /opt/ros/humble/setup.bash

if [ -f /workspace/install/upper_body_msgs/share/upper_body_msgs/local_setup.bash ]; then
  source /workspace/install/upper_body_msgs/share/upper_body_msgs/local_setup.bash
fi

export ROS_DOMAIN_ID=$ROS_DOMAIN_ID_VALUE
export ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY_VALUE

echo
echo '=== NODES ==='
ros2 node list || true

echo
echo '=== TOPICS ==='
ros2 topic list | sort | grep -E 'pose|upper_body|joint|lowcmd|lowstate' || true

echo
echo '=== /pose/landmarks info ==='
ros2 topic info /pose/landmarks -v || true

echo
echo '=== /upper_body/start_calibration info ==='
ros2 topic info /upper_body/start_calibration -v || true

echo
echo '=== /upper_body/command_geom info ==='
ros2 topic info /upper_body/command_geom -v || true

echo
echo '=== one command sample ==='
timeout 3 ros2 topic echo /upper_body/command_geom --once || true
"
