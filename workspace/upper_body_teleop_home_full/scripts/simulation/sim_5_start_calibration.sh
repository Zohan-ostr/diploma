#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SIM 5. ЗАПУСК КАЛИБРОВКИ
# ============================================================
#
# Назначение:
#   отправить сигнал в /upper_body/start_calibration.
#
# Что делает retarget после сигнала:
#   1. Просит оператора держать T-позу с прямыми руками.
#   2. Собирает несколько кадров.
#   3. Считает ложный угол сгиба локтя.
#   4. Сохраняет elbow_bend_bias.
#   5. После этого включает копирующее управление.
#
# Почему публикуем несколько раз:
#   одноразовое ros2 topic pub --once иногда теряется,
#   если publisher и subscriber не успели состыковаться.
#
# Что можно менять:
#   - COUNTDOWN;
#   - количество публикаций --times;
#   - частоту -r.
#
# Что нельзя менять:
#   - TOPIC=/upper_body/start_calibration;
#   - тип std_msgs/msg/Bool.
# ============================================================

CONTAINER_NAME="${CONTAINER_NAME:-h1_camera_pipeline}"
ROS_DOMAIN_ID_VALUE="${ROS_DOMAIN_ID_VALUE:-42}"
ROS_LOCALHOST_ONLY_VALUE="${ROS_LOCALHOST_ONLY_VALUE:-0}"
TOPIC="${TOPIC:-/upper_body/start_calibration}"
COUNTDOWN="${COUNTDOWN:-3}"

echo "============================================================"
echo " SIM 5: START CALIBRATION"
echo "============================================================"
echo "CONTAINER_NAME: $CONTAINER_NAME"
echo "TOPIC:          $TOPIC"
echo "COUNTDOWN:      $COUNTDOWN sec"
echo "PUBLISH:        10 Hz for 3 sec"
echo "============================================================"
echo
echo "Встань перед камерой в T-позу с прямыми руками."
echo "Калибровка начнётся через $COUNTDOWN секунды."
echo

for i in $(seq "$COUNTDOWN" -1 1); do
  echo "$i..."
  sleep 1
done

echo "START CALIBRATION"

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

ros2 topic pub --times 30 -r 10 $TOPIC std_msgs/msg/Bool '{data: true}'
"
