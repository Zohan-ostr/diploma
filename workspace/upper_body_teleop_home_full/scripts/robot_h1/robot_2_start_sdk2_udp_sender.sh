#!/usr/bin/env bash
set -euo pipefail

cd ~/WS_OZ/diploma

export NET_IFACE="${NET_IFACE:-eth0}"
export KP_ARM="${KP_ARM:-25.0}"
export KD_ARM="${KD_ARM:-1.5}"

# Для реального робота оставляем шаги умеренными.
# Ускорение retarget остаётся на ноутбуке, а здесь безопасный low-level sender.
export MAX_STEP_RAD="${MAX_STEP_RAD:-0.012}"
export ELBOW_MAX_STEP="${ELBOW_MAX_STEP:-0.060}"
export TIMEOUT_SEC="${TIMEOUT_SEC:-0.35}"
export UDP_PORT="${UDP_PORT:-50051}"

echo "============================================================"
echo " ROBOT 2: START SDK2 UDP LOWCMD SENDER"
echo "============================================================"
echo "pwd:            $(pwd)"
echo "NET_IFACE:      $NET_IFACE"
echo "KP_ARM:         $KP_ARM"
echo "KD_ARM:         $KD_ARM"
echo "MAX_STEP_RAD:   $MAX_STEP_RAD"
echo "ELBOW_MAX_STEP: $ELBOW_MAX_STEP"
echo "TIMEOUT_SEC:    $TIMEOUT_SEC"
echo "UDP_PORT:       $UDP_PORT"
echo "============================================================"
echo
echo "Это отправляет команды на реальные моторы рук H1."
echo "Убедись, что робот безопасно закреплён/готов."
echo

read -r -p "Type YES to start real robot sender: " CONFIRM

if [ "$CONFIRM" != "YES" ]; then
  echo "Abort."
  exit 0
fi

source /opt/ros/foxy/setup.bash
source /home/unitree/unitree_ros2/cyclonedds_ws/install/setup.bash

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY=0

export LD_LIBRARY_PATH=~/WS_OZ/unitree_sdk2/thirdparty/lib/x86_64:/usr/local/lib:$LD_LIBRARY_PATH

chmod +x scripts/robot_h1/run_sdk2_udp_sender_on_robot.sh || true

NET_IFACE="$NET_IFACE" \
KP_ARM="$KP_ARM" \
KD_ARM="$KD_ARM" \
MAX_STEP_RAD="$MAX_STEP_RAD" \
ELBOW_MAX_STEP="$ELBOW_MAX_STEP" \
TIMEOUT_SEC="$TIMEOUT_SEC" \
UDP_PORT="$UDP_PORT" \
bash scripts/robot_h1/run_sdk2_udp_sender_on_robot.sh
