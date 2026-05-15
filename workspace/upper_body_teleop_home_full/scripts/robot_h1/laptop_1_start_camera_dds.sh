#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

CAMERA_ID="${1:-0}"

source scripts/robot_h1/laptop_robot_dds_env.sh

export ROS_DOMAIN_ID_VALUE="$ROS_DOMAIN_ID"
export ROS_LOCALHOST_ONLY_VALUE="0"
export VIDEO_DEVICE="${VIDEO_DEVICE:-/dev/video${CAMERA_ID}}"

echo "============================================================"
echo " LAPTOP 1: START CAMERA FOR REAL H1 VIA DDS"
echo "============================================================"
echo "CAMERA_ID:       $CAMERA_ID"
echo "VIDEO_DEVICE:    $VIDEO_DEVICE"
echo "ROS_DOMAIN_ID:   $ROS_DOMAIN_ID"
echo "ROBOT_IP:        ${ROBOT_IP:-192.168.123.162}"
echo "ROBOT_NET_IFACE: $ROBOT_NET_IFACE"
echo "============================================================"

xhost +local:docker >/dev/null 2>&1 || true

bash scripts/home_run_camera.sh "$CAMERA_ID"
