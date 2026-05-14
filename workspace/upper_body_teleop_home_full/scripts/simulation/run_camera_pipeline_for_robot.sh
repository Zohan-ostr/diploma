#!/usr/bin/env bash
set -euo pipefail

CAMERA_ID="${1:-0}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

echo "============================================================"
echo " RUN CAMERA PIPELINE FOR REAL H1 ROBOT"
echo "============================================================"
echo "PROJECT_DIR: $PROJECT_DIR"
echo "CAMERA_ID:   $CAMERA_ID"
echo "ROS_DOMAIN:  0"
echo "============================================================"

xhost +local:docker >/dev/null 2>&1 || true

ROS_DOMAIN_ID_VALUE=0 \
ROS_LOCALHOST_ONLY_VALUE=0 \
bash scripts/home_run_camera.sh "$CAMERA_ID"
