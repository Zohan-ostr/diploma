#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

CAMERA_ID="${1:-0}"

export ROS_DOMAIN_ID_VALUE="42"
export ROS_LOCALHOST_ONLY_VALUE="0"
export VIDEO_DEVICE="${VIDEO_DEVICE:-/dev/video${CAMERA_ID}}"

xhost +local:docker >/dev/null 2>&1 || true

bash scripts/home_run_camera.sh "$CAMERA_ID"
