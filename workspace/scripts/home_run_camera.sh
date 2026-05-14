#!/usr/bin/env bash
set -e

CAMERA_ID="${1:-0}"
VIDEO_DEVICE="/dev/video${CAMERA_ID}"

ROS_DOMAIN_ID_VALUE="${ROS_DOMAIN_ID_VALUE:-42}"
ROS_LOCALHOST_ONLY_VALUE="${ROS_LOCALHOST_ONLY_VALUE:-0}"

CONTAINER_NAME="${CONTAINER_NAME:-h1_camera_pipeline}"
COMPOSE_FILE="${COMPOSE_FILE:-compose/compose.home.yaml}"
SERVICE_NAME="${SERVICE_NAME:-home-dev}"

SKIP_BUILD="${SKIP_BUILD:-0}"

# ВАЖНО:
# BUILD_TARGETS означает "собрать пакет и все его зависимости".
# Для камеры обычно достаточно home_pipeline.
# Если нужен adapter в этом же install, добавь h1_robot_adapter.
BUILD_TARGETS="${BUILD_TARGETS:-home_pipeline h1_robot_adapter}"

LAUNCH_PACKAGE="${LAUNCH_PACKAGE:-home_pipeline}"
LAUNCH_FILE="${LAUNCH_FILE:-home_camera.launch.py}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "============================================================"
echo " HOME CAMERA PIPELINE"
echo "============================================================"
echo "PROJECT_DIR:              $PROJECT_DIR"
echo "COMPOSE_FILE:             $COMPOSE_FILE"
echo "SERVICE_NAME:             $SERVICE_NAME"
echo "CONTAINER_NAME:           $CONTAINER_NAME"
echo "CAMERA_ID:                $CAMERA_ID"
echo "VIDEO_DEVICE:             $VIDEO_DEVICE"
echo "ROS_DOMAIN_ID_VALUE:      $ROS_DOMAIN_ID_VALUE"
echo "ROS_LOCALHOST_ONLY_VALUE: $ROS_LOCALHOST_ONLY_VALUE"
echo "SKIP_BUILD:               $SKIP_BUILD"
echo "BUILD_TARGETS:            $BUILD_TARGETS"
echo "LAUNCH:                   $LAUNCH_PACKAGE $LAUNCH_FILE"
echo "============================================================"
echo

if [ ! -f "$COMPOSE_FILE" ]; then
  echo "ERROR: compose file not found: $COMPOSE_FILE"
  exit 1
fi

if [ ! -e "$VIDEO_DEVICE" ]; then
  echo "ERROR: camera device not found: $VIDEO_DEVICE"
  echo "Available video devices:"
  ls -l /dev/video* 2>/dev/null || true
  exit 1
fi

if command -v xhost >/dev/null 2>&1; then
  xhost +local:docker >/dev/null || true
fi

echo "Stopping old container if exists: $CONTAINER_NAME"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "Starting camera container..."
echo

docker compose -f "$COMPOSE_FILE" run \
  --rm \
  --name "$CONTAINER_NAME" \
  --use-aliases \
  -e DISPLAY="${DISPLAY:-:0}" \
  -e QT_X11_NO_MITSHM=1 \
  -e VIDEO_DEVICE="$VIDEO_DEVICE" \
  -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID_VALUE" \
  -e ROS_LOCALHOST_ONLY="$ROS_LOCALHOST_ONLY_VALUE" \
  "$SERVICE_NAME" \
  bash -lc "
    set -e

    cd /workspace

    set +u
    source /opt/ros/humble/setup.bash
    set -u

    export ROS_DOMAIN_ID='$ROS_DOMAIN_ID_VALUE'
    export ROS_LOCALHOST_ONLY='$ROS_LOCALHOST_ONLY_VALUE'

    echo '============================================================'
    echo ' INSIDE CAMERA CONTAINER'
    echo '============================================================'
    echo \"container: \$(hostname)\"
    echo \"pwd:       \$(pwd)\"
    echo \"camera:    $VIDEO_DEVICE\"
    echo \"ROS_DOMAIN_ID=\$ROS_DOMAIN_ID\"
    echo \"ROS_LOCALHOST_ONLY=\$ROS_LOCALHOST_ONLY\"
    echo '============================================================'
    echo

    echo 'Video devices inside container:'
    ls -l /dev/video* 2>/dev/null || true
    echo

    echo 'Workspace packages:'
    colcon list | grep -E 'home_pipeline|h1_robot_adapter|upper_body|unitree_go|unitree_api|unitree_hg' || true
    echo

    if [ '$SKIP_BUILD' = '0' ]; then
      echo 'Building workspace with dependencies...'
      echo 'Targets: $BUILD_TARGETS'
      colcon build --packages-up-to $BUILD_TARGETS
    else
      echo 'Skipping build.'
    fi

    if [ ! -f /workspace/install/setup.bash ]; then
      echo 'ERROR: /workspace/install/setup.bash not found.'
      echo 'Build probably failed or was skipped before first successful build.'
      exit 1
    fi

    set +u
    source /workspace/install/setup.bash
    set -u

    echo
    echo 'Installed relevant ROS packages:'
    ros2 pkg list | grep -E 'home_pipeline|h1_robot_adapter|upper_body|unitree_go|unitree_api|unitree_hg' || true
    echo

    echo 'Executables:'
    ros2 pkg executables home_pipeline || true
    ros2 pkg executables h1_robot_adapter || true
    echo

    echo 'Launching camera pipeline...'
    ros2 launch '$LAUNCH_PACKAGE' '$LAUNCH_FILE' camera_id:='$CAMERA_ID'
  "
