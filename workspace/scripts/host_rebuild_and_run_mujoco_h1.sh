#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/diploma/workspace/upper_body_teleop_home_full}"
COMPOSE_FILE="${COMPOSE_FILE:-compose/compose.home.yaml}"
SERVICE="${SERVICE:-home-dev}"
CONTAINER_NAME="${CONTAINER_NAME:-h1_mujoco_sim}"

# Интерфейс, который использует MuJoCo DDS.
# У тебя сейчас рабочий вариант: wlp0s20f3
MUJOCO_NET_IFACE="${MUJOCO_NET_IFACE:-wlp0s20f3}"

cd "$PROJECT_DIR"

echo "PROJECT_DIR:       $PROJECT_DIR"
echo "COMPOSE_FILE:      $COMPOSE_FILE"
echo "SERVICE:           $SERVICE"
echo "CONTAINER_NAME:    $CONTAINER_NAME"
echo "MUJOCO_NET_IFACE:  $MUJOCO_NET_IFACE"
echo

echo "==> Allow Docker containers to use X11..."
xhost +local:docker >/dev/null || true

echo "==> Stop old container if exists..."
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "==> Build image without cache..."
docker compose -f "$COMPOSE_FILE" build "$SERVICE" #--no-cache

echo
echo "==> Start container and run MuJoCo H1 simulation..."
echo "    Container name: $CONTAINER_NAME"
echo "    To stop: Ctrl+C, then docker rm -f $CONTAINER_NAME"
echo

docker compose -f "$COMPOSE_FILE" run \
  --name "$CONTAINER_NAME" \
  "$SERVICE" \
  bash -lc "
    set -e

    cd /workspace

    source /opt/ros/humble/setup.bash
    source /workspace/install/setup.bash 2>/dev/null || true

    export ROS_DOMAIN_ID=42
    export ROS_LOCALHOST_ONLY=0
    export MUJOCO_NET_IFACE='$MUJOCO_NET_IFACE'

    echo 'Inside container:'
    echo '  ROS_DOMAIN_ID='\"\$ROS_DOMAIN_ID\"
    echo '  ROS_LOCALHOST_ONLY='\"\$ROS_LOCALHOST_ONLY\"
    echo '  MUJOCO_NET_IFACE='\"\$MUJOCO_NET_IFACE\"
    echo

    bash scripts/sim_run_mujoco_h1.sh
  "
