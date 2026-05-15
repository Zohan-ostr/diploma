#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-h1_camera_pipeline}"
ROBOT_IP="${ROBOT_IP:-192.168.123.162}"
ROBOT_PORT="${ROBOT_PORT:-50051}"

echo "============================================================"
echo " LAPTOP 4: START UDP FORWARDER TO REAL H1"
echo "============================================================"
echo "CONTAINER_NAME: $CONTAINER_NAME"
echo "INPUT:          /upper_body/command_geom"
echo "ROBOT_IP:       $ROBOT_IP"
echo "ROBOT_PORT:     $ROBOT_PORT"
echo "ROS_DOMAIN_ID:  42"
echo "============================================================"

for i in $(seq 1 60); do
  if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    break
  fi

  if [ "$i" -eq 60 ]; then
    echo "ERROR: camera container is not running: $CONTAINER_NAME"
    exit 1
  fi

  sleep 1
done

docker exec -it "$CONTAINER_NAME" bash -lc "
set -e

cd /workspace

source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash 2>/dev/null || true

export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0

if [ -d /workspace/scripts/robot_h1/sdk2_upper_body_bridge ]; then
  colcon build \
    --base-paths src/upper_body_msgs scripts/robot_h1/sdk2_upper_body_bridge \
    --packages-select upper_body_msgs sdk2_h1_upper_body_bridge || true

  source /workspace/install/setup.bash
fi

pkill -f '[r]os_upper_body_udp_forwarder' || true

ros2 run sdk2_h1_upper_body_bridge ros_upper_body_udp_forwarder \
  --ros-args \
  -p input_topic:=/upper_body/command_geom \
  -p udp_host:=${ROBOT_IP} \
  -p udp_port:=${ROBOT_PORT}
"
