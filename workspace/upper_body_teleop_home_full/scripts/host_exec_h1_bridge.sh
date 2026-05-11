#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-h1_mujoco_sim}"

ROS_DOMAIN_ID_VALUE="${ROS_DOMAIN_ID_VALUE:-42}"

INPUT_TOPIC="${INPUT_TOPIC:-/upper_body/command_geom}"

UNITREE_NET_IFACE="${UNITREE_NET_IFACE:-wlp0s20f3}"
UNITREE_SDK_DOMAIN="${UNITREE_SDK_DOMAIN:-1}"

ARM_VELOCITY_LIMIT="${ARM_VELOCITY_LIMIT:-100.0}"
KP_LOW="${KP_LOW:-400.0}"
KD_LOW="${KD_LOW:-8.0}"
KP_HIGH="${KP_HIGH:-300.0}"
KD_HIGH="${KD_HIGH:-5.0}"
COMMAND_TIMEOUT_SEC="${COMMAND_TIMEOUT_SEC:-0.5}"
CONTROL_HZ="${CONTROL_HZ:-250.0}"
TEST_MODE="${TEST_MODE:-none}"

CID="$(docker ps --filter "name=^/${CONTAINER_NAME}$" --format "{{.ID}}" | head -n 1)"

if [ -z "$CID" ]; then
  echo "ERROR: running container '$CONTAINER_NAME' not found."
  docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"
  exit 1
fi

echo "Found container:"
docker ps --filter "id=$CID" --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"

echo
echo "Starting clean bridge:"
echo "  INPUT_TOPIC:          $INPUT_TOPIC"
echo "  UNITREE_NET_IFACE:    $UNITREE_NET_IFACE"
echo "  UNITREE_SDK_DOMAIN:   $UNITREE_SDK_DOMAIN"
echo "  ARM_VELOCITY_LIMIT:   $ARM_VELOCITY_LIMIT"
echo "  KP_LOW/KD_LOW:        $KP_LOW / $KD_LOW"
echo "  COMMAND_TIMEOUT_SEC:  $COMMAND_TIMEOUT_SEC"
echo "  TEST_MODE:            $TEST_MODE"
echo

echo "Killing old bridge/controller processes..."
docker exec "$CID" bash -lc "
  pkill -f '[h]1_sdk2py_upper_body_bridge' || true
  pkill -f '[h]1_unitree_style_arm_controller' || true
  pkill -f '[a]rm_sdk_to_lowcmd_bridge' || true
"

docker exec -it "$CID" bash -lc "
  set -e

  cd /workspace

  source /opt/ros/humble/setup.bash
  source /workspace/install/setup.bash 2>/dev/null || true

  export ROS_DOMAIN_ID='$ROS_DOMAIN_ID_VALUE'
  export ROS_LOCALHOST_ONLY=0

  export INPUT_TOPIC='$INPUT_TOPIC'
  export UNITREE_NET_IFACE='$UNITREE_NET_IFACE'
  export UNITREE_SDK_DOMAIN='$UNITREE_SDK_DOMAIN'
  export ARM_VELOCITY_LIMIT='$ARM_VELOCITY_LIMIT'
  export KP_LOW='$KP_LOW'
  export KD_LOW='$KD_LOW'
  export KP_HIGH='$KP_HIGH'
  export KD_HIGH='$KD_HIGH'
  export COMMAND_TIMEOUT_SEC='$COMMAND_TIMEOUT_SEC'
  export CONTROL_HZ='$CONTROL_HZ'
  export TEST_MODE='$TEST_MODE'

  colcon build --symlink-install --packages-select h1_robot_adapter
  source /workspace/install/setup.bash

  ros2 run h1_robot_adapter h1_sdk2py_upper_body_bridge --ros-args \
    -p input_topic:='$INPUT_TOPIC' \
    -p arm_velocity_limit:='$ARM_VELOCITY_LIMIT' \
    -p kp_arm:='$KP_LOW' \
    -p kd_arm:='$KD_LOW' \
    -p kp_body:='$KP_HIGH' \
    -p kd_body:='$KD_HIGH' \
    -p command_timeout_sec:='$COMMAND_TIMEOUT_SEC' \
    -p control_hz:='$CONTROL_HZ' \
    -p test_mode:='$TEST_MODE'
"
