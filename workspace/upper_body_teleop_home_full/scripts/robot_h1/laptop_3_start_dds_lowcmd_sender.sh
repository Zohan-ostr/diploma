#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-h1_camera_pipeline}"
ROBOT_IP="${ROBOT_IP:-192.168.123.162}"
ROBOT_NET_IFACE="${ROBOT_NET_IFACE:-}"

if [ -z "$ROBOT_NET_IFACE" ]; then
  ROBOT_NET_IFACE="$(ip route get "$ROBOT_IP" | awk "/dev/ {for(i=1;i<=NF;i++) if(\$i==\"dev\") print \$(i+1)}" | head -1)"
fi

if [ -z "$ROBOT_NET_IFACE" ]; then
  ROBOT_NET_IFACE="enx00e04c36022c"
fi

echo "============================================================"
echo " LAPTOP 3: START DDS LOWCMD SENDER TO REAL H1"
echo "============================================================"
echo "CONTAINER_NAME: $CONTAINER_NAME"
echo "INPUT:          /upper_body/command_geom"
echo "OUTPUT:         /lowcmd"
echo "LOWSTATE:       /lowstate"
echo "ROS_DOMAIN_ID:  0"
echo "ROBOT_IP:       $ROBOT_IP"
echo "ROBOT_IFACE:    $ROBOT_NET_IFACE"
echo "KP_ARM:         25.0"
echo "KD_ARM:         1.5"
echo "============================================================"
echo
echo "Это отправляет DDS /lowcmd на реального робота."
echo "Коэффициенты оставлены безопасными: KP_ARM=25.0 KD_ARM=1.5"
echo

read -r -p "Type YES to start DDS LowCmd sender: " CONFIRM
if [ "$CONFIRM" != "YES" ]; then
  echo "Abort."
  exit 0
fi

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

docker exec -it \
  -e ROBOT_IP="$ROBOT_IP" \
  -e ROBOT_NET_IFACE="$ROBOT_NET_IFACE" \
  "$CONTAINER_NAME" bash -lc '
set -e

cd /workspace

source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash

if [ -f /workspace/unitree_ros2/cyclonedds_ws/install/setup.bash ]; then
  source /workspace/unitree_ros2/cyclonedds_ws/install/setup.bash
fi

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

export CYCLONEDDS_URI="<CycloneDDS>
  <Domain>
    <General>
      <Interfaces>
        <NetworkInterface name=\"${ROBOT_NET_IFACE}\" priority=\"default\" multicast=\"default\" />
      </Interfaces>
      <AllowMulticast>true</AllowMulticast>
    </General>
    <Discovery>
      <Peers>
        <Peer Address=\"${ROBOT_IP}\"/>
      </Peers>
    </Discovery>
  </Domain>
</CycloneDDS>"

echo "Checking robot DDS topics..."
ros2 topic list | grep -E "lowstate|lowcmd|arm_sdk|upper_body" || true

echo
echo "Checking /lowstate info..."
ros2 topic info /lowstate -v || true

echo
echo "Starting sender..."
python3 /workspace/scripts/robot_h1/laptop_dds_upper_body_to_lowcmd.py \
  --ros-args \
  -p input_topic:=/upper_body/command_geom \
  -p lowcmd_topic:=/lowcmd \
  -p lowstate_topic:=/lowstate \
  -p rate_hz:=250.0 \
  -p kp_arm:=25.0 \
  -p kd_arm:=1.5 \
  -p max_step_rad:=0.012 \
  -p yaw_max_step_rad:=0.020 \
  -p elbow_max_step_rad:=0.025 \
  -p command_timeout_sec:=0.35
'
