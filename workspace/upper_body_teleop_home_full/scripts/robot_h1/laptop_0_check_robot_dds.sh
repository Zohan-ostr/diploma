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
echo " LAPTOP 0: CHECK REAL H1 DDS"
echo "============================================================"
echo "CONTAINER_NAME: $CONTAINER_NAME"
echo "ROBOT_IP:       $ROBOT_IP"
echo "ROBOT_IFACE:    $ROBOT_NET_IFACE"
echo "ROS_DOMAIN_ID:  0"
echo "============================================================"

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

echo
echo "===== topic list ====="
ros2 topic list | grep -E "lowstate|lowcmd|arm_sdk|sport|loco|upper_body" || true

echo
echo "===== /lowstate info ====="
ros2 topic info /lowstate -v || true

echo
echo "===== /lowstate hz, 5 sec ====="
timeout 5 ros2 topic hz /lowstate || true
'
