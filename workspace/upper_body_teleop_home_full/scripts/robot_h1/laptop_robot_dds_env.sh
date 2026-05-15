#!/usr/bin/env bash

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY=0

ROBOT_IP="${ROBOT_IP:-192.168.123.162}"
ROBOT_NET_IFACE="${ROBOT_NET_IFACE:-}"

if [ -z "$ROBOT_NET_IFACE" ]; then
  ROBOT_NET_IFACE="$(ip route get "$ROBOT_IP" | awk '/dev/ {for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}' | head -1)"
fi

if [ -z "$ROBOT_NET_IFACE" ]; then
  ROBOT_NET_IFACE="enx00e04c36022c"
fi

export ROBOT_NET_IFACE

export CYCLONEDDS_URI="<CycloneDDS>
  <Domain>
    <General>
      <Interfaces>
        <NetworkInterface name=\"$ROBOT_NET_IFACE\" priority=\"default\" multicast=\"default\" />
      </Interfaces>
      <AllowMulticast>true</AllowMulticast>
    </General>
    <Discovery>
      <Peers>
        <Peer Address=\"$ROBOT_IP\"/>
      </Peers>
    </Discovery>
  </Domain>
</CycloneDDS>"
