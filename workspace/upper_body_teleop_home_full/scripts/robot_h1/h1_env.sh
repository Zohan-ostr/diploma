#!/usr/bin/env bash

# ============================================================
# Common environment for real Unitree H1 launch scripts
# ============================================================

PROJECT_DIR="${PROJECT_DIR:-$HOME/diploma/workspace/upper_body_teleop_home_full}"

# Real H1 works in domain 0.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export UNITREE_DOMAIN_ID="${UNITREE_DOMAIN_ID:-0}"

# Real robot Ethernet interface.
# Override before launch if your interface name is different:
#   export UNITREE_NET_IFACE=enx...
export UNITREE_NET_IFACE="${UNITREE_NET_IFACE:-enx00e04c36022c}"

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"

export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/src/home_pipeline:$PROJECT_DIR/src/h1_robot_adapter:$PYTHONPATH"

source /opt/ros/humble/setup.bash 2>/dev/null || true
source "$PROJECT_DIR/install/setup.bash" 2>/dev/null || true
source "$PROJECT_DIR/install/upper_body_msgs/share/upper_body_msgs/local_setup.bash" 2>/dev/null || true

print_h1_env() {
  echo "[H1 ENV] PROJECT_DIR=$PROJECT_DIR"
  echo "[H1 ENV] UNITREE_NET_IFACE=$UNITREE_NET_IFACE"
  echo "[H1 ENV] UNITREE_DOMAIN_ID=$UNITREE_DOMAIN_ID"
  echo "[H1 ENV] ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
  echo "[H1 ENV] RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
}
