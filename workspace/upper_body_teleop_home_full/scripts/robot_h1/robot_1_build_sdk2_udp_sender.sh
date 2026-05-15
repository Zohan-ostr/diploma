#!/usr/bin/env bash
set -euo pipefail

cd ~/WS_OZ/diploma

echo "============================================================"
echo " ROBOT 1: BUILD SDK2 UDP SENDER"
echo "============================================================"
echo "pwd: $(pwd)"
echo "============================================================"

source /opt/ros/foxy/setup.bash
source /home/unitree/unitree_ros2/cyclonedds_ws/install/setup.bash

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY=0

export LD_LIBRARY_PATH=~/WS_OZ/unitree_sdk2/thirdparty/lib/x86_64:/usr/local/lib:$LD_LIBRARY_PATH

chmod +x scripts/robot_h1/build_sdk2_upper_body_bridge_on_robot.sh || true

bash scripts/robot_h1/build_sdk2_upper_body_bridge_on_robot.sh
