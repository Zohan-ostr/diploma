#!/usr/bin/env bash
set -euo pipefail

cd "${HOME}/WS_OZ/diploma"

source /opt/ros/foxy/setup.bash
source /home/unitree/unitree_ros2/cyclonedds_ws/install/setup.bash

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
export ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-0}
export LD_LIBRARY_PATH=${HOME}/WS_OZ/unitree_sdk2/thirdparty/lib/x86_64:/usr/local/lib:${LD_LIBRARY_PATH:-}

rm -rf build/sdk2_h1_upper_body_bridge install/sdk2_h1_upper_body_bridge

colcon build \
  --base-paths src/upper_body_msgs scripts/robot_h1/sdk2_upper_body_bridge \
  --packages-select upper_body_msgs sdk2_h1_upper_body_bridge \
  --cmake-args -DPYTHON_EXECUTABLE=/usr/bin/python3
