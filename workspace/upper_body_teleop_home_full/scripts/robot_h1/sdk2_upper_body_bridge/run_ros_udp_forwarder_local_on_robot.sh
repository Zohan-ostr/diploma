#!/usr/bin/env bash
set -euo pipefail

cd "${HOME}/WS_OZ/diploma"

source /opt/ros/foxy/setup.bash
source /home/unitree/unitree_ros2/cyclonedds_ws/install/setup.bash
source install/setup.bash

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
export ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-0}

ros2 run sdk2_h1_upper_body_bridge ros_upper_body_udp_forwarder \
  --ros-args \
  -p input_topic:=${INPUT_TOPIC:-/upper_body/command_geom} \
  -p udp_host:=${UDP_HOST:-127.0.0.1} \
  -p udp_port:=${UDP_PORT:-50051}
