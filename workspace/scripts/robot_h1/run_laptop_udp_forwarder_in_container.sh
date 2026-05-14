#!/usr/bin/env bash
set -euo pipefail

cd /workspace
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash

export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
export ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-0}

python3 scripts/robot_h1/laptop_upper_body_udp_forwarder.py \
  --ros-args \
  -p input_topic:=${INPUT_TOPIC:-/upper_body/command_geom} \
  -p robot_ip:=${ROBOT_IP:-192.168.123.162} \
  -p robot_port:=${ROBOT_PORT:-50051}
