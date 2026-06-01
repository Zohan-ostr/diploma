#!/usr/bin/env bash
set -e
source ~/diploma/workspace/upper_body_teleop_home_full/scripts/robot_h1/h1_robot_env.sh

ros2 topic pub --once /upper_body/start_calibration std_msgs/msg/Bool "{data: true}"
