#!/usr/bin/env bash
set -e

source ~/diploma/workspace/upper_body_teleop_home_full/scripts/robot_h1/h1_robot_env.sh

echo "ROS topics:"
ros2 topic list | sort | grep -E 'pose|upper_body|lowcmd|lowstate|h1' || true

echo
echo "Pose landmarks:"
timeout 2 ros2 topic echo /pose/landmarks --once || true

echo
echo "Vector FABRIK command:"
timeout 2 ros2 topic echo /upper_body/command_geom --once || true
