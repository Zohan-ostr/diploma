#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

ROS_DOMAIN_ID_VALUE=42 \
ROS_LOCALHOST_ONLY_VALUE=0 \
CONTAINER_NAME=h1_camera_pipeline \
UDP_POSE_PORT=50060 \
bash scripts/robot_h1/laptop_1_start_realsense_pose_dds.sh
