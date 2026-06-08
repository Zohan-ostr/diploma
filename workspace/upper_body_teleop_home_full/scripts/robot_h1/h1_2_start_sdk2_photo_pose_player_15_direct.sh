#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."
source scripts/robot_h1/h1_env.sh

echo "============================================================"
echo " H1 DIRECT SDK2 PHOTO POSE PLAYER 15"
echo "============================================================"
echo "UNITREE_NET_IFACE: $UNITREE_NET_IFACE"
echo "UNITREE_DOMAIN_ID: $UNITREE_DOMAIN_ID"
echo "NO ROS sender. Direct rt/lowcmd via unitree_sdk2py."
echo "CONTROL: [ previous, ] next, e elbow test, r T-pose, q quit"
echo "============================================================"

echo "Stopping ROS sender / old pose players..."
pkill -f "upper_body_teleop_runtime.h1_sdk2py_upper_body_sender" 2>/dev/null || true
pkill -f "upper_body_teleop_runtime.photo_pose_player_15" 2>/dev/null || true
pkill -f "upper_body_teleop_runtime.sdk2_photo_pose_player_15_direct" 2>/dev/null || true
sleep 1

python3 -m upper_body_teleop_runtime.sdk2_photo_pose_player_15_direct
