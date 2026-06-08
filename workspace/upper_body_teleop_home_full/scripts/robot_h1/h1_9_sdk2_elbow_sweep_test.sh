#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."
source scripts/robot_h1/h1_env.sh

echo "============================================================"
echo " H1 9: SDK2 DIRECT ELBOW SWEEP TEST"
echo "============================================================"
echo "UNITREE_NET_IFACE: $UNITREE_NET_IFACE"
echo "UNITREE_DOMAIN_ID: $UNITREE_DOMAIN_ID"
echo "This test bypasses ROS sender and publishes rt/lowcmd directly."
echo "============================================================"

echo "Stopping ROS sender / pose players..."
pkill -f "upper_body_teleop_runtime.h1_sdk2py_upper_body_sender" 2>/dev/null || true
pkill -f "upper_body_teleop_runtime.photo_pose_player_15" 2>/dev/null || true
pkill -f "upper_body_teleop_runtime.sdk2_photo_pose_player_15_direct" 2>/dev/null || true
pkill -f "upper_body_teleop_runtime.sdk2_h1_elbow_sweep_test" 2>/dev/null || true
sleep 1

python3 -m upper_body_teleop_runtime.sdk2_h1_elbow_sweep_test
