#!/usr/bin/env bash
set -e

source ~/diploma/workspace/upper_body_teleop_home_full/scripts/robot_h1/h1_robot_env.sh

cd ~/diploma/workspace/upper_body_teleop_home_full

echo "============================================================"
echo " H1 1: START REALSENSE MEDIAPIPE NODE"
echo "============================================================"
echo "Publishes: /pose/landmarks"
echo

if ros2 pkg executables home_pipeline 2>/dev/null | grep -q 'realsense_mediapipe_node'; then
  echo "[H1 CAMERA] running via ros2 run"
  ros2 run home_pipeline realsense_mediapipe_node \
    --ros-args \
    -p width:=640 \
    -p height:=480 \
    -p fps:=30 \
    -p preview:=true \
    -p preview_mirror:=true \
    -p depth_window:=5 \
    -p min_depth_m:=0.15 \
    -p max_depth_m:=6.0
  exit 0
fi

echo "[H1 CAMERA] ros2 executable not found, running source file directly"

python3 src/home_pipeline/home_pipeline/realsense_mediapipe_node.py \
  --ros-args \
  -p width:=640 \
  -p height:=480 \
  -p fps:=30 \
  -p preview:=true \
  -p preview_mirror:=true \
  -p depth_window:=5 \
  -p min_depth_m:=0.15 \
  -p max_depth_m:=6.0
