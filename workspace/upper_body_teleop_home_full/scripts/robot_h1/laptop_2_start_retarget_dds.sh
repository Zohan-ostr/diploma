#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-h1_camera_pipeline}"

echo "============================================================"
echo " LAPTOP 2: START RETARGET FOR REAL H1 VIA DDS"
echo "============================================================"
echo "CONTAINER_NAME: $CONTAINER_NAME"
echo "INPUT:          /pose/landmarks"
echo "OUTPUT:         /upper_body/command_geom"
echo "ROS_DOMAIN_ID:  0"
echo "============================================================"

for i in $(seq 1 60); do
  if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    break
  fi

  if [ "$i" -eq 60 ]; then
    echo "ERROR: camera container is not running: $CONTAINER_NAME"
    exit 1
  fi

  sleep 1
done

docker exec -it "$CONTAINER_NAME" bash -lc '
set -e

cd /workspace

source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

echo "============================================================"
echo " INSIDE CONTAINER: START REAL-H1 RETARGET"
echo "============================================================"
echo "pwd:           $(pwd)"
echo "ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo "============================================================"

echo
echo "Checking /pose/landmarks:"
ros2 topic info /pose/landmarks -v || true

echo
echo "Stopping old retarget processes safely..."
python3 - <<PY
import os
import signal
import subprocess

me = os.getpid()
out = subprocess.check_output(["ps", "-eo", "pid=,comm=,args="], text=True)

targets = [
    "sim_geometric_retarget_old_with_tpose_yaw.py",
    "sim_geometric_retarget_ik2.py",
]

for line in out.splitlines():
    parts = line.strip().split(None, 2)
    if len(parts) < 3:
        continue

    pid_s, comm, args = parts
    pid = int(pid_s)

    if pid == me:
        continue

    is_python = comm.startswith("python")
    is_target = is_python and any(t in args for t in targets)

    if is_target:
        print(f"killing pid={pid} args={args}")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
PY

sleep 1

echo
echo "Starting RealSense two-stage IK retarget:"
echo "  /pose/landmarks -> IK -> /upper_body/command_geom"
echo

python3 /workspace/scripts/robot_h1/realsense_ik_upper_body_retarget.py \
  --ros-args \
  -p input_topic:=/pose/landmarks \
  -p output_topic:=/upper_body/command_geom \
  -p calibration_topic:=/upper_body/start_calibration \
  -p calibration_frames:=45 \
  -p max_joint_step:=0.180 \
  -p elbow_max_step:=0.320 \
  -p yaw_max_step:=0.450 \
  -p joint_alpha:=0.75 \
  -p pitch_gain:=1.00 \
  -p invert_pitch_forward_above_shoulder:=true \
  -p invert_yaw_forward_above_shoulder:=true \
  -p invert_fabrik_target_forward_above_shoulder:=true \
  -p elbow_gain:=1.55 \
  -p elbow_bend_deadzone:=0.015 \
  -p elbow_bend_response_gain:=1.75 \
  -p use_calibrated_elbow_bias:=true \
  -p left_elbow_straight:=1.57 \
  -p right_elbow_straight:=1.57 \
  -p elbow_deadzone_rad:=0.20 \
  -p left_yaw_up:=1.74 \
  -p right_yaw_up:=-1.74 \
  -p left_yaw_down:=-1.30 \
  -p right_yaw_down:=1.30 \
  -p standard_z_sign:=1.0 \
  -p use_standard_body_frame:=true \
  -p locked_right_yaw:=-1.74 \
  -p locked_left_yaw:=1.74 \
  -p lock_shoulder_yaw:=true \
  -p elbow_only_debug:=false \
  -p robot_upper_len:=0.31 \
  -p robot_fore_len:=0.31 \
  -p fabrik_wrist_weight:=1.00 \
  -p fabrik_position_weight:=2.00