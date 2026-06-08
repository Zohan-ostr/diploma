#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-h1_camera_pipeline}"
INPUT_TOPIC="${INPUT_TOPIC:-/pose/landmarks}"
OUTPUT_TOPIC="${OUTPUT_TOPIC:-/upper_body/command_geom}"
CALIBRATION_TOPIC="${CALIBRATION_TOPIC:-/upper_body/start_calibration}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

echo "============================================================"
echo " SIM 3: START PROJECTION RETARGET"
echo "============================================================"
echo "CONTAINER_NAME:       $CONTAINER_NAME"
echo "INPUT_TOPIC:          $INPUT_TOPIC"
echo "OUTPUT_TOPIC:         $OUTPUT_TOPIC"
echo "CALIBRATION_TOPIC:    $CALIBRATION_TOPIC"
echo "ROS_DOMAIN_ID:        $ROS_DOMAIN_ID"
echo "PITCH/ROLL:           two-stage segmented FK search"
echo "ELBOW:                angle between arm vectors"
echo "YAW:                  two-stage segmented FK search"
echo "============================================================"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "ERROR: container is not running: $CONTAINER_NAME"
  echo "Start camera container first:"
  echo "  bash scripts/simulation/sim_2_start_camera.sh 0"
  exit 1
fi

docker exec -it "$CONTAINER_NAME" bash -lc "
set -e

cd /workspace

source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash 2>/dev/null || true

export PYTHONPATH=/workspace/src:/workspace/src/home_pipeline:/workspace/src/h1_robot_adapter:\$PYTHONPATH
export ROS_DOMAIN_ID=$ROS_DOMAIN_ID
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

echo
echo 'Stopping old retarget python nodes safely...'

python3 - <<'PY'
import os
import signal

targets = [
    'upper_body_teleop_runtime.vector_fabrik_retarget',
    'upper_body_teleop_runtime.vector_projection_retarget_sim',
]

self_pid = os.getpid()
parent_pid = os.getppid()

for pid_s in os.listdir('/proc'):
    if not pid_s.isdigit():
        continue

    pid = int(pid_s)

    if pid in (self_pid, parent_pid):
        continue

    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as f:
            parts = [x.decode(errors='ignore') for x in f.read().split(b'\\0') if x]
    except Exception:
        continue

    if not parts:
        continue

    joined = ' '.join(parts)

    # Убиваем только Python-процессы retarget, не bash -lc.
    is_python = 'python' in os.path.basename(parts[0])
    is_target = any(t in joined for t in targets)

    if is_python and is_target:
        try:
            print('killing:', pid, joined)
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception as e:
            print('kill failed:', pid, e)
PY

sleep 1

echo
echo 'Checking module...'
python3 -m py_compile /workspace/src/upper_body_teleop_runtime/vector_projection_retarget_sim.py

echo
echo 'Starting projection retarget...'
echo

python3 -m upper_body_teleop_runtime.vector_projection_retarget_sim \
  --ros-args \
  -p input_topic:=$INPUT_TOPIC \
  -p output_topic:=$OUTPUT_TOPIC \
  -p calibration_topic:=$CALIBRATION_TOPIC \
  -p calibration_frames:=45 \
  -p calibration_duration_sec:=2.0 \
  -p min_visibility:=0.01 \
  -p landmark_alpha:=0.35 \
  -p joint_alpha:=0.55 \
  -p robot_upper_len:=0.31 \
  -p robot_fore_len:=0.31 \
  -p segment_pr_segments:=10 \
  -p segment_pr_continuity_weight:=0.002 \
  -p projection_pitch_min_norm:=0.12 \
  -p projection_pitch_full_norm:=0.35 \
  -p projection_pitch_gain:=1.0 \
  -p projection_roll_gain:=1.0 \
  -p yaw_grid:=81 \
  -p yaw_direction_weight:=1.0 \
  -p yaw_wrist_position_weight:=4.0 \
  -p yaw_refine_window:=0.18 \
  -p yaw_refine_grid:=17 \
  -p yaw_continuity_weight:=0.05 \
  -p segment_yaw_segments:=10 \
  -p segment_yaw_continuity_weight:=0.01 \
  -p left_yaw_down:=-1.30 \
  -p left_yaw_up:=1.74 \
  -p right_yaw_down:=1.30 \
  -p right_yaw_up:=-1.74 \
  -p left_elbow_straight:=1.57 \
  -p right_elbow_straight:=1.57 \
  -p elbow_gain:=1.35 \
  -p elbow_bend_deadzone:=0.015 \
  -p elbow_bend_response_gain:=1.75 \
  -p use_calibrated_elbow_bias:=true \
  -p max_joint_step:=0.110 \
  -p yaw_max_step:=0.300 \
  -p elbow_max_step:=0.180 \
  -p standard_z_sign:=1.0
"
