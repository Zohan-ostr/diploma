#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SIM 4. START SDK2 SENDER FOR MUJOCO
# ============================================================
#
# Назначение:
#   /upper_body/command_geom
#     -> h1_robot_adapter.h1_sdk2py_upper_body_bridge
#     -> rt/lowcmd
#     -> Unitree MuJoCo
#
# Это старый рабочий SDK2 Python bridge.
# ============================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

CONTAINER_NAME="${CONTAINER_NAME:-h1_mujoco_sim}"

ROS_DOMAIN_ID_VALUE="${ROS_DOMAIN_ID_VALUE:-42}"
ROS_LOCALHOST_ONLY_VALUE="${ROS_LOCALHOST_ONLY_VALUE:-0}"
RMW_IMPLEMENTATION_VALUE="${RMW_IMPLEMENTATION_VALUE:-rmw_fastrtps_cpp}"

UNITREE_DOMAIN_ID_VALUE="${UNITREE_DOMAIN_ID_VALUE:-42}"
UNITREE_NET_IFACE_VALUE="${UNITREE_NET_IFACE_VALUE:-lo}"

INPUT_TOPIC="${INPUT_TOPIC:-/upper_body/command_geom}"

CONTROL_HZ="${CONTROL_HZ:-250.0}"
KP_LOW="${KP_LOW:-700.0}"
KD_LOW="${KD_LOW:-12.0}"
KP_HIGH="${KP_HIGH:-300.0}"
KD_HIGH="${KD_HIGH:-5.0}"
ARM_VELOCITY_LIMIT="${ARM_VELOCITY_LIMIT:-3.0}"
COMMAND_TIMEOUT_SEC="${COMMAND_TIMEOUT_SEC:-0.5}"
TEST_MODE="${TEST_MODE:-none}"

LOG_PATH="${LOG_PATH:-/workspace/logs/simulation/sim_sdk2_sender.log}"
PID_PATH="${PID_PATH:-/tmp/sim_sdk2_sender.pid}"

echo "============================================================"
echo " SIM 4: START SDK2 SENDER"
echo "============================================================"
echo "CONTAINER_NAME:       $CONTAINER_NAME"
echo "INPUT_TOPIC:          $INPUT_TOPIC"
echo "SDK_CMD_TOPIC:        rt/lowcmd"
echo "SDK_STATE_TOPIC:      rt/lowstate"
echo "ROS_DOMAIN_ID:        $ROS_DOMAIN_ID_VALUE"
echo "UNITREE_DOMAIN_ID:    $UNITREE_DOMAIN_ID_VALUE"
echo "UNITREE_NET_IFACE:    $UNITREE_NET_IFACE_VALUE"
echo "CONTROL_HZ:           $CONTROL_HZ"
echo "KP_LOW/KD_LOW:        $KP_LOW / $KD_LOW"
echo "KP_HIGH/KD_HIGH:      $KP_HIGH / $KD_HIGH"
echo "ARM_VELOCITY_LIMIT:   $ARM_VELOCITY_LIMIT"
echo "COMMAND_TIMEOUT_SEC:  $COMMAND_TIMEOUT_SEC"
echo "LOG_PATH:             $LOG_PATH"
echo "============================================================"

CID="$(docker ps --filter "name=^/${CONTAINER_NAME}$" --format "{{.ID}}" | head -n 1)"

if [ -z "$CID" ]; then
  echo "ERROR: container is not running: $CONTAINER_NAME"
  echo "Start MuJoCo first:"
  echo "  bash scripts/simulation/sim_1_start_mujoco.sh"
  exit 1
fi

mkdir -p logs/simulation logs/colcon

docker exec \
  -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID_VALUE" \
  -e ROS_LOCALHOST_ONLY="$ROS_LOCALHOST_ONLY_VALUE" \
  -e RMW_IMPLEMENTATION="$RMW_IMPLEMENTATION_VALUE" \
  -e UNITREE_DOMAIN_ID="$UNITREE_DOMAIN_ID_VALUE" \
  -e UNITREE_NET_IFACE="$UNITREE_NET_IFACE_VALUE" \
  -e INPUT_TOPIC="$INPUT_TOPIC" \
  -e CONTROL_HZ="$CONTROL_HZ" \
  -e KP_LOW="$KP_LOW" \
  -e KD_LOW="$KD_LOW" \
  -e KP_HIGH="$KP_HIGH" \
  -e KD_HIGH="$KD_HIGH" \
  -e ARM_VELOCITY_LIMIT="$ARM_VELOCITY_LIMIT" \
  -e COMMAND_TIMEOUT_SEC="$COMMAND_TIMEOUT_SEC" \
  -e TEST_MODE="$TEST_MODE" \
  -e LOG_PATH="$LOG_PATH" \
  -e PID_PATH="$PID_PATH" \
  "$CID" bash -lc '
set -eo pipefail

cd /workspace

mkdir -p logs/simulation logs/colcon

source /opt/ros/humble/setup.bash
source install/upper_body_msgs/share/upper_body_msgs/local_setup.bash 2>/dev/null || true

export PYTHONPATH=/workspace/src:/workspace/src/home_pipeline:/workspace/src/h1_robot_adapter:$PYTHONPATH

echo
echo "============================================================"
echo " INSIDE SIM SDK2 SENDER CONTAINER"
echo "============================================================"
echo "pwd:                  $(pwd)"
echo "ROS_DOMAIN_ID:        $ROS_DOMAIN_ID"
echo "ROS_LOCALHOST_ONLY:   $ROS_LOCALHOST_ONLY"
echo "RMW_IMPLEMENTATION:   $RMW_IMPLEMENTATION"
echo "UNITREE_DOMAIN_ID:    $UNITREE_DOMAIN_ID"
echo "UNITREE_NET_IFACE:    $UNITREE_NET_IFACE"
echo "INPUT_TOPIC:          $INPUT_TOPIC"
echo "CONTROL_HZ:           $CONTROL_HZ"
echo "KP_LOW/KD_LOW:        $KP_LOW / $KD_LOW"
echo "KP_HIGH/KD_HIGH:      $KP_HIGH / $KD_HIGH"
echo "ARM_VELOCITY_LIMIT:   $ARM_VELOCITY_LIMIT"
echo "COMMAND_TIMEOUT_SEC:  $COMMAND_TIMEOUT_SEC"
echo "LOG_PATH:             $LOG_PATH"
echo "PID_PATH:             $PID_PATH"
echo "============================================================"
echo

echo "Stopping old SDK2 sender/bridge..."

python3 - <<PY
import os
import signal

targets = {
    "upper_body_teleop_runtime.sim_sdk2_upper_body_sender",
    "h1_robot_adapter.h1_sdk2py_upper_body_bridge",
}

my_pid = os.getpid()

for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue

    p = int(pid)
    if p == my_pid:
        continue

    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            parts = [x.decode(errors="ignore") for x in f.read().split(b"\\0") if x]
    except Exception:
        continue

    kill_it = False
    for i, part in enumerate(parts):
        if part == "-m" and i + 1 < len(parts) and parts[i + 1] in targets:
            kill_it = True
        if part in targets:
            kill_it = True

    if kill_it:
        print("killing old process:", pid, " ".join(parts))
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass
PY

rm -f "$PID_PATH"

echo
echo "Checking old working SDK2 bridge module..."
python3 - <<PY
import h1_robot_adapter.h1_sdk2py_upper_body_bridge
print("h1_sdk2py_upper_body_bridge module OK")
PY

rm -f "$LOG_PATH"

echo
echo "Starting old working H1 SDK2 bridge in background..."

nohup python3 -m h1_robot_adapter.h1_sdk2py_upper_body_bridge \
  --ros-args \
  -p input_topic:="$INPUT_TOPIC" \
  -p command_timeout_sec:="$COMMAND_TIMEOUT_SEC" \
  -p kp_arm:="$KP_LOW" \
  -p kd_arm:="$KD_LOW" \
  -p kp_body:="$KP_HIGH" \
  -p kd_body:="$KD_HIGH" \
  -p arm_velocity_limit:="$ARM_VELOCITY_LIMIT" \
  -p control_hz:="$CONTROL_HZ" \
  -p test_mode:="$TEST_MODE" \
  > "$LOG_PATH" 2>&1 &

NEW_PID="$!"
echo "$NEW_PID" > "$PID_PATH"

echo "Started PID: $NEW_PID"

sleep 4

echo
echo "Sender process check:"
if kill -0 "$NEW_PID" 2>/dev/null; then
  echo "sender is running"
else
  echo "ERROR: sender exited"
  echo
  echo "Sender log:"
  sed -n "1,260p" "$LOG_PATH" || true
  exit 1
fi

echo
echo "Sender log:"
sed -n "1,240p" "$LOG_PATH" || true

echo
echo "ROS input topic info:"
ros2 topic info "$INPUT_TOPIC" -v || true

echo
echo "Done."
'

echo
echo "============================================================"
echo " SIM 4 DONE"
echo "============================================================"
echo "SDK2 sender is running inside container: $CONTAINER_NAME"
echo "Log:"
echo "  logs/simulation/sim_sdk2_sender.log"
echo
