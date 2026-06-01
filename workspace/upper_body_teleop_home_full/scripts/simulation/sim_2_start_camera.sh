#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SIM 2. START CAMERA / MEDIAPIPE PIPELINE
# ============================================================
#
# Маршрут:
#   webcam_mediapipe_to_udp.py      на хосте
#     -> UDP 127.0.0.1:5007
#     -> udp_pose_landmarks_to_ros.py в контейнере
#     -> /pose/landmarks
#
# Retarget запускается отдельно:
#   scripts/simulation/sim_3_start_retarget.sh
# ============================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

CAMERA_ID="${1:-${CAMERA_ID:-0}}"

COMPOSE_FILE="${COMPOSE_FILE:-compose/compose.home.yaml}"
SERVICE="${SERVICE:-home-dev}"
CONTAINER_NAME="${CONTAINER_NAME:-h1_camera_pipeline}"

ROS_DOMAIN_ID_VALUE="${ROS_DOMAIN_ID_VALUE:-42}"
ROS_LOCALHOST_ONLY_VALUE="${ROS_LOCALHOST_ONLY_VALUE:-0}"
RMW_IMPLEMENTATION_VALUE="${RMW_IMPLEMENTATION_VALUE:-rmw_fastrtps_cpp}"

UDP_HOST="${UDP_HOST:-127.0.0.1}"
UDP_PORT="${UDP_PORT:-5007}"
OUTPUT_TOPIC="${OUTPUT_TOPIC:-/pose/landmarks}"

LOG_DIR="$PROJECT_DIR/logs/simulation"
HOST_CAMERA_LOG="$LOG_DIR/sim_2_host_camera.log"
HOST_PID_FILE="$LOG_DIR/sim_2_host_camera.pid"

CONTAINER_BRIDGE_LOG="/workspace/logs/simulation/sim_2_udp_pose_bridge.log"
BRIDGE_PID_FILE="/tmp/sim_2_udp_pose_bridge.pid"

echo "============================================================"
echo " SIM 2: START CAMERA / MEDIAPIPE PIPELINE"
echo "============================================================"
echo "PROJECT_DIR:        $PROJECT_DIR"
echo "CONTAINER_NAME:     $CONTAINER_NAME"
echo "CAMERA_ID:          $CAMERA_ID"
echo "UDP_HOST:           $UDP_HOST"
echo "UDP_PORT:           $UDP_PORT"
echo "OUTPUT_TOPIC:       $OUTPUT_TOPIC"
echo "ROS_DOMAIN_ID:      $ROS_DOMAIN_ID_VALUE"
echo "RMW_IMPLEMENTATION: $RMW_IMPLEMENTATION_VALUE"
echo "============================================================"

mkdir -p "$LOG_DIR"

if [ ! -f src/upper_body_teleop_runtime/webcam_mediapipe_to_udp.py ]; then
  echo "ERROR: src/upper_body_teleop_runtime/webcam_mediapipe_to_udp.py not found"
  exit 1
fi

if [ ! -f src/upper_body_teleop_runtime/udp_pose_landmarks_to_ros.py ]; then
  echo "ERROR: src/upper_body_teleop_runtime/udp_pose_landmarks_to_ros.py not found"
  exit 1
fi

echo
echo "==> Allow Docker containers to use X11..."
xhost +local:docker >/dev/null 2>&1 || true

echo
echo "==> Build Docker image with cache..."
docker compose -f "$COMPOSE_FILE" build "$SERVICE"

echo
echo "==> Ensure camera container is running..."
if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  docker compose -f "$COMPOSE_FILE" run -d \
    --name "$CONTAINER_NAME" \
    -e DISPLAY="${DISPLAY:-:0}" \
    -e QT_X11_NO_MITSHM=1 \
    -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID_VALUE" \
    -e ROS_LOCALHOST_ONLY="$ROS_LOCALHOST_ONLY_VALUE" \
    -e RMW_IMPLEMENTATION="$RMW_IMPLEMENTATION_VALUE" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    "$SERVICE" bash -lc 'sleep infinity'
fi

echo
echo "==> Start UDP -> ROS bridge inside container..."

docker exec \
  -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID_VALUE" \
  -e ROS_LOCALHOST_ONLY="$ROS_LOCALHOST_ONLY_VALUE" \
  -e RMW_IMPLEMENTATION="$RMW_IMPLEMENTATION_VALUE" \
  -e UDP_PORT="$UDP_PORT" \
  -e OUTPUT_TOPIC="$OUTPUT_TOPIC" \
  -e BRIDGE_PID_FILE="$BRIDGE_PID_FILE" \
  -e CONTAINER_BRIDGE_LOG="$CONTAINER_BRIDGE_LOG" \
  "$CONTAINER_NAME" bash -lc '
set -eo pipefail

cd /workspace
mkdir -p logs/simulation logs/colcon

source /opt/ros/humble/setup.bash

if [ ! -f install/upper_body_msgs/share/upper_body_msgs/local_setup.bash ]; then
  echo "Building upper_body_msgs..."
  colcon --log-base logs/colcon build \
    --base-paths src/upper_body_msgs \
    --packages-select upper_body_msgs
fi

source install/upper_body_msgs/share/upper_body_msgs/local_setup.bash

export PYTHONPATH=/workspace/src:/workspace/src/home_pipeline:/workspace/src/h1_robot_adapter:$PYTHONPATH

echo "Stopping old UDP -> ROS bridge safely..."

python3 - <<PY
import os
import signal

target_module = "upper_body_teleop_runtime.udp_pose_landmarks_to_ros"
target_script = "udp_pose_landmarks_to_ros.py"
my_pid = os.getpid()

for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue

    p = int(pid)
    if p == my_pid:
        continue

    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmd = [x.decode(errors="ignore") for x in f.read().split(b"\\0") if x]
    except Exception:
        continue

    is_python = bool(cmd) and ("python" in os.path.basename(cmd[0]))
    is_module = "-m" in cmd and target_module in cmd
    is_script = any(part.endswith(target_script) for part in cmd)

    if is_python and (is_module or is_script):
        print("killing old bridge:", p, " ".join(cmd))
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass
PY

rm -f "$BRIDGE_PID_FILE" "$CONTAINER_BRIDGE_LOG"

nohup python3 -m upper_body_teleop_runtime.udp_pose_landmarks_to_ros \
  --ros-args \
  -p udp_port:="$UDP_PORT" \
  -p output_topic:="$OUTPUT_TOPIC" \
  > "$CONTAINER_BRIDGE_LOG" 2>&1 &

echo "$!" > "$BRIDGE_PID_FILE"

sleep 2

echo
echo "Bridge PID:"
cat "$BRIDGE_PID_FILE"

echo
echo "Bridge log:"
sed -n "1,160p" "$CONTAINER_BRIDGE_LOG" || true

echo
echo "Pose topic after bridge start:"
ros2 topic info "$OUTPUT_TOPIC" -v || true
'

echo
echo "==> Start host webcam -> UDP..."

if [ -f "$HOST_PID_FILE" ]; then
  OLD_PID="$(cat "$HOST_PID_FILE" 2>/dev/null || true)"
  if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Stopping old host camera PID: $OLD_PID"
    kill "$OLD_PID" 2>/dev/null || true
    sleep 1
    kill -9 "$OLD_PID" 2>/dev/null || true
  fi
  rm -f "$HOST_PID_FILE"
fi

python3 - <<'PY'
import os
import signal

target_module = "upper_body_teleop_runtime.webcam_mediapipe_to_udp"
target_script = "webcam_mediapipe_to_udp.py"
my_pid = os.getpid()

for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue

    p = int(pid)
    if p == my_pid:
        continue

    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmd = [x.decode(errors="ignore") for x in f.read().split(b"\0") if x]
    except Exception:
        continue

    is_python = bool(cmd) and ("python" in os.path.basename(cmd[0]))
    is_module = "-m" in cmd and target_module in cmd
    is_script = any(part.endswith(target_script) for part in cmd)

    if is_python and (is_module or is_script):
        print("killing old host camera:", p, " ".join(cmd))
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass
PY

rm -f "$HOST_CAMERA_LOG"

PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/src/home_pipeline:$PROJECT_DIR/src/h1_robot_adapter:${PYTHONPATH:-}" \
nohup python3 -m upper_body_teleop_runtime.webcam_mediapipe_to_udp \
  --camera "$CAMERA_ID" \
  --udp_host "$UDP_HOST" \
  --udp_port "$UDP_PORT" \
  > "$HOST_CAMERA_LOG" 2>&1 &

echo "$!" > "$HOST_PID_FILE"

sleep 4

echo
echo "Host camera PID:"
cat "$HOST_PID_FILE"

echo
echo "Host camera log:"
sed -n "1,180p" "$HOST_CAMERA_LOG" || true

echo
echo "==> Final check /pose/landmarks..."
docker exec \
  -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID_VALUE" \
  -e ROS_LOCALHOST_ONLY="$ROS_LOCALHOST_ONLY_VALUE" \
  -e RMW_IMPLEMENTATION="$RMW_IMPLEMENTATION_VALUE" \
  "$CONTAINER_NAME" bash -lc '
cd /workspace
source /opt/ros/humble/setup.bash
source install/upper_body_msgs/share/upper_body_msgs/local_setup.bash 2>/dev/null || true
ros2 topic info /pose/landmarks -v || true
'

echo
echo "============================================================"
echo " SIM 2 DONE"
echo "============================================================"
echo "Logs:"
echo "  $HOST_CAMERA_LOG"
echo "  logs/simulation/sim_2_udp_pose_bridge.log"
echo

# ============================================================
# KEEP SIM 2 ALIVE
# ============================================================
# Скрипт должен оставаться в терминале.
# Ctrl+C останавливает:
#   - host webcam -> UDP;
#   - UDP -> ROS bridge внутри контейнера.
# ============================================================

cleanup_sim2() {
  echo
  echo "==> SIM 2 cleanup..."

  if [ -f "$HOST_PID_FILE" ]; then
    HOST_PID="$(cat "$HOST_PID_FILE" 2>/dev/null || true)"
    if [ -n "$HOST_PID" ] && kill -0 "$HOST_PID" 2>/dev/null; then
      echo "Stopping host camera PID: $HOST_PID"
      kill "$HOST_PID" 2>/dev/null || true
      sleep 1
      kill -9 "$HOST_PID" 2>/dev/null || true
    fi
    rm -f "$HOST_PID_FILE"
  fi

  docker exec "$CONTAINER_NAME" bash -lc '
    if [ -f /tmp/sim_2_udp_pose_bridge.pid ]; then
      PID="$(cat /tmp/sim_2_udp_pose_bridge.pid 2>/dev/null || true)"
      if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        echo "Stopping UDP -> ROS bridge PID: $PID"
        kill "$PID" 2>/dev/null || true
        sleep 1
        kill -9 "$PID" 2>/dev/null || true
      fi
      rm -f /tmp/sim_2_udp_pose_bridge.pid
    fi
  ' 2>/dev/null || true

  echo "SIM 2 stopped."
}

trap cleanup_sim2 INT TERM EXIT

echo
echo "============================================================"
echo " SIM 2 IS RUNNING"
echo "============================================================"
echo "Press Ctrl+C here to stop camera pipeline."
echo "Host camera log:"
echo "  $HOST_CAMERA_LOG"
echo "Bridge log:"
echo "  logs/simulation/sim_2_udp_pose_bridge.log"
echo "============================================================"

while true; do
  sleep 1
done
