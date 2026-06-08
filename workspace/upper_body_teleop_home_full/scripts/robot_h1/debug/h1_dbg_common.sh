#!/usr/bin/env bash

# Common helpers for real H1 debug scripts.
# This file is sourced by h1_dbg_*.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

LOG_BASE="$PROJECT_DIR/logs/robot_h1_debug"
RUN_ID_FILE="$LOG_BASE/current_run_id"

mkdir -p "$LOG_BASE"

SCRIPT_NAME="$(basename "$0")"

if [ "$SCRIPT_NAME" = "h1_dbg_0_preflight.sh" ] || [ ! -f "$RUN_ID_FILE" ]; then
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
  echo "$RUN_ID" > "$RUN_ID_FILE"
else
  RUN_ID="$(cat "$RUN_ID_FILE" 2>/dev/null || date +%Y%m%d_%H%M%S)"
fi

LOG_ROOT="$LOG_BASE/$RUN_ID"
mkdir -p "$LOG_ROOT"

ln -sfn "$LOG_ROOT" "$LOG_BASE/latest"

ROS_DOMAIN_ID_VALUE="${ROS_DOMAIN_ID_VALUE:-0}"
RMW_IMPLEMENTATION_VALUE="${RMW_IMPLEMENTATION_VALUE:-rmw_fastrtps_cpp}"
ROS_LOCALHOST_ONLY_VALUE="${ROS_LOCALHOST_ONLY_VALUE:-0}"

CAMERA_CONTAINER="${CAMERA_CONTAINER:-h1_camera_pipeline}"

H1_IP="${H1_IP:-192.168.123.161}"
H1_AUX_IP="${H1_AUX_IP:-192.168.123.164}"

UNITREE_DOMAIN_ID="${UNITREE_DOMAIN_ID:-0}"
UNITREE_NET_IFACE="${UNITREE_NET_IFACE:-}"

export ROS_DOMAIN_ID="$ROS_DOMAIN_ID_VALUE"
export RMW_IMPLEMENTATION="$RMW_IMPLEMENTATION_VALUE"
export ROS_LOCALHOST_ONLY="$ROS_LOCALHOST_ONLY_VALUE"

print_header() {
  local title="$1"
  echo
  echo "============================================================"
  echo " $title"
  echo "============================================================"
  echo "PROJECT_DIR:              $PROJECT_DIR"
  echo "LOG_ROOT:                 $LOG_ROOT"
  echo "ROS_DOMAIN_ID:            $ROS_DOMAIN_ID_VALUE"
  echo "RMW_IMPLEMENTATION:       $RMW_IMPLEMENTATION_VALUE"
  echo "ROS_LOCALHOST_ONLY:       $ROS_LOCALHOST_ONLY_VALUE"
  echo "CAMERA_CONTAINER:         $CAMERA_CONTAINER"
  echo "H1_IP:                    $H1_IP"
  echo "H1_AUX_IP:                $H1_AUX_IP"
  echo "UNITREE_DOMAIN_ID:        $UNITREE_DOMAIN_ID"
  echo "UNITREE_NET_IFACE:        ${UNITREE_NET_IFACE:-<not set>}"
  echo "============================================================"
  echo
}

log_note() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG_ROOT/debug_notes.log"
}

capture_bash() {
  local name="$1"
  shift
  local logfile="$LOG_ROOT/${name}.log"

  {
    echo "============================================================"
    echo "COMMAND: bash -lc $*"
    echo "TIME:    $(date '+%F %T')"
    echo "============================================================"
    bash -lc "$*"
    rc=$?
    echo
    echo "============================================================"
    echo "RC: $rc"
    echo "END: $(date '+%F %T')"
    echo "============================================================"
    exit "$rc"
  } 2>&1 | tee -a "$logfile"

  return "${PIPESTATUS[0]}"
}

capture_cmd() {
  local name="$1"
  shift
  local logfile="$LOG_ROOT/${name}.log"

  {
    echo "============================================================"
    echo "COMMAND: $*"
    echo "TIME:    $(date '+%F %T')"
    echo "============================================================"
    "$@"
    rc=$?
    echo
    echo "============================================================"
    echo "RC: $rc"
    echo "END: $(date '+%F %T')"
    echo "============================================================"
    exit "$rc"
  } 2>&1 | tee -a "$logfile"

  return "${PIPESTATUS[0]}"
}

container_exists() {
  docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$1"
}

ros_container_cmd() {
  local cmd="$1"

  if ! container_exists "$CAMERA_CONTAINER"; then
    echo "Container $CAMERA_CONTAINER is not running"
    return 0
  fi

  docker exec -i \
    -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID_VALUE" \
    -e RMW_IMPLEMENTATION="$RMW_IMPLEMENTATION_VALUE" \
    -e ROS_LOCALHOST_ONLY="$ROS_LOCALHOST_ONLY_VALUE" \
    "$CAMERA_CONTAINER" bash -lc "
cd /workspace 2>/dev/null || true
source /opt/ros/humble/setup.bash 2>/dev/null || true
source install/setup.bash 2>/dev/null || true
source install/upper_body_msgs/share/upper_body_msgs/local_setup.bash 2>/dev/null || true
export ROS_DOMAIN_ID=$ROS_DOMAIN_ID_VALUE
export RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION_VALUE
export ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY_VALUE
$cmd
"
}

ros_quick_snapshot() {
  local label="$1"

  {
    echo
    echo "================ ROS SNAPSHOT: $label ================"
    echo "TIME: $(date '+%F %T')"

    ros_container_cmd '
echo "--- nodes ---"
ros2 node list 2>/dev/null || true

echo
echo "--- topics ---"
ros2 topic list -t 2>/dev/null || true

echo
echo "--- /pose/landmarks ---"
ros2 topic info /pose/landmarks -v 2>/dev/null || true

echo
echo "--- /upper_body/command_geom ---"
ros2 topic info /upper_body/command_geom -v 2>/dev/null || true

echo
echo "--- /upper_body/start_calibration ---"
ros2 topic info /upper_body/start_calibration -v 2>/dev/null || true
'
  } >> "$LOG_ROOT/ros_snapshots.log" 2>&1
}

process_snapshot() {
  local label="$1"

  {
    echo
    echo "================ PROCESS SNAPSHOT: $label ================"
    echo "TIME: $(date '+%F %T')"

    echo
    echo "--- host python/project processes ---"
    ps aux | grep -E "upper_body|mediapipe|retarget|sdk2|h1_|webcam|udp_pose" | grep -v grep || true

    echo
    echo "--- docker ps ---"
    docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' 2>/dev/null || true

    if container_exists "$CAMERA_CONTAINER"; then
      echo
      echo "--- container processes: $CAMERA_CONTAINER ---"
      docker exec "$CAMERA_CONTAINER" bash -lc \
        'ps aux | grep -E "upper_body|mediapipe|retarget|sdk2|h1_|webcam|udp_pose|python3" | grep -v grep || true' \
        2>/dev/null || true
    fi
  } >> "$LOG_ROOT/process_snapshots.log" 2>&1
}

network_snapshot() {
  local label="$1"

  {
    echo
    echo "================ NETWORK SNAPSHOT: $label ================"
    echo "TIME: $(date '+%F %T')"

    echo
    echo "--- ip -br addr ---"
    ip -br addr || true

    echo
    echo "--- ip route ---"
    ip route || true

    if [ -n "$UNITREE_NET_IFACE" ]; then
      echo
      echo "--- selected interface ---"
      ip addr show "$UNITREE_NET_IFACE" || true

      echo
      echo "--- ping H1 controller $H1_IP ---"
      ping -I "$UNITREE_NET_IFACE" -c 3 -W 1 "$H1_IP" || true

      echo
      echo "--- ping H1 aux PC $H1_AUX_IP ---"
      ping -I "$UNITREE_NET_IFACE" -c 2 -W 1 "$H1_AUX_IP" || true
    else
      echo
      echo "UNITREE_NET_IFACE is not set"
    fi
  } >> "$LOG_ROOT/network_snapshots.log" 2>&1
}

start_monitor() {
  local name="$1"
  local period="${2:-5}"
  local pidfile="$LOG_ROOT/${name}_monitor.pid"

  (
    while true; do
      sleep "$period"
      ros_quick_snapshot "$name"
      process_snapshot "$name"
      network_snapshot "$name"
    done
  ) >/dev/null 2>&1 &

  echo "$!" > "$pidfile"
  log_note "Started monitor $name PID=$(cat "$pidfile")"
}

stop_monitor() {
  local name="$1"
  local pidfile="$LOG_ROOT/${name}_monitor.pid"

  if [ -f "$pidfile" ]; then
    local pid
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
  fi
}

kill_pattern_everywhere() {
  local pattern="$1"

  echo "Killing pattern everywhere: $pattern" | tee -a "$LOG_ROOT/kill.log"

  pkill -f "$pattern" 2>/dev/null || true

  docker ps --format '{{.Names}}' 2>/dev/null | while read -r c; do
    docker exec "$c" bash -lc "pkill -f '$pattern' 2>/dev/null || true" 2>/dev/null || true
  done
}

tail_debug_logs() {
  echo
  echo "============================================================"
  echo " LAST DEBUG LOGS"
  echo "============================================================"

  for f in \
    "$LOG_ROOT/debug_notes.log" \
    "$LOG_ROOT/network_snapshots.log" \
    "$LOG_ROOT/process_snapshots.log" \
    "$LOG_ROOT/ros_snapshots.log"
  do
    if [ -f "$f" ]; then
      echo
      echo "-------------------- $(basename "$f") --------------------"
      tail -n 80 "$f" || true
    fi
  done

  echo
  echo "Logs directory:"
  echo "  $LOG_ROOT"
  echo
}

wait_forever() {
  local title="$1"

  echo
  echo "============================================================"
  echo " $title IS RUNNING"
  echo "============================================================"
  echo "Логи пишутся сюда:"
  echo "  $LOG_ROOT"
  echo
  echo "Нажми Ctrl+C в этом терминале, чтобы остановить debug-обёртку."
  echo "============================================================"
  echo

  while true; do
    sleep 1
  done
}
