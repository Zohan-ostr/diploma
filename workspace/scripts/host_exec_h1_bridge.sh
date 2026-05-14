#!/usr/bin/env bash
set -eo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-h1_mujoco_sim}"

ROS_DOMAIN_ID_VALUE="${ROS_DOMAIN_ID_VALUE:-42}"
ROS_LOCALHOST_ONLY_VALUE="${ROS_LOCALHOST_ONLY_VALUE:-0}"

INPUT_TOPIC="${INPUT_TOPIC:-/upper_body/command_geom}"
TEST_MODE="${TEST_MODE:-none}"

ARM_VELOCITY_LIMIT="${ARM_VELOCITY_LIMIT:-120.0}"
KP_LOW="${KP_LOW:-700.0}"
KD_LOW="${KD_LOW:-12.0}"
COMMAND_TIMEOUT_SEC="${COMMAND_TIMEOUT_SEC:-0.5}"

UNITREE_NET_IFACE="${UNITREE_NET_IFACE:-wlp0s20f3}"
UNITREE_DOMAIN_ID="${UNITREE_DOMAIN_ID:-42}"

echo "============================================================"
echo " START H1 BRIDGE"
echo "============================================================"
echo "CONTAINER_NAME:       $CONTAINER_NAME"
echo "INPUT_TOPIC:          $INPUT_TOPIC"
echo "TEST_MODE:            $TEST_MODE"
echo "ARM_VELOCITY_LIMIT:   $ARM_VELOCITY_LIMIT"
echo "KP_LOW:               $KP_LOW"
echo "KD_LOW:               $KD_LOW"
echo "COMMAND_TIMEOUT_SEC:  $COMMAND_TIMEOUT_SEC"
echo "UNITREE_NET_IFACE:    $UNITREE_NET_IFACE"
echo "UNITREE_DOMAIN_ID:    $UNITREE_DOMAIN_ID"
echo "ROS_DOMAIN_ID:        $ROS_DOMAIN_ID_VALUE"
echo "============================================================"
echo

CID="$(docker ps --filter "name=^/${CONTAINER_NAME}$" --format "{{.ID}}" | head -n 1)"

if [ -z "$CID" ]; then
  echo "ERROR: container not running: $CONTAINER_NAME"
  docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}"
  exit 1
fi

echo "Found container:"
docker ps --filter "id=$CID" --format "table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}"
echo

TMP_INNER="$(mktemp /tmp/run_h1_bridge_inside.XXXXXX.sh)"

cat > "$TMP_INNER" <<'INNER'
#!/usr/bin/env bash
set -eo pipefail

cd /workspace

source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash 2>/dev/null || true

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"

export INPUT_TOPIC="${INPUT_TOPIC:-/upper_body/command_geom}"
export TEST_MODE="${TEST_MODE:-none}"
export ARM_VELOCITY_LIMIT="${ARM_VELOCITY_LIMIT:-120.0}"
export KP_LOW="${KP_LOW:-700.0}"
export KD_LOW="${KD_LOW:-12.0}"
export COMMAND_TIMEOUT_SEC="${COMMAND_TIMEOUT_SEC:-0.5}"
export UNITREE_NET_IFACE="${UNITREE_NET_IFACE:-wlp0s20f3}"
export UNITREE_DOMAIN_ID="${UNITREE_DOMAIN_ID:-42}"
UNITREE_DOMAIN_ID="${UNITREE_DOMAIN_ID:-42}"

BRIDGE_LOG="/tmp/h1_bridge.log"
BRIDGE_PID="/tmp/h1_bridge.pid"

echo "============================================================"
echo " INSIDE H1 BRIDGE CONTAINER"
echo "============================================================"
echo "hostname:             $(hostname)"
echo "pwd:                  $(pwd)"
echo "ROS_DOMAIN_ID:        $ROS_DOMAIN_ID"
echo "ROS_LOCALHOST_ONLY:   $ROS_LOCALHOST_ONLY"
echo "INPUT_TOPIC:          $INPUT_TOPIC"
echo "TEST_MODE:            $TEST_MODE"
echo "ARM_VELOCITY_LIMIT:   $ARM_VELOCITY_LIMIT"
echo "KP_LOW:               $KP_LOW"
echo "KD_LOW:               $KD_LOW"
echo "COMMAND_TIMEOUT_SEC:  $COMMAND_TIMEOUT_SEC"
echo "UNITREE_NET_IFACE:    $UNITREE_NET_IFACE"
echo "UNITREE_DOMAIN_ID:    $UNITREE_DOMAIN_ID"
echo "============================================================"
echo

echo "Killing old bridge by PID file only..."
if [ -f "$BRIDGE_PID" ]; then
  OLD_PID="$(cat "$BRIDGE_PID" 2>/dev/null || true)"
  if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Killing old bridge PID: $OLD_PID"
    kill "$OLD_PID" 2>/dev/null || true
    sleep 1
    kill -9 "$OLD_PID" 2>/dev/null || true
  fi
  rm -f "$BRIDGE_PID"
fi

echo
echo "Checking package executables:"
ros2 pkg executables h1_robot_adapter || true

echo
echo "Checking input topic:"
ros2 topic info "$INPUT_TOPIC" -v || true

echo
echo "Checking lowstate:"
ros2 topic info /lowstate -v || true

echo
echo "Starting bridge in background..."
rm -f "$BRIDGE_LOG"

nohup ros2 run h1_robot_adapter h1_sdk2py_upper_body_bridge \
  --ros-args \
  -p input_topic:="$INPUT_TOPIC" \
  -p test_mode:="$TEST_MODE" \
  -p arm_velocity_limit:="$ARM_VELOCITY_LIMIT" \
  -p kp_low:="$KP_LOW" \
  -p kd_low:="$KD_LOW" \
  -p command_timeout_sec:="$COMMAND_TIMEOUT_SEC" \
  > "$BRIDGE_LOG" 2>&1 &

echo "$!" > "$BRIDGE_PID"

sleep 3

echo
echo "Bridge PID:"
cat "$BRIDGE_PID" || true

echo
echo "Processes:"
ps aux | grep -E "h1_sdk2py|h1_unitree|arm_sdk_to_lowcmd" | grep -v grep || true

echo
echo "Bridge log first lines:"
sed -n '1,200p' "$BRIDGE_LOG" || true

echo
echo "Topic /lowcmd info:"
ros2 topic info /lowcmd -v || true

echo
echo "Try /lowcmd hz for 5 seconds:"
timeout 6 ros2 topic hz /lowcmd || true

echo
echo "Done inside container."
INNER

chmod +x "$TMP_INNER"

echo "Copying inner bridge runner into container..."
docker cp "$TMP_INNER" "$CID:/tmp/run_h1_bridge_inside.sh"
rm -f "$TMP_INNER"

echo "Running bridge inside container..."
docker exec \
  -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID_VALUE" \
  -e ROS_LOCALHOST_ONLY="$ROS_LOCALHOST_ONLY_VALUE" \
  -e INPUT_TOPIC="$INPUT_TOPIC" \
  -e TEST_MODE="$TEST_MODE" \
  -e ARM_VELOCITY_LIMIT="$ARM_VELOCITY_LIMIT" \
  -e KP_LOW="$KP_LOW" \
  -e KD_LOW="$KD_LOW" \
  -e COMMAND_TIMEOUT_SEC="$COMMAND_TIMEOUT_SEC" \
  -e UNITREE_NET_IFACE="$UNITREE_NET_IFACE" \
  -e UNITREE_DOMAIN_ID="$UNITREE_DOMAIN_ID" \
  -it "$CID" \
  bash /tmp/run_h1_bridge_inside.sh

echo
echo "============================================================"
echo " HOST DONE"
echo "============================================================"
echo "To watch bridge log:"
echo "  docker exec -it $CONTAINER_NAME bash -lc 'tail -f /tmp/h1_bridge.log'"
