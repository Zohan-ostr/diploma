#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-h1_camera_pipeline}"

ROBOT_IP="${ROBOT_IP:-192.168.123.162}"
NET_IFACE="${NET_IFACE:-}"

if [ -z "$NET_IFACE" ]; then
  NET_IFACE="$(ip route get "$ROBOT_IP" | awk "/dev/ {for(i=1;i<=NF;i++) if(\$i==\"dev\") print \$(i+1)}" | head -1)"
fi

if [ -z "$NET_IFACE" ]; then
  NET_IFACE="enx00e04c36022c"
fi

SDK_CMD_TOPIC="${SDK_CMD_TOPIC:-rt/lowcmd}"
SDK_STATE_TOPIC="${SDK_STATE_TOPIC:-rt/lowstate}"
UNITREE_DOMAIN_ID="${UNITREE_DOMAIN_ID:-0}"
UDP_PORT="${UDP_PORT:-50051}"

echo "============================================================"
echo " LAPTOP 4: START SDK2PY SENDER TO REAL H1"
echo "============================================================"
echo "CONTAINER_NAME:  $CONTAINER_NAME"
echo "NET_IFACE:       $NET_IFACE"
echo "UNITREE_DOMAIN:  $UNITREE_DOMAIN_ID"
echo "SDK_CMD_TOPIC:   $SDK_CMD_TOPIC"
echo "SDK_STATE_TOPIC: $SDK_STATE_TOPIC"
echo "UDP_PORT:        $UDP_PORT"
echo "KP_ARM:          25.0"
echo "KD_ARM:          1.5"
echo "============================================================"

docker exec -it \
  -e NET_IFACE="$NET_IFACE" \
  -e UNITREE_DOMAIN_ID="$UNITREE_DOMAIN_ID" \
  -e SDK_CMD_TOPIC="$SDK_CMD_TOPIC" \
  -e SDK_STATE_TOPIC="$SDK_STATE_TOPIC" \
  -e UDP_PORT="$UDP_PORT" \
  "$CONTAINER_NAME" bash -lc '
set -e

cd /workspace

python3 - <<PY
try:
    import unitree_sdk2py
    print("OK import unitree_sdk2py")
except Exception as e:
    print("FAIL import unitree_sdk2py:", e)
    raise
PY

python3 /workspace/scripts/robot_h1/laptop_sdk2py_udp_to_h1_sender.py \
  --net_iface "${NET_IFACE}" \
  --domain "${UNITREE_DOMAIN_ID}" \
  --sdk_cmd_topic "${SDK_CMD_TOPIC}" \
  --sdk_state_topic "${SDK_STATE_TOPIC}" \
  --udp_bind_host "0.0.0.0" \
  --udp_port "${UDP_PORT}" \
  --rate_hz 250.0 \
  --kp_arm 25.0 \
  --kd_arm 1.5 \
  --max_step_rad 0.012 \
  --yaw_max_step_rad 0.020 \
  --elbow_max_step_rad 0.025 \
  --command_timeout_sec 0.35 \
  --startup_tpose_sec -1.0
'
