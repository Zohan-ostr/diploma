#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

export INPUT_TOPIC="/upper_body/command_geom"
export TEST_MODE="none"

# Fast simulation response
export ARM_VELOCITY_LIMIT="250.0"
export KP_LOW="1300.0"
export KD_LOW="25.0"
export COMMAND_TIMEOUT_SEC="0.7"

export ROS_DOMAIN_ID_VALUE="42"
export ROS_LOCALHOST_ONLY_VALUE="0"

export UNITREE_NET_IFACE="${UNITREE_NET_IFACE:-$(ip route | awk '/default/ {print $5; exit}')}"
if [ -z "${UNITREE_NET_IFACE:-}" ]; then
  export UNITREE_NET_IFACE="lo"
fi

docker exec h1_camera_pipeline bash -lc "pkill -f '[s]im_upper_body_to_lowcmd_bridge.py' || true" >/dev/null 2>&1 || true

bash scripts/host_exec_h1_bridge.sh
