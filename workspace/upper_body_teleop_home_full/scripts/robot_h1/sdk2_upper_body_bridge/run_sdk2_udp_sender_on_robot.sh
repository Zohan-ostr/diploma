#!/usr/bin/env bash
set -euo pipefail

cd "${HOME}/WS_OZ/diploma"

export LD_LIBRARY_PATH=${HOME}/WS_OZ/unitree_sdk2/thirdparty/lib/x86_64:/usr/local/lib:${LD_LIBRARY_PATH:-}

IFACE=${IFACE:-eth0}
KP=${KP:-25.0}
KD=${KD:-1.5}
MAX_STEP=${MAX_STEP:-0.012}
TIMEOUT_SEC=${TIMEOUT_SEC:-0.35}
UDP_PORT=${UDP_PORT:-50051}

./install/sdk2_h1_upper_body_bridge/lib/sdk2_h1_upper_body_bridge/sdk2_h1_udp_lowcmd_sender \
  "$IFACE" "$KP" "$KD" "$MAX_STEP" "$TIMEOUT_SEC" "$UDP_PORT"
