#!/usr/bin/env bash
set -eo pipefail

REPORT="${REPORT:-/tmp/h1_robot_connection_report.txt}"
: > "$REPORT"

log() {
  echo "$@" | tee -a "$REPORT"
}

run() {
  {
    echo
    echo "===== $* ====="
    "$@" 2>&1 || true
  } | tee -a "$REPORT"
}

log "============================================================"
log " H1 ROBOT CONNECTION REPORT"
log "============================================================"
log "date:      $(date)"
log "hostname:  $(hostname)"
log "user:      $(whoami)"
log "pwd:       $(pwd)"
log "report:    $REPORT"
log "============================================================"

run uname -a
run lsb_release -a
run ip -br addr
run ip route

log
log "===== ROS 2 CHECK ====="
if [ -f /opt/ros/humble/setup.bash ]; then
  source /opt/ros/humble/setup.bash
  log "ROS Humble found"
else
  log "ROS Humble not found"
fi

if command -v ros2 >/dev/null 2>&1; then
  run ros2 --version
  run ros2 topic list -t
else
  log "ros2 command not found"
fi

log
log "===== PYTHON / UNITREE SDK CHECK ====="
run python3 --version

python3 - <<'PY' 2>&1 | tee -a "$REPORT" || true
for m in ["rclpy", "unitree_sdk2py"]:
    try:
        __import__(m)
        print(f"OK import {m}")
    except Exception as e:
        print(f"FAIL import {m}: {e}")
PY

IFACES="$(ip -o link show | awk -F': ' '{print $2}' | grep -Ev '^(lo|docker|br-|veth|virbr|zt|tailscale)' | tr '\n' ' ')"
DEFAULT_IFACE="$(ip route | awk '/default/ {print $5; exit}')"

log
log "===== DETECT NETWORK INTERFACES ====="
log "candidate interfaces: $IFACES"
log "default interface:    ${DEFAULT_IFACE:-none}"

DOMAINS="${UNITREE_TEST_DOMAINS:-0 1 42}"
if [ -n "${UNITREE_NET_IFACE:-}" ]; then
  TEST_IFACES="$UNITREE_NET_IFACE"
elif [ -n "$DEFAULT_IFACE" ]; then
  TEST_IFACES="$DEFAULT_IFACE $IFACES"
else
  TEST_IFACES="$IFACES"
fi
TEST_IFACES="$(echo $TEST_IFACES | tr ' ' '\n' | awk 'NF && !seen[$0]++' | tr '\n' ' ')"

log
log "===== UNITREE rt/lowstate SCAN ====="
log "Only subscribes to rt/lowstate. No motor commands are sent."

for domain in $DOMAINS; do
  for iface in $TEST_IFACES; do
    log
    log "--- test sdk2py: domain=$domain iface=$iface ---"
    timeout 6 python3 - "$domain" "$iface" <<'PY' 2>&1 | tee -a "$REPORT" || true
import sys, time
domain = int(sys.argv[1])
iface = sys.argv[2]

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
except Exception as e:
    print("IMPORT_FAIL", e)
    raise SystemExit(0)

try:
    print(f"ChannelFactoryInitialize({domain}, {iface!r})")
    ChannelFactoryInitialize(domain, iface)

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init()

    for _ in range(40):
        msg = sub.Read()
        if msg is not None:
            print("OK_LOWSTATE")
            print("tick:", getattr(msg, "tick", None))
            print("motor[12].q right_shoulder_pitch:", msg.motor_state[12].q)
            print("motor[13].q right_shoulder_roll :", msg.motor_state[13].q)
            print("motor[14].q right_shoulder_yaw  :", msg.motor_state[14].q)
            print("motor[15].q right_elbow         :", msg.motor_state[15].q)
            print("motor[16].q left_shoulder_pitch :", msg.motor_state[16].q)
            print("motor[19].q left_elbow          :", msg.motor_state[19].q)
            raise SystemExit(0)
        time.sleep(0.1)

    print("NO_LOWSTATE")
except Exception as e:
    print("ERROR", repr(e))
PY
  done
done

log
log "===== SUMMARY ====="
log "If OK_LOWSTATE was found, use:"
log "  export UNITREE_DOMAIN_ID=<domain>"
log "  export UNITREE_NET_IFACE=<iface>"
log
log "Report saved to: $REPORT"
log "============================================================"
